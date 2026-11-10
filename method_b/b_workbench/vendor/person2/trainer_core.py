#!/usr/bin/env python3
"""Single-NVIDIA-GPU QLoRA pilot. See README before running real experiments.
Only standard-library imports are needed for --help, subset and unit tests.
GPU/model integration MUST be validated on the user's machine.
"""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

MODEL = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
COUNT_FIELDS = ('prompt_tokens', 'completion_tokens', 'supervised_tokens', 'total_tokens')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pandas as pd
        rows = pd.read_parquet(path).to_dict('records')
    elif path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        raise ValueError('Use a self-contained .jsonl manifest or .parquet; ID-only text is insufficient.')
    if not rows:
        raise ValueError('Empty dataset')
    for r in rows:
        for field in ('row_id', 'problem_group_id', 'dataset_id', 'problem', 'completion', 'final_answer'):
            if field not in r or not isinstance(r[field], str) or not r[field].strip():
                raise ValueError(f'Missing/non-string/empty {field}; agree an explicit schema conversion with Person 1.')
        for field in COUNT_FIELDS:
            if field not in r or isinstance(r[field], bool) or int(r[field]) != r[field] or r[field] < 0:
                raise ValueError(f'Invalid/missing {field}: {r["row_id"]}')
    return rows

def source(r):
    name = r['dataset_id'].lower()
    if 'orca' in name:
        return 'orca'
    if 'numina' in name:
        return 'numina'
    raise ValueError(f'Unknown dataset_id {r["dataset_id"]!r}; adapt source() explicitly.')

def check_unique(rows, allow_repeats=False):
    # Intentional repeated exposures are permitted only with byte-equivalent row contents.
    ids, groups = {}, {}
    for r in rows:
        rid, gid = r['row_id'], r['problem_group_id']
        if gid in groups and groups[gid] != rid:
            raise ValueError(f'Two representatives of problem group {gid}')
        if rid in ids and (not allow_repeats or ids[rid] != r):
            raise ValueError(f'Duplicate/conflicting row_id {rid}')
        ids[rid], groups[gid] = r, rid

def load_contract(template_path, policy_path):
    # JSON preserves every newline and trailing space in Person 1's exact template.
    obj = read_json(template_path)
    if set(obj) != {'template'} or not isinstance(obj['template'], str):
        raise ValueError('Template JSON must contain exactly one string field: template')
    template = obj['template']
    if template.count('{problem}') != 1:
        raise ValueError('Template must contain exactly one literal {problem} placeholder.')
    policy = read_json(policy_path)
    required = {'confirmed_with_person1', 'add_special_tokens', 'append_eos', 'max_total_tokens', 'provenance'}
    if set(policy) != required or policy['confirmed_with_person1'] is not True or not policy['provenance']:
        raise ValueError('Obtain the exact formatting/tokenization policy from Person 1 and confirm it in JSON.')
    if policy['add_special_tokens'] is not False or policy['append_eos'] is not True:
        raise ValueError('This implementation supports add_special_tokens=false + one explicitly appended EOS. Reconcile other policies before use.')
    if not isinstance(policy['max_total_tokens'], int) or policy['max_total_tokens'] < 2:
        raise ValueError('Invalid context cap')
    return template, policy

def encode_row(r, tokenizer, template, cap):
    prompt = template.replace('{problem}', r['problem'])
    completion = r['completion']  # Never strip, rewrite final markers or add reasoning.
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    joined = tokenizer.encode(prompt + completion, add_special_tokens=False)
    if not pids:
        raise ValueError(f'{r["row_id"]}: empty prompt token sequence')
    boundary_method = 'prefix'
    prompt_count = len(pids)
    if joined[:len(pids)] != pids:
        # Match Person 1 cell 25: full-text offsets may recover a valid boundary.
        offset_enc = tokenizer(prompt + completion, add_special_tokens=False, return_offsets_mapping=True)
        offsets = offset_enc['offset_mapping']
        boundary = len(prompt)
        if any(left < boundary < right for left, right in offsets):
            raise ValueError(f'{r["row_id"]}: token crosses prompt/completion boundary; inspect with Person 1.')
        prompt_count = sum(right <= boundary for left, right in offsets)
        joined = offset_enc['input_ids']
        if not 0 < prompt_count < len(joined):
            raise ValueError(f'{r["row_id"]}: invalid offset-based boundary')
        # Reject ambiguous/noncontiguous offsets rather than supervise prompt characters.
        if any(right > boundary for left, right in offsets[:prompt_count]) or any(left < boundary for left, right in offsets[prompt_count:]):
            raise ValueError(f'{r["row_id"]}: ambiguous offset ordering at boundary')
        boundary_method = 'offsets'
    cids = joined[prompt_count:]
    eos = tokenizer.eos_token_id
    if eos is None or not cids:
        raise ValueError(f'{r["row_id"]}: missing EOS or empty completion')
    special = set(getattr(tokenizer, 'all_special_ids', [eos]))
    if any(t in special for t in joined):
        raise ValueError(f'{r["row_id"]}: unexpected special tokens inside raw text; inspect with Person 1.')
    ids = joined + [eos]
    labels = [-100] * prompt_count + cids + [eos]
    if len(ids) > cap:
        raise ValueError(f'{r["row_id"]}: {len(ids)} tokens exceeds {cap}; no truncation is allowed.')
    counts = dict(prompt_tokens=prompt_count, completion_tokens=len(cids),
                  supervised_tokens=sum(x != -100 for x in labels[1:]), total_tokens=len(ids))
    if counts['supervised_tokens'] != len(cids) + 1:
        raise ValueError('Shifted target accounting failed')
    return dict(input_ids=ids, labels=labels, counts=counts, boundary_method=boundary_method)

def fingerprint(args):
    return dict(manifest_sha256=sha(args.data), template_sha256=sha(args.template),
                policy_sha256=sha(args.policy), model=MODEL, revision=REVISION,
                code_sha256=sha(__file__))

def csv_write(path, rows):
    if not rows:
        raise ValueError('No CSV records')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def reconcile(rows, tokenizer, template, policy):
    records, encoded = [], []
    for r in rows:
        rec = dict(row_id=r['row_id'], problem_group_id=r['problem_group_id'], dataset_id=r['dataset_id'])
        try:
            e = encode_row(r, tokenizer, template, policy['max_total_tokens'])
            for field in COUNT_FIELDS:
                rec['person1_' + field] = int(r[field])
                rec['person2_' + field] = e['counts'][field]
            rec['actual_supervised_tokens'] = e['counts']['supervised_tokens']
            rec['boundary_method'] = e['boundary_method']
            rec['match'] = all(rec['person1_' + f] == rec['person2_' + f] for f in COUNT_FIELDS)
            rec['error'] = '' if rec['match'] else 'Token-count mismatch'
            encoded.append(e)
        except ValueError as exc:
            for field in COUNT_FIELDS:
                rec['person1_' + field], rec['person2_' + field] = int(r[field]), ''
            rec['actual_supervised_tokens'], rec['boundary_method'] = '', 'failed'
            rec['match'], rec['error'] = False, str(exc)
            encoded.append(None)
        records.append(rec)
    return records, encoded

def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, use_fast=True)

def verify(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.data); check_unique(rows, args.allow_repeats)
    template, policy = load_contract(args.template, args.policy)
    tok = get_tokenizer()
    records, enc = reconcile(rows, tok, template, policy)
    csv_write(out / 'person2_token_verification.csv', records)
    passed = all(r['match'] for r in records)
    summary = fingerprint(args)
    summary.update(passed=passed, rows=len(rows), allow_repeats=args.allow_repeats,
                   unique_examples=len({r['row_id'] for r in rows}), mismatches=sum(not r['match'] for r in records),
                   template_utf8_sha256=hashlib.sha256(template.encode('utf-8')).hexdigest(),
                   actual_transformers_version=importlib.metadata.version('transformers'),
                   scope_note='Verification covers only this exact manifest, not the whole Person 1 working pool.')
    if passed:
        summary['processed_supervised_tokens_one_pass'] = sum(e['counts']['supervised_tokens'] for e in enc)
        summary['source_supervised_tokens'] = dict(collections.Counter())
        for r, e in zip(rows, enc):
            k = source(r)
            summary['source_supervised_tokens'][k] = summary['source_supervised_tokens'].get(k, 0) + e['counts']['supervised_tokens']
        csv_write(out / 'mask_preview.csv', [dict(position=i, token_id=t, token=tok.convert_ids_to_tokens(t),
                                                 label=enc[0]['labels'][i], supervised=enc[0]['labels'][i] != -100)
                                             for i, t in enumerate(enc[0]['input_ids'])])
    dump(out / 'verification.json', summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit('STOP: send the verification CSV to Person 1. Do not change counts to make the check pass.')


def check_nesting(args):
    previous_ids = []
    result = []
    for path in args.data:
        rows = load_rows(path); check_unique(rows)
        ids = [r['row_id'] for r in rows]
        if len(ids) < len(previous_ids) or ids[:len(previous_ids)] != previous_ids:
            raise ValueError(f'{path}: not an ordered prefix extension of previous manifest')
        counts = collections.Counter(source(r) for r in rows)
        if counts['orca'] != counts['numina']:
            raise ValueError(f'{path}: not balanced by examples')
        result.append(dict(path=str(path), rows=len(rows), source_counts=dict(counts), sha256=sha(path)))
        previous_ids = ids
    print(json.dumps(result, indent=2))

def subset(args):
    rows = load_rows(args.data); check_unique(rows)
    chosen = []
    for k in ('orca', 'numina'):
        part = [r for r in rows if source(r) == k][:16]
        if len(part) != 16:
            raise ValueError(f'Need at least 16 {k} examples')
        chosen.extend(part)
    random.Random(42).shuffle(chosen)
    with open(args.out, 'x', encoding='utf-8') as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote {len(chosen)} provisional debug examples; discard their trained adapter.')

def batches_for_epoch(n, epoch, seed, accumulation):
    order = list(range(n)); random.Random(seed + epoch).shuffle(order)
    return [order[i:i + accumulation] for i in range(0, n, accumulation)]

def provenance_guard(args, rows, current):
    verified = read_json(args.verification)
    if verified.get('passed') is not True:
        raise ValueError('Token verification did not pass')
    for k, v in current.items():
        if verified.get(k) != v:
            raise ValueError(f'Verification stale or for other input: {k}')
    if args.mode == 'debug':
        if len(rows) != 32:
            raise ValueError('Debug mode requires exactly 32 examples')
        return None
    if args.mode == 'preliminary':
        if args.epochs != 1 or args.parent:
            raise ValueError('Preliminary runs require one epoch independently from base')
        check_unique(rows)
        counts = collections.Counter(source(r) for r in rows)
        if counts['orca'] != counts['numina']:
            raise ValueError('Preliminary manifest must be 50/50 by examples')
        return None  # No claim of evaluation/protection clearance.
    if not args.clearance:
        raise ValueError('Real runs require --clearance: documented manual review and DEV+TEST overlap clearance')
    clearance = read_json(args.clearance)
    for k in ('manual_reviews_confirmed', 'dev_and_test_decontamination_confirmed'):
        if clearance.get(k) is not True:
            raise ValueError(f'Clearance incomplete: {k}')
    if not clearance.get('evidence'):
        raise ValueError('Record actual review/overlap evidence paths in clearance; a boolean alone is insufficient')
    if current['manifest_sha256'] not in clearance.get('approved_manifest_sha256', []):
        raise ValueError('This manifest is not approved in the clearance file')
    if args.mode == 'curve':
        if args.epochs != 1 or args.parent:
            raise ValueError('Curve runs must start from base and run exactly one epoch')
        check_unique(rows)
        c = collections.Counter(source(r) for r in rows)
        if c['orca'] != c['numina']:
            raise ValueError('Curve manifest must be 50/50 by examples')
    if args.mode == 'branch' and (not args.parent or args.epochs != 1):
        raise ValueError('Branches need a parent and exactly one pass through the exposure manifest')
    return clearance

def environment(torch):
    names = ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'safetensors', 'numpy')
    return dict(python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in names},
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_count_used=1,
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)

def build_model(dtype, adapter=None, trainable=True):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False,
        quantization_config=quant, torch_dtype=dtype, device_map={'': 0}, attn_implementation='eager')
    base.config.use_cache = False
    # Apply the same preparation/casts for reload checks and training.
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=trainable,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    else:
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32,
                                              lora_dropout=0.05, target_modules='all-linear', bias='none'))
    return model

def probe_logits(model, encoded, dtype):
    import torch
    model.eval()
    # Probe logits at the end of a short prefix; no gold scoring or benchmark use here.
    ids = torch.tensor([encoded['input_ids'][:min(32, len(encoded['input_ids']))]], device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        return model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float().cpu()

def save_checkpoint(path, model, optimizer, scaler, tok, state, config, encoded, dtype):
    import torch
    import numpy as np
    path = Path(path)
    temp = path.with_name(path.name + '.incomplete')
    temp.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temp / 'adapter', safe_serialization=True)
    tok.save_pretrained(temp / 'tokenizer')
    probe = probe_logits(model, encoded[0], dtype)
    torch.save(probe, temp / 'probe.pt')
    payload = dict(state=state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(payload, temp / 'training_state.pt')
    dump(temp / 'run_config.json', config)
    temp.rename(path)
    model.train()

def load_training_state(path):
    import torch
    # Only load checkpoints created by this script that YOU trust: pickle can execute code.
    return torch.load(Path(path) / 'training_state.pt', map_location='cpu', weights_only=False)

def parent_budget_check(args, rows, encoded, parent_config, parent_state):
    if parent_config['mode'] != 'curve' or parent_config['rows'] != 1000 or not parent_state['state']['complete']:
        raise ValueError('Pilot branch parent must be a completed 1,000-example curve run')
    seen = set(parent_state['state']['seen_groups'])
    if args.arm != 'REPEAT' and any(r['problem_group_id'] in seen for r in rows):
        raise ValueError('Fresh branch contains a parent-seen problem group')
    if args.arm == 'REPEAT' and any(r['problem_group_id'] not in seen for r in rows):
        raise ValueError('REPEAT contains an unseen problem group')
    target = parent_state['state']['source_tokens']
    actual = collections.Counter()
    for r, e in zip(rows, encoded):
        actual[source(r)] += e['counts']['supervised_tokens']
    for k in ('orca', 'numina'):
        if not target.get(k) or abs(actual[k] - target[k]) / target[k] > 0.01:
            raise ValueError(f'{k} branch budget differs by more than 1%; ask Person 4 to reconcile whole-example selection.')
    return dict(parent_processed_tokens=parent_state['state']['processed_tokens'],
                target_source_tokens=target, actual_source_tokens=dict(actual))

def train(args):
    import torch
    import numpy as np
    from transformers import set_seed
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This training implementation requires one CUDA/NVIDIA GPU; no CPU/MPS or distributed fallback.')
    if args.epochs < 1 or args.accumulation < 1 or args.save_every < 1 or args.lr <= 0:
        raise ValueError('epochs, accumulation, save-every and learning rate must be positive')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Use a NEW output folder for every run/resume segment; existing output is never overwritten.')
    rows = load_rows(args.data)
    check_unique(rows, allow_repeats=args.mode == 'branch' and args.arm == 'REPEAT')
    template, policy = load_contract(args.template, args.policy)
    current = fingerprint(args)
    clearance = provenance_guard(args, rows, current)
    tok = get_tokenizer()
    records, encoded = reconcile(rows, tok, template, policy)
    if not all(r['match'] for r in records):
        raise ValueError('Actual tokenizer reconciliation failed immediately before training')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    env = environment(torch)
    config = dict(**current, experiment_status='preliminary' if args.mode == 'preliminary' else args.mode,
                  evaluation_readiness='pending' if args.mode == 'preliminary' else 'see_mode_and_clearance',
                  mode=args.mode, arm=args.arm if args.mode == 'branch' else None,
                  rows=len(rows), epochs=args.epochs, seed=args.seed, accumulation=args.accumulation,
                  learning_rate=args.lr, warmup_updates=10, microbatch_size=1, packing=False,
                  dtype=str(dtype), environment=env, parent=str(Path(args.parent).resolve()) if args.parent else None,
                  branch_optimizer_policy='restore_parent_no_new_warmup' if args.mode == 'branch' else None,
                  clearance_sha256=sha(args.clearance) if clearance else None)
    restore = None
    parent_state = None
    adapter = None
    if args.parent:
        if args.mode != 'branch':
            raise ValueError('--parent is only for branch mode')
        pcfg = read_json(Path(args.parent) / 'run_config.json')
        for k in ('model', 'revision', 'template_sha256', 'policy_sha256', 'code_sha256', 'dtype', 'learning_rate', 'accumulation'):
            if pcfg.get(k) != config[k]:
                raise ValueError(f'Parent configuration differs: {k}')
        if pcfg['environment']['packages'] != env['packages']:
            raise ValueError('Parent package versions differ; reproduce the parent environment')
        parent_state = load_training_state(args.parent)
        config['branch_budget'] = parent_budget_check(args, rows, encoded, pcfg, parent_state)
        config['parent_state_sha256'] = sha(Path(args.parent) / 'training_state.pt')
        config['parent_adapter_sha256'] = sha(Path(args.parent) / 'adapter' / 'adapter_model.safetensors')
        adapter = Path(args.parent) / 'adapter'
    if args.resume:
        previous = read_json(Path(args.resume) / 'run_config.json')
        if previous != config:
            raise ValueError('Resume config differs. Keep all inputs/seed/epochs/parent/environment identical.')
        restore = load_training_state(args.resume)
        adapter = Path(args.resume) / 'adapter'
    set_seed(args.seed)
    model = build_model(dtype, adapter=adapter)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, p in trainable):
        raise ValueError('Expected only LoRA parameters to be trainable')
    params = [p for n, p in trainable]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    state = dict(next_epoch=0, next_group=0, optimizer_updates=0, processed_tokens=0, input_tokens=0,
                 source_tokens={}, seen_groups=[], seen_ids=[], training_seconds=0.0, complete=False,
                 parent_optimizer_updates=0)
    if parent_state:
        optimizer.load_state_dict(parent_state['optimizer']); scaler.load_state_dict(parent_state['scaler'])
        state['parent_optimizer_updates'] = parent_state['state']['optimizer_updates']
        # Same branch seed + fresh deterministic data order; parent optimizer restored for every arm.
    if restore:
        state = restore['state']
        if state['complete']:
            raise ValueError('Checkpoint already completed; use branch mode for new data')
        optimizer.load_state_dict(restore['optimizer']); scaler.load_state_dict(restore['scaler'])
    # Prevent accidental same-object/no-update success in the debug validation.
    initial = {n: p.detach().float().cpu().clone() for n, p in trainable} if args.mode == 'debug' else None
    before = probe_logits(model, encoded[0], dtype)
    if args.resume:
        saved_probe = torch.load(Path(args.resume) / 'probe.pt', weights_only=True)
        if not torch.allclose(saved_probe, before, rtol=1e-3, atol=1e-3):
            raise ValueError('Reload logit probe differs; inspect loading/precision before resuming')
    # Set RNG after model construction/probe. On resume restore exact checkpoint RNG.
    if restore:
        random.setstate(restore['python_rng']); np.random.set_state(restore['numpy_rng'])
        torch.set_rng_state(restore['torch_rng']); torch.cuda.set_rng_state_all(restore['cuda_rng'])
    else:
        set_seed(args.seed)
    out.mkdir(parents=True)
    dump(out / 'run_config.json', config)
    dump(out / 'trainable_parameters.json', {'names': [n for n, p in trainable], 'count': sum(p.numel() for p in params)})
    (out / 'requirements.lock.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    model.train(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter(); old_seconds = state['training_seconds']
    seen_ids, seen_groups = set(state['seen_ids']), set(state['seen_groups'])
    segment_updates = 0
    stop = False
    with open(out / 'training_log.jsonl', 'w', encoding='utf-8') as log:
        for epoch in range(state['next_epoch'], args.epochs):
            groups = batches_for_epoch(len(rows), epoch, args.seed, args.accumulation)
            first = state['next_group'] if epoch == state['next_epoch'] else 0
            for g in range(first, len(groups)):
                indices = groups[g]
                target_total = sum(encoded[i]['counts']['supervised_tokens'] for i in indices)
                update = state['optimizer_updates'] + 1
                lr = args.lr if args.mode == 'branch' else args.lr * min(update / 10.0, 1.0)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                for i in indices:
                    e = encoded[i]
                    ids = torch.tensor([e['input_ids']], dtype=torch.long, device='cuda')
                    labels = torch.tensor([e['labels']], dtype=torch.long, device='cuda')
                    with torch.autocast('cuda', dtype=dtype):
                        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                        # Explicit next-token objective. Chunk float32 cross entropy to limit
                        # temporary memory for Qwen's large vocabulary. No label shift twice.
                        loss_sum = torch.zeros((), device='cuda')
                        for pos in range(0, ids.shape[1] - 1, 128):
                            end = min(pos + 128, ids.shape[1] - 1)
                            loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                                logits[0, pos:end].float(), labels[0, pos + 1:end + 1],
                                ignore_index=-100, reduction='sum')
                        loss = loss_sum / target_total
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; stop and diagnose data/precision before comparing runs')
                    weighted_loss += loss.detach().item()
                    scaler.scale(loss).backward()
                    del logits, loss_sum, loss, ids, labels
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError('Nonfinite gradients: no skipped updates are silently counted; diagnose precision and restart from checkpoint.')
                scaler.step(optimizer); scaler.update()
                state['optimizer_updates'] += 1; segment_updates += 1
                state['processed_tokens'] += target_total
                state['input_tokens'] += sum(encoded[i]['counts']['total_tokens'] for i in indices)
                for i in indices:
                    r, e = rows[i], encoded[i]
                    seen_ids.add(r['row_id']); seen_groups.add(r['problem_group_id'])
                    k = source(r)
                    state['source_tokens'][k] = state['source_tokens'].get(k, 0) + e['counts']['supervised_tokens']
                state['seen_ids'], state['seen_groups'] = sorted(seen_ids), sorted(seen_groups)
                state['next_epoch'], state['next_group'] = (epoch + 1, 0) if g + 1 == len(groups) else (epoch, g + 1)
                torch.cuda.synchronize()
                state['training_seconds'] = old_seconds + time.perf_counter() - start
                record = dict(epoch=epoch, optimizer_update=state['optimizer_updates'], target_tokens=target_total,
                              cumulative_processed_tokens=state['processed_tokens'], loss=weighted_loss,
                              lr=lr, grad_norm=float(grad_norm), elapsed_seconds=state['training_seconds'],
                              peak_allocated_GB=torch.cuda.max_memory_allocated() / 1e9)
                log.write(json.dumps(record) + '\n'); log.flush()
                print(json.dumps(record), flush=True)
                if args.stop_after and segment_updates >= args.stop_after:
                    stop = True; break
                if state['optimizer_updates'] % args.save_every == 0:
                    save_checkpoint(out / f'checkpoint-{state["optimizer_updates"]:06d}', model, optimizer, scaler,
                                    tok, state, config, encoded, dtype)
            if stop:
                break
    state['complete'] = state['next_epoch'] >= args.epochs
    final_path = out / ('final' if state['complete'] else 'paused')
    save_checkpoint(final_path, model, optimizer, scaler, tok, state, config, encoded, dtype)
    csv_write(out / 'seen_groups.csv', [dict(problem_group_id=g) for g in sorted(seen_groups)])
    if args.mode == 'curve' and len(rows) == 1000 and state['complete']:
        csv_write(out / 'parent_seen_groups_R1000.csv', [dict(problem_group_id=g) for g in sorted(seen_groups)])
    actual_unique = {r['row_id']: e['counts']['supervised_tokens'] for r, e in zip(rows, encoded) if r['row_id'] in seen_ids}
    summary = dict(**state, experiment_status=config['experiment_status'],
                   evaluation_readiness=config['evaluation_readiness'], unique_examples=len(seen_ids), unique_supervised_tokens=sum(actual_unique.values()),
                   peak_allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
                   seconds_include_checkpoint_overhead=True, checkpoint=str(final_path.resolve()))
    if parent_state:
        summary['total_parent_plus_branch_tokens'] = parent_state['state']['processed_tokens'] + state['processed_tokens']
    if initial is not None:
        delta = max((p.detach().float().cpu() - initial[n]).abs().max().item() for n, p in trainable)
        summary['max_adapter_delta_this_segment'] = delta
        if not delta > 0:
            raise RuntimeError('Debug adapters did not change')
    dump(out / 'training_summary.json', summary)
    print(f'Saved {final_path}. Evaluate with your separately validated Person 3 evaluator.')

def reload_check(args):
    import torch
    cfg = read_json(Path(args.checkpoint) / 'run_config.json')
    if cfg['model'] != MODEL or cfg['revision'] != REVISION:
        raise ValueError('Checkpoint model identity differs')
    for k in ('manifest_sha256', 'template_sha256', 'policy_sha256', 'code_sha256', 'wrapper_sha256', 'reference_code_sha256'):
        if fingerprint(args)[k] != cfg[k]:
            raise ValueError('Reload check must use original data/template/policy/code')
    dtype = torch.bfloat16 if cfg['dtype'] == 'torch.bfloat16' else torch.float16
    rows = load_rows(args.data); template, policy = load_contract(args.template, args.policy)
    enc = encode_row(rows[0], get_tokenizer(), template, policy['max_total_tokens'])
    model = build_model(dtype, Path(args.checkpoint) / 'adapter', trainable=False)
    actual = probe_logits(model, enc, dtype)
    expected = torch.load(Path(args.checkpoint) / 'probe.pt', weights_only=True)
    maximum = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
    print(json.dumps(dict(passed=passed, max_absolute_logit_difference=maximum)))
    if not passed:
        raise SystemExit('Reload mismatch: inspect before real runs')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    def common(q):
        q.add_argument('--data', required=True); q.add_argument('--template', required=True); q.add_argument('--policy', required=True)
    q = sub.add_parser('check-nesting'); q.add_argument('--data', nargs='+', required=True); q.set_defaults(func=check_nesting)
    q = sub.add_parser('subset'); q.add_argument('--data', required=True); q.add_argument('--out', required=True); q.set_defaults(func=subset)
    q = sub.add_parser('verify'); common(q); q.add_argument('--out', required=True); q.add_argument('--allow-repeats', action='store_true'); q.set_defaults(func=verify)
    q = sub.add_parser('train'); common(q)
    q.add_argument('--verification', required=True); q.add_argument('--out', required=True)
    q.add_argument('--mode', choices=['debug', 'preliminary', 'curve', 'branch'], required=True)
    q.add_argument('--clearance'); q.add_argument('--epochs', type=int, default=1)
    q.add_argument('--seed', type=int, default=42); q.add_argument('--accumulation', type=int, default=8)
    q.add_argument('--lr', type=float, default=1e-4); q.add_argument('--save-every', type=int, default=100)
    q.add_argument('--parent'); q.add_argument('--resume'); q.add_argument('--stop-after', type=int)
    q.add_argument('--arm', choices=['REPEAT', 'RANDOM', 'FIT', 'MATCHED_RANDOM'])
    q.set_defaults(func=train)
    q = sub.add_parser('reload-check'); common(q); q.add_argument('--checkpoint', required=True); q.set_defaults(func=reload_check)
    args = p.parse_args()
    if args.command == 'train' and args.mode == 'branch' and not args.arm:
        p.error('Branch mode requires --arm')
    if args.command == 'train' and args.mode != 'branch' and args.arm:
        p.error('--arm only belongs to branch mode')
    args.func(args)

if __name__ == '__main__':
    main()

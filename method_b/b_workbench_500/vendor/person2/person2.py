"""Person 2 v3 integration; run --help. NVIDIA GPU needed only for training/reload."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import trainer_core as core
from capstone_person1 import person1_data_contract as contract
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
VERSION = 'person2-dual-v3-20261015'
REFERENCE_ROOT = Path(contract.__file__).resolve().parent
REFERENCE_FILES = ['__init__.py','person1_data_contract.py','pipeline_controls.py','instance_matching.py']
MODELS = {
    '0.5B': ('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987'),
    '1.5B': ('Qwen/Qwen2.5-1.5B', '8faed761d45a263340a0528343f099c05c9a4323'),
}
SELECTED_MODEL = '0.5B'

@lru_cache(maxsize=2)
def model_tokenizer(model_id, revision):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
    contract.verify_tokenizer(tok)
    return tok

def get_tokenizer():
    return model_tokenizer(core.MODEL,core.REVISION)

def select_model(name):
    global SELECTED_MODEL
    SELECTED_MODEL = name
    core.MODEL, core.REVISION = MODELS[name]

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p, obj): core.dump(p, obj)
def digest(p): return core.sha(p)

def collate(encoded):
    # Used both in verification and the actual training encoder.
    row = contract.pad_encoded(encoded, len(encoded['input_ids']))
    n, p = len(row['input_ids']), encoded['prompt_tokens']
    assert row['labels'][:p] == [-100]*p
    assert row['attention_mask'] == [1]*n
    assert row['labels'][-1] == contract.EXPECTED_END_ID
    assert sum(v != -100 for v in row['labels'][1:]) == encoded['supervised_tokens']
    return row

def encode(r, tokenizer, template, cap):
    if template != contract.PROMPT_TEMPLATE: raise ValueError('Prompt contract differs')
    e = contract.validate_manifest_record(r, tokenizer, max_total_tokens=cap)
    row = collate(e)
    if SELECTED_MODEL == '1.5B':
        reference_tok = model_tokenizer(*MODELS['0.5B'])
        reference = contract.validate_manifest_record(r, reference_tok, max_total_tokens=cap)
        # Equal counts alone do not prove equal tokenization. Compare actual IDs and masks too.
        for key in ('input_ids','labels','attention_mask'):
            if row[key] != reference[key]:
                raise ValueError(f'1.5B tokenizer differs on {r["row_id"]}: {key}. Reconcile budgets with Person 1; no silent reuse.')
    # Actual microbatch size is 1; no padding is needed in the training path.
    # Exercise the same positional-padding helper with extra padding in verification.
    padded = contract.pad_encoded(e, len(e['input_ids'])+3)
    assert padded['labels'][-3:] == [-100]*3
    assert padded['attention_mask'][-3:] == [0]*3
    assert padded['labels'][len(e['input_ids'])-1] == contract.EXPECTED_END_ID
    assert sum(x != -100 for x in padded['labels']) == e['supervised_tokens']
    return dict(input_ids=row['input_ids'], labels=row['labels'],
                counts={k:e[k] for k in core.COUNT_FIELDS}, boundary_method='v3_combined_offsets')

_original_fingerprint = core.fingerprint

def fingerprint(args):
    f = _original_fingerprint(args)
    f['wrapper_sha256'] = digest(__file__)
    f['reference_code_sha256'] = {p.name:digest(p) for p in [REFERENCE_ROOT/name for name in REFERENCE_FILES]}
    return f

core.encode_row = encode
core.fingerprint = fingerprint
core.get_tokenizer = get_tokenizer


def setup(release, scope, work):
    release, work = Path(release).resolve(), Path(work).resolve()
    if work == release or release in work.parents: raise ValueError('WORK must be outside frozen release')
    meta = read(release/'metadata/release_manifest.json')
    if not meta['readiness']['data_ready_for_trainer']: raise ValueError('Release is not data-ready')
    for relative in [f'manifests/{scope}.jsonl', f'manifests/{scope}_summary.json', 'configs/formatting_contract.json']:
        path = release/relative
        if not path.exists(): raise FileNotFoundError(f'Missing {path}. Download archive 01; release_records is metadata only.')
        if digest(path) != meta['files'][relative]['sha256']: raise ValueError(f'Release hash mismatch: {relative}')
    origin_path = release/'metadata/encoder_origin_manifest.json'
    if digest(origin_path) != meta['original_manifest_sha256']:
        raise ValueError('Encoder origin manifest hash mismatch')
    origin = read(origin_path)
    for path in [REFERENCE_ROOT/name for name in REFERENCE_FILES]:
        key = 'code/capstone_person1/'+path.name
        expected = meta['files'].get(key, origin['files'].get(key, {})).get('sha256')
        if digest(path) != expected: raise ValueError(f'Encoder differs from verified provenance: {key}')
    cfg = read(release/'configs/formatting_contract.json')
    if cfg['contract_version'] != contract.CONTRACT_VERSION: raise ValueError('Unsupported contract')
    work.mkdir(parents=True, exist_ok=True)
    template, policy = work/'prompt_template.json', work/'tokenization_policy.json'
    write(template, {'template': cfg['prompt_template']})
    write(policy, dict(confirmed_with_person1=True, add_special_tokens=False, append_eos=True,
        max_total_tokens=cfg['max_total_tokens'], provenance=f"{release}; {contract.CONTRACT_VERSION}"))
    return SimpleNamespace(data=str(release/f'manifests/{scope}.jsonl'), template=str(template), policy=str(policy),
        release=release, scope=scope, work=work, allow_repeats=False)


def verify(a, who):
    a.out = a.work/'checks'/a.scope
    # Re-encode and validate aggregate counts against the actual release summary.
    summary = read(a.release/f'manifests/{a.scope}_summary.json')
    contract.validate_manifest(a.data, summary, tokenizer=get_tokenizer())
    core.verify(a)
    rows = list(csv.DictReader((a.out/'person2_token_verification.csv').open()))
    with (a.out/'actual_labels.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=['row_id','actual_supervised_tokens']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    write(a.out/'actual_labels_metadata.json', dict(trainer_version=VERSION,
        contract_version=contract.CONTRACT_VERSION, model=core.MODEL, model_revision=core.REVISION,
        tokenizer_compatibility='every row IDs/labels/attention matched 0.5B reference' if SELECTED_MODEL=='1.5B' else 'reference model', manifest_sha256=digest(a.data),
        prompt_mask_verified=True, eos_label_verified=True, padding_by_position_verified=True,
        confirmed_by=who, confirmed_utc=now(), check_method='programmatic assertions on actual collator output; inspect mask_preview.csv',
        padding_note='training microbatch=1, no padding; extra-padding regression executed for every row'))
    print('Send actual_labels.csv and actual_labels_metadata.json to Person 1; inspect mask_preview.csv.')


def clearance(a, path):
    c = read(path)
    # Never turn a DATA_READY status into evaluation clearance automatically.
    for key in ('person3_evaluator_frozen','development_and_test_protection_verified'):
        if c.get(key) is not True: raise ValueError('Team readiness incomplete: '+key)
    if c.get('release_manifest_sha256') != digest(a.release/'metadata/release_manifest.json'):
        raise ValueError('Readiness is for a different data release')
    if not c.get('confirmed_by') or not c.get('confirmed_utc'): raise ValueError('Actual identity/time required')
    for key in ('evaluation_contract','overlap_report'):
        evidence=c.get(key,{})
        p=Path(evidence.get('path',''))
        if not p.is_file() or digest(p)!=evidence.get('sha256'): raise ValueError('Missing/stale evidence: '+key)
    return dict(manual_reviews_confirmed=True,dev_and_test_decontamination_confirmed=True,
        evidence={'team_readiness_path':str(Path(path).resolve()),'team_readiness_sha256':digest(path)},
        approved_manifest_sha256=[digest(a.data)])


def train(a, ns):
    debug=a.scope=='debug_32'
    a.mode='debug' if debug else ('preliminary' if ns.preliminary else 'curve')
    a.epochs=ns.debug_epochs if debug else 1
    a.seed=ns.seed; a.accumulation=8;a.lr=1e-4;a.save_every=100
    a.parent=None;a.resume=ns.resume;a.stop_after=ns.stop_after;a.arm=None
    a.verification=a.work/'checks'/a.scope/'verification.json'
    a.out=a.work/'runs'/ns.run_name;a.clearance=None
    if not debug:
        if ns.preliminary and ns.readiness:
            raise ValueError('Choose preliminary or readiness-backed training, not both')
        if not ns.preliminary and not ns.readiness:
            raise ValueError('Use --preliminary for a pending-evaluation experiment, or supply --readiness')
        gate=None if ns.preliminary else clearance(a,ns.readiness)
        # Require actual debug training AND reload evidence before main runs.
        report=read(a.work/'debug_reload_passed.json')
        if report['release_manifest_sha256']!=digest(a.release/'metadata/release_manifest.json'):
            raise ValueError('Debug was for a different release')
        if report['fingerprint'] != fingerprint(setup(a.release,'debug_32',a.work)):
            raise ValueError('Debug code/config has changed')
        if digest(Path(report['checkpoint'])/'adapter/adapter_model.safetensors')!=report['adapter_sha256']:
            raise ValueError('Debug adapter changed')
        if gate is not None:
            a.clearance=a.work/(ns.run_name+'_clearance.json');write(a.clearance,gate)
        # Selected manifest is verified by core.train. No R1000 dependency for preliminary runs.
        if not ns.preliminary:
            v=read(a.work/'checks/train_n1000/verification.json')
            ref=fingerprint(setup(a.release,'train_n1000',a.work))
            if not v.get('passed') or any(v.get(k)!=value for k,value in ref.items()):
                raise ValueError('Verify all R1000 labels first for a formal curve run')
    core.train(a)
    if a.mode == 'preliminary':
        write(a.out/'PRELIMINARY_STATUS.json', dict(
            experiment_status='preliminary', person3_evaluator_frozen=False,
            development_and_test_protection_verified=False,
            note='These checks have not been established by this run; false does not mean overlap was found.',
            release_manifest_sha256=digest(a.release/'metadata/release_manifest.json'),
            created_utc=now(), intended_use='Checkpoint loading and preliminary evaluation; final study eligibility pending'))


def reload(a, ns):
    a.checkpoint=Path(ns.checkpoint).resolve()
    core.reload_check(a)
    summary = read(a.checkpoint.parent/'training_summary.json')
    if not summary.get('complete'):
        raise ValueError('A paused checkpoint reload does not establish a completed smoke/main run')
    report=dict(status='passed',checkpoint=str(a.checkpoint),adapter_sha256=digest(a.checkpoint/'adapter/adapter_model.safetensors'),
        release_manifest_sha256=digest(a.release/'metadata/release_manifest.json'),fingerprint=fingerprint(a),verified_utc=now())
    write(a.work/('debug_reload_passed.json' if a.scope=='debug_32' else a.scope+'_reload_passed.json'),report)


def parent(a, ns):
    if a.scope!='train_n1000': raise ValueError('Parent binding requires train_n1000')
    run=Path(ns.run).resolve();s=read(run/'training_summary.json');cfg=read(run/'run_config.json')
    if cfg.get('mode') != 'curve': raise ValueError('Only readiness-backed curve runs can become formal parents')
    if not s['complete'] or s['unique_examples']!=1000 or cfg['manifest_sha256']!=digest(a.data):
        raise ValueError('Not a completed matching R1000')
    check=read(a.work/'train_n1000_reload_passed.json')
    adapter=run/'final/adapter/adapter_model.safetensors'
    if check['adapter_sha256']!=digest(adapter) or check['checkpoint']!=str(run/'final'): raise ValueError('Reload this checkpoint first')
    # Match observed seen groups exactly and check selection freshness.
    groups={r['problem_group_id'] for r in core.load_rows(a.data)}
    actual={r['problem_group_id'] for r in csv.DictReader((run/'seen_groups.csv').open())}
    if actual!=groups: raise ValueError('Actual seen groups differ')
    selection=a.release/'manifests/selection_bank.jsonl'
    meta=read(a.release/'metadata/release_manifest.json')
    if digest(selection)!=meta['files']['manifests/selection_bank.jsonl']['sha256']: raise ValueError('Selection hash differs')
    if groups & {r['problem_group_id'] for r in core.load_rows(selection)}: raise ValueError('Selection overlap')
    write(run/'actual_parent.json',dict(actual_run_id=run.name,checkpoint_id=str(run/'final'),
        checkpoint_sha256=digest(adapter),checkpoint_digest_scope='final/adapter/adapter_model.safetensors',
        actual_manifest_sha256=digest(a.data),confirmed_by=ns.confirmed_by,confirmed_utc=now()))
    print('Parent evidence ready:',run/'actual_parent.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','train','reload','parent'])
    p.add_argument('--model',choices=list(MODELS),required=True)
    p.add_argument('--release',required=True);p.add_argument('--work',required=True)
    p.add_argument('--scope',choices=['debug_32']+[f'train_n{n}' for n in [500,1000,2000,4000,8000,16000,40000]],required=True)
    p.add_argument('--confirmed-by');p.add_argument('--run-name');p.add_argument('--readiness')
    p.add_argument('--preliminary',action='store_true',help='Train with evaluation/protection readiness explicitly pending')
    p.add_argument('--debug-epochs',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint');p.add_argument('--run');p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    n=p.parse_args()
    if n.command in ['verify','parent'] and not n.confirmed_by:p.error('--confirmed-by required')
    if n.command=='train' and not n.run_name:p.error('--run-name required')
    if n.command=='reload' and not n.checkpoint:p.error('--checkpoint required')
    if n.command=='parent' and not n.run:p.error('--run required')
    select_model(n.model)
    a=setup(n.release,n.scope,n.work)
    {'verify':lambda:verify(a,n.confirmed_by),'train':lambda:train(a,n),
     'reload':lambda:reload(a,n),'parent':lambda:parent(a,n)}[n.command]()

if __name__=='__main__':main()

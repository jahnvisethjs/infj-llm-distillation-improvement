#!/usr/bin/env python3
"""Eleven screened B controls, durable Slurm jobs, no benchmark evaluation."""
import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import gc
import getpass
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
VERSION = 'screened-b-4955-controls-training-v1'
BANK_SHA = 'ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f'
B_MANIFEST_SHA = '04112f96b282ba8a58fcd8900f6ca91b656dbac9abbd17a4cdef2b02a9bb00e0'
B_SELECTION_RECEIPT_SHA = '8db876bef7e086b9b44210680b72eae19910fee11dda5606a2915d318b3dd866'
ENGINE_SHA = '50809b9e374fcc05a9b637c381396b46223ab85abcda3ca07cd4b6a5a4af882f'
ENGINE_PINS_SHA = '9692100ef849dbceba97291802987076eb867f34f7d52471225f9182f33e70d8'
PILOT_RUNNER_SHA = 'fc83b1b2395c1d6292751c41544f5736534cc5ec7143d8d32cc7b8e0acebb0e6'
PILOT_ID = 'be46c67690fb33ba680a5cd578bd418c6c06f4f2bcfadb39b329a10cb9fd5759'
PILOT_ADAPTER_SHA = '54ca4286a02a9d94686190c599405edb1c979f724fbb261ef0617385717bbe1b'
TARGET = 318845
ACTUAL_B_TOKENS = 317486
PROFILES = (('0.5B',17),('0.5B',123),('1.5B',17),('1.5B',123))
SEEDS = (42,137,2026)
MANIFESTS = {'B'}
MAX_ATTEMPTS = 3

class IntegrityError(ValueError):
    """Stop on scientific or provenance drift; never relax constraints."""

class SelectionBlocked(IntegrityError):
    pass

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def check_hash(path,wanted):
    if sha(path)!=wanted: raise IntegrityError('Changed or incorrect file, preserved: '+str(path))
    return wanted

def save_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()==data:return
        raise IntegrityError('Different existing evidence preserved: '+str(path))
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(tmp,path)

def save(path,value):
    save_bytes(path,(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())

def save_rows(path,records):
    save_bytes(path,''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records).encode())

def current_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

@contextlib.contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise IntegrityError('Another submission/worker is active for this output directory')
        yield

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def package_check():
    pins=read(HERE/'PACKAGE_PINS.json')
    for name,value in pins.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise IntegrityError('Invalid package member')
        check_hash(HERE/rel,value)
    return sha(HERE/'PACKAGE_PINS.json')

def control_root(a):
    return Path(a.work).resolve()/'outputs/B_screened_4955_controls_v1'/digest({
        'version':VERSION,'package':package_check(),'work':str(Path(a.work).resolve()),
        'b_root':str(Path(a.b_root).resolve())})[:16]

def artifact(c,*parts):
    return c.out/c.profile/c.identity[:16]/Path(*parts)

def pilot_root(work):
    return Path(work)/'outputs/C0_screened_pilot_v1/0.5B_seed17/be46c67690fb33ba'

def engine_module(work):
    directory=Path(work)/'C0_C1_Overnight_Training_20261110'
    check_hash(directory/'nightly.py',ENGINE_SHA)
    check_hash(directory/'PACKAGE_PINS.json',ENGINE_PINS_SHA)
    obj=load_module('_verified_existing_c0_c1_runtime',directory/'nightly.py')
    obj.package_check()
    return obj

def load_context(a):
    package_check()
    engine=engine_module(a.work)
    source=engine.load_context(SimpleNamespace(**vars(a)))
    # Existing screened tokenizer proofs and parents are validated read-only.
    engine.verify_proof(source)
    path=pilot_root(a.work)/'selection/B/train.jsonl'
    check_hash(path,B_MANIFEST_SHA)
    selection_path=path.parent/'selection_preview.json'
    check_hash(selection_path,B_SELECTION_RECEIPT_SHA)
    report=read(selection_path)
    if (report.get('method')!='B' or report.get('rows')!=1000
        or report.get('actual_tokens')!=ACTUAL_B_TOKENS
        or report.get('manifest_sha256')!=B_MANIFEST_SHA
        or report.get('solver',{}).get('passed') is not True
        or report.get('solver',{}).get('tolerance_relaxed') is not False):
        raise IntegrityError('Screened B pilot selection proof differs')
    selected=rows(path)
    if source.checks.validate_selection(selected,source.bank,source.rebinned)!=ACTUAL_B_TOKENS:
        raise IntegrityError('Screened B selection violates membership, rows, strata, order or token bounds')
    deps={'version':VERSION,'package_sha256':package_check(),'source_screened_identity':source.identity,
          'bank_sha256':BANK_SHA,'B_manifest_sha256':B_MANIFEST_SHA,
          'B_selection_source_sha256':B_SELECTION_RECEIPT_SHA,'profile':source.profile,
          'parent_sha256':source.entry['expected_adapter_sha256']}
    return SimpleNamespace(**{**vars(source),'a':a,'work':Path(a.work).resolve(),
        'out':control_root(a)/'profiles','identity':digest(deps),'dependencies':deps,
        'source_context':source,'engine':engine,'selected':selected,'source_B_path':path,
        'labels':source.labels,'control':control_root(a)})

def freeze(c):
    return {'version':VERSION,'identity':c.identity,'dependencies':c.dependencies,
            'bank_rows':4955,'bank_sha256':BANK_SHA,'method':'B','profile':c.profile,
            'parent_sha256':c.entry['expected_adapter_sha256'],'parent_seed':c.a.parent_seed,
            'target_tokens':TARGET,'actual_selected_tokens':ACTUAL_B_TOKENS,'rows':1000,
            'selection_seed':42,'strata':20,'rows_per_stratum':50,'global_tolerance':0.005,
            'stratum_tolerance':0.05,'continuation_seeds':list(SEEDS),
            'benchmark_evaluation_performed':False,'reserved_final_tests_performed':False,
            'original_matcher_flags_resolved':False,
            'source_B_manifest':str(c.source_B_path),
            'limitations':read(HERE/'protocol.json')['limitations']}

def require_prepared(c):
    path=artifact(c,'freeze.json')
    if read(path)!=freeze(c):raise IntegrityError('Frozen B context changed')
    check_hash(artifact(c,'candidate_bank.jsonl'),BANK_SHA)
    return sha(path)

def verify_proof(c):
    require_prepared(c)
    proof=c.engine.verify_proof(c.source_context)
    c.labels=c.source_context.labels
    if len(c.labels)!=5955:raise IntegrityError('Incomplete parent plus candidate tokenizer proof')
    return proof

def selection_record(c):
    return {'identity':c.identity,'method':'B','rows':1000,'actual_tokens':ACTUAL_B_TOKENS,
            'target_tokens':TARGET,'manifest_sha256':B_MANIFEST_SHA,'bank_sha256':BANK_SHA,
            'parent_sha256':c.entry['expected_adapter_sha256'],'selection_seed':42,
            'source_manifest':str(c.source_B_path),'source_selection_sha256':B_SELECTION_RECEIPT_SHA,
            'reference_scoring_used':False,'candidate_generations_used':False,
            'selection_reused_across_training_seeds':True}

def prepare(c):
    save_bytes(artifact(c,'candidate_bank.jsonl'),c.bank_path.read_bytes())
    save(artifact(c,'freeze.json'),freeze(c))
    save_bytes(artifact(c,'selection/B/train.jsonl'),c.source_B_path.read_bytes())
    save(artifact(c,'selection/B/selection.json'),selection_record(c))
    load_selection(c,'B');verify_proof(c)

def load_selection(c,method):
    if method!='B':raise IntegrityError('This package authorizes screened B training only')
    path=artifact(c,'selection/B/train.jsonl');check_hash(path,B_MANIFEST_SHA)
    report=read(path.parent/'selection.json')
    if report!=selection_record(c):raise IntegrityError('B selection receipt/context changed')
    selected=rows(path)
    if c.checks.validate_selection(selected,c.bank,c.rebinned)!=ACTUAL_B_TOKENS:
        raise IntegrityError('Selected B examples differ from the frozen screened bank')
    return selected,report

def environment_check(c):
    expected=read(c.runtime/c.entry['records_relative_path']/'run_config.json')['environment']['packages']
    for name,version in expected.items():
        if importlib.metadata.version(name)!=version:
            raise IntegrityError('Parent training package version differs: '+name)

def original_pilot(a):
    pkg=Path(a.work)/'C0_Screened_05B_Seed17_20261110'
    check_hash(pkg/'screened_c0.py',PILOT_RUNNER_SHA)
    pilot=load_module('_screened_B_completed_pilot',pkg/'screened_c0.py')
    c=pilot.load_context(SimpleNamespace(work=str(a.work),b_root=str(a.b_root)))
    if c.identity!=PILOT_ID:raise IntegrityError('Original pilot identity differs')
    directory=pilot.run_directory(c,'B','pilot01')
    co=pilot.validate_completion(c,directory,'B')
    if co['bank_sha256']!=BANK_SHA or co['adapter_sha256']!=PILOT_ADAPTER_SHA:
        raise IntegrityError('Original screened B pilot checkpoint differs')
    return co,str(directory)

def child(a,action,**kwargs):
    cmd=[sys.executable,'-B',str(HERE/'screened_b.py'),action,'--work',str(a.work),
         '--b-root',str(a.b_root),'--model',a.model,'--parent-seed',str(a.parent_seed)]
    for key,value in kwargs.items():
        cmd.append('--'+key.replace('_','-'))
        if value is not True:cmd.append(str(value))
    return cmd

# The following seven functions are copied without changes from the validated
# C0/C1 runner. They reuse B's exact trainer and validate real consumption,
# optimizer isolation, parent integrity, adapter changes and reload probes.

def training_source_for_method(source, method):
    if method not in MANIFESTS or source.count("'preliminary_document_aligned_B'") != 1:
        raise IntegrityError("Unexpected parent trainer label or method")
    return source.replace("'preliminary_document_aligned_B'", repr("screened_training_only_" + method), 1)

def gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise IntegrityError("Exactly one allocated NVIDIA GPU required; use the notebook launch cell")
    return torch

def run_directory(c, method, seed, attempt, smoke=False):
    return artifact(c, "smokes" if smoke else "runs", method, f"seed{seed}_{attempt}")

def verify_adapter_update(parent_path, continued_path):
    from safetensors import safe_open
    import torch
    changed = []
    with safe_open(str(parent_path), framework="pt", device="cpu") as parent, \
            safe_open(str(continued_path), framework="pt", device="cpu") as continued:
        names = set(parent.keys())
        if names != set(continued.keys()) or not names or any("lora_" not in name for name in names):
            raise ValueError("Continued adapter tensor identities differ from original LoRA parent")
        for name in sorted(names):
            before, after = parent.get_tensor(name), continued.get_tensor(name)
            if before.shape != after.shape or not torch.isfinite(after).all():
                raise ValueError("Continued adapter has incompatible or nonfinite tensors")
            if not torch.equal(before, after):
                changed.append(name)
    if not changed:
        raise ValueError("Continuation did not change parent adapter tensor values")
    return changed

def validate_consumption(summary, selected, logs):
    tokens = sum(r["supervised_tokens"] for r in selected)
    updates = (len(selected) + 7) // 8
    if (summary.get("complete") is not True or summary.get("processed_tokens") != tokens
            or summary.get("unique_examples") != len(selected)
            or summary.get("unique_supervised_tokens") != tokens
            or summary.get("optimizer_updates") != updates
            or summary.get("parent_optimizer_updates") != 0
            or set(summary.get("seen_ids", [])) != {r["row_id"] for r in selected}
            or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in selected}):
        raise ValueError("Training consumption, IDs, updates or optimizer isolation did not reconcile")
    if len(logs) != updates or sum(r["target_tokens"] for r in logs) != tokens:
        raise ValueError("Training log does not cover the complete token budget")
    cumulative = 0
    for update, record in enumerate(logs, 1):
        cumulative += record["target_tokens"]
        expected_rate = 1e-4 * min(update / 10.0, 1.0)
        if (record.get("optimizer_update") != update or record.get("cumulative_processed_tokens") != cumulative
                or not math.isclose(record["lr"], expected_rate, rel_tol=1e-12, abs_tol=1e-15)
                or not math.isfinite(record["loss"]) or not math.isfinite(record["grad_norm"])):
            raise ValueError("Training log update, token, warmup or finite-loss check failed")
    return tokens

def train(c, method, seed, attempt, smoke=False):
    verify_proof(c)
    selected, selection = load_selection(c, method)
    if seed not in SEEDS:
        raise ValueError("Continuation seed must be 42, 137 or 2026")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise ValueError("Use a simple new attempt name")
    protocol_hash = require_prepared(c)
    smoke_key = digest({"identity": c.identity, "selection": digest(selection), "method": method, "seed": seed})
    if not smoke:
        smoke_dir = run_directory(c, method, seed, attempt, smoke=True)
        validate_completion(c, smoke_dir, method, seed, smoke=True)
    torch = gpu()
    parent_record = read(c.runtime / c.entry["records_relative_path"] / "run_config.json")
    for name, version in parent_record["environment"]["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Training environment differs from original parent: {name}; keep the working B environment")
    p2 = c.common.load_person2()
    p2.select_model(c.a.model)
    tok = p2.get_tokenizer()
    if digest(tok.backend_tokenizer.to_str()) != verify_proof(c)["tokenizer_sha256"]:
        raise ValueError("Training tokenizer differs from verification")
    core = p2.core
    dest = run_directory(c, method, seed, attempt, smoke=smoke)
    if dest.exists():
        raise ValueError("Existing training attempt preserved; choose a new attempt")
    training_rows = selected[:32] if smoke else selected
    inputs = artifact(c, "training_inputs", method, digest(selection)[:12], "smoke" if smoke else "full")
    save_rows(inputs / "train.jsonl", training_rows)
    for name in ("prompt_template.json", "tokenization_policy.json"):
        save_bytes(inputs / name, (c.condition / "training_inputs" / name).read_bytes())
    # Retain B's initialization/update/save loop. Change only its reporting
    # label so both methods describe the adopted screened experiment.
    execution_source = training_source_for_method(c.train_source, method)
    exec(compile(execution_source, str(c.runtime / "continuation.py") + "::reused_train", "exec"), core.__dict__)
    original_parent = c.root / c.entry["adapter_relative_path"]
    parent_groups = {r["problem_group_id"] for r in c.parent}
    encoded_preview = [p2.encode(r, tok, c.contract["prompt_template"], c.contract["max_total_tokens"])
                       for r in training_rows]
    label_map = {r["row_id"]: r for r in c.labels}
    for row, encoded in zip(training_rows, encoded_preview):
        if (encoded["counts"]["supervised_tokens"] != row["supervised_tokens"]
                or digest({k: encoded[k] for k in ("input_ids", "labels")}) != label_map[row["row_id"]]["encoded_sha256"]):
            raise ValueError("Actual selected-row tokenization differs from original verified labels")
    expected_tokens = sum(e["counts"]["supervised_tokens"] for e in encoded_preview)
    def guard(args, actual_rows, current):
        if (actual_rows != training_rows or args.arm != method or args.mode != "branch"
                or (core.MODEL, core.REVISION) != (c.entry["model"], c.entry["revision"])):
            raise ValueError("Training rows/model/method differ")
        if any(r["problem_group_id"] in parent_groups for r in actual_rows):
            raise ValueError("Selected examples overlap parent groups")
        return None
    def budget_check(args, actual_rows, encoded, parent_config, parent_state):
        actual = sum(e["counts"]["supervised_tokens"] for e in encoded)
        if actual != expected_tokens or (not smoke and abs(actual - TARGET) > TARGET * Fraction(5, 1000)):
            raise ValueError("Trainer label exposure differs")
        return {"actual_supervised_tokens": actual, "target_supervised_tokens": TARGET,
                "smoke_only": smoke, "parent_historical_processed_tokens": parent_state["state"]["processed_tokens"]}
    core.provenance_guard = guard
    core.parent_budget_check = budget_check
    core.get_tokenizer = p2.get_tokenizer
    code_hash = hashlib.sha256(execution_source.encode()).hexdigest()
    core.fingerprint = lambda args: {"manifest_sha256": sha(args.data), "template_sha256": sha(args.template),
          "policy_sha256": sha(args.policy), "model": c.entry["model"], "revision": c.entry["revision"],
          "code_sha256": code_hash, "wrapper_sha256": sha(__file__),
          "reference_code_sha256": {p: sha(c.runtime / "vendor/capstone_person1" / p)
                                    for p in ("__init__.py", "person1_data_contract.py", "pipeline_controls.py", "instance_matching.py")}}
    summary_parent = read(c.runtime / c.entry["records_relative_path"] / "training_summary.json")
    if summary_parent.get("complete") is not True:
        raise ValueError("Original parent training is incomplete")
    args = SimpleNamespace(data=str(inputs / "train.jsonl"), template=str(inputs / "prompt_template.json"),
        policy=str(inputs / "tokenization_policy.json"), verification=None, out=str(dest), mode="branch", arm=method,
        parent=str(original_parent.parent), epochs=1, seed=seed, accumulation=8, lr=1e-4,
        save_every=c.cfg["training"]["checkpoint_every_updates"], resume=None, stop_after=4 if smoke else None,
        clearance=None, smoke=smoke, parent_metadata={"state": {"seen_groups": sorted(parent_groups),
        "processed_tokens": summary_parent["processed_tokens"]}, "one_pass_supervised_tokens": TARGET},
        continuation_metadata={"method": method, "bridge_identity": c.identity, "selection_sha256": digest(selection),
        "smoke_only": smoke, "screening_protocol_sha256": protocol_hash, "continuation_training_seed": seed,
        "shared_universe_sha256": BANK_SHA, "initial_optimizer_state": "empty", "fresh_warmup_updates": 10,
        "protocol_version": VERSION, "parent_training_seed": c.a.parent_seed, "original_matcher_flags_resolved": False,
        "original_parent_adapter_sha256": c.entry["expected_adapter_sha256"]})
    core.train(args)
    result = read(dest / "training_summary.json")
    if (not result.get("complete") or result["processed_tokens"] != expected_tokens
            or set(result["seen_ids"]) != {r["row_id"] for r in training_rows}
            or set(result["seen_groups"]) != {r["problem_group_id"] for r in training_rows}):
        raise ValueError("Training consumption/IDs did not reconcile")
    check_hash(original_parent / "adapter_model.safetensors", c.entry["expected_adapter_sha256"])
    checkpoint = Path(result["checkpoint"])
    changed = verify_adapter_update(original_parent / "adapter_model.safetensors", checkpoint / "adapter/adapter_model.safetensors")
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = core.build_model(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                adapter=checkpoint / "adapter", trainable=False)
    first = p2.encode(training_rows[0], tok, c.contract["prompt_template"], c.contract["max_total_tokens"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    observed = core.probe_logits(reloaded, first, dtype)
    expected = torch.load(checkpoint / "probe.pt", map_location="cpu", weights_only=True)
    if not torch.allclose(observed, expected, rtol=1e-3, atol=1e-3):
        raise ValueError("Saved adapter reload probe differs")
    report = {"passed": True, "identity": c.identity, "method": method, "seed": seed, "smoke_only": smoke,
              "smoke_key": smoke_key, "reload_probe_matched": True, "changed_adapter_tensors": changed,
              "actual_tokens": expected_tokens, "unique_examples": len(training_rows), "selection_sha256": digest(selection),
              "parent_sha256": c.entry["expected_adapter_sha256"], "checkpoint": str(checkpoint),
              "adapter_sha256": sha(checkpoint / "adapter/adapter_model.safetensors"),
              "run_config_sha256": sha(dest / "run_config.json"), "summary_sha256": sha(dest / "training_summary.json"),
              "screening_protocol_sha256": protocol_hash, "training_seconds": result["training_seconds"],
              "version": VERSION, "parent_seed": c.a.parent_seed, "bank_sha256": BANK_SHA,
              "original_matcher_flags_resolved": False}
    save(dest / "completion.json", report)
    validate_completion(c, dest, method, seed, smoke=smoke)
    print(json.dumps({**report, "run_directory": str(dest)}, indent=2))

def validate_completion(c, directory, method, seed, smoke=False):
    protocol_hash = require_prepared(c)
    selected, selection = load_selection(c, method)
    selected = selected[:32] if smoke else selected
    directory = Path(directory).resolve()
    if directory.parent != artifact(c, "smokes" if smoke else "runs", method).resolve():
        raise ValueError("Training completion is outside this frozen experiment")
    receipt = read(directory / "completion.json")
    expected = {
        "passed": True, "identity": c.identity, "method": method, "seed": seed,
        "smoke_only": smoke, "reload_probe_matched": True, "version": VERSION,
        "parent_seed": c.a.parent_seed, "bank_sha256": BANK_SHA,
        "selection_sha256": digest(selection), "parent_sha256": c.entry["expected_adapter_sha256"],
        "screening_protocol_sha256": protocol_hash, "unique_examples": len(selected),
        "actual_tokens": sum(r["supervised_tokens"] for r in selected),
        "original_matcher_flags_resolved": False,
        "smoke_key": digest({"identity": c.identity, "selection": digest(selection), "method": method, "seed": seed}),
    }
    if any(receipt.get(k) != value for k, value in expected.items()) or not receipt.get("changed_adapter_tensors"):
        raise ValueError("Training completion has the wrong protocol, method, seed, selection or smoke status")
    check_hash(directory / "run_config.json", receipt["run_config_sha256"])
    check_hash(directory / "training_summary.json", receipt["summary_sha256"])
    config = read(directory / "run_config.json")
    summary = read(directory / "training_summary.json")
    wanted_config = {
        "model": c.entry["model"], "revision": c.entry["revision"], "seed": seed,
        "epochs": 1, "accumulation": 8, "learning_rate": 1e-4, "warmup_updates": 10,
        "microbatch_size": 1, "packing": False, "arm": method,
        "shared_universe_sha256": BANK_SHA, "parent_adapter_sha256": c.entry["expected_adapter_sha256"],
        "original_parent_adapter_sha256": c.entry["expected_adapter_sha256"],
        "bridge_identity": c.identity, "protocol_version": VERSION, "parent_training_seed": c.a.parent_seed,
        "selection_sha256": digest(selection), "screening_protocol_sha256": protocol_hash,
        "initial_optimizer_state": "empty", "fresh_warmup_updates": 10,
        "branch_optimizer_policy": "fresh_optimizer_and_scheduler", "smoke_only": smoke,
        "wrapper_sha256": sha(__file__),
        "code_sha256": hashlib.sha256(training_source_for_method(c.train_source, method).encode()).hexdigest(),
    }
    if any(config.get(k) != value for k, value in wanted_config.items()):
        raise ValueError("Executed training configuration differs from the frozen recipe")
    inputs = artifact(c, "training_inputs", method, digest(selection)[:12], "smoke" if smoke else "full")
    check_hash(inputs / "train.jsonl", config["manifest_sha256"])
    if rows(inputs / "train.jsonl") != selected or (not smoke and config["manifest_sha256"] != sha(artifact(c, "selection", method, "train.jsonl"))):
        raise ValueError("Executed exposure differs from frozen selected examples")
    validate_consumption(summary, selected, rows(directory / "training_log.jsonl"))
    checkpoint = Path(summary["checkpoint"]).resolve()
    if checkpoint != directory / "final" or receipt["checkpoint"] != str(checkpoint):
        raise ValueError("Expected a complete final checkpoint from this run")
    if read(checkpoint / "run_config.json") != config:
        raise ValueError("Checkpoint and run configuration differ")
    check_hash(checkpoint / "adapter/adapter_model.safetensors", receipt["adapter_sha256"])
    check_hash(c.root / c.entry["adapter_relative_path"] / "adapter_model.safetensors",
               c.entry["expected_adapter_sha256"])
    return receipt

def units():
    return [{'id':f'B_{size}_p{parent}_t{seed}','model':size,'parent_seed':parent,'seed':seed}
            for size,parent in PROFILES for seed in SEEDS
            if (size,parent,seed)!=('0.5B',17,42)]

def profile_args(a,unit):
    return SimpleNamespace(**{**vars(a),'model':unit['model'],'parent_seed':unit['parent_seed']})

def find_training(c,method,seed):
    if method!='B':raise IntegrityError('Only B controls are authorized')
    if (c.a.model,c.a.parent_seed,seed)==('0.5B',17,42):return original_pilot(c.a)
    for number in range(1,MAX_ATTEMPTS+1):
        directory=run_directory(c,'B',seed,f'attempt{number:02d}')
        if (directory/'completion.json').is_file():
            return validate_completion(c,directory,'B',seed),str(directory)
    return None

def train_unit(a):
    if (a.model,a.parent_seed,a.seed)==('0.5B',17,42):
        original_pilot(a);return
    c=load_context(a);prepare(c)
    if find_training(c,'B',a.seed):return
    errors=[]
    for number in range(1,MAX_ATTEMPTS+1):
        attempt=f'attempt{number:02d}'
        dest=run_directory(c,'B',a.seed,attempt)
        smoke=run_directory(c,'B',a.seed,attempt,smoke=True)
        if dest.exists():continue
        if smoke.exists() and not (smoke/'completion.json').is_file():continue
        print('Attempt',attempt,'original parent',c.entry['expected_adapter_sha256'],flush=True)
        commands=[]
        if (smoke/'completion.json').is_file():validate_completion(c,smoke,'B',a.seed,smoke=True)
        else:commands.append(child(a,'train-one',seed=a.seed,attempt=attempt,smoke=True))
        commands.append(child(a,'train-one',seed=a.seed,attempt=attempt))
        failed=False
        for command in commands:
            result=subprocess.run(command)
            if result.returncode in (65,66):raise IntegrityError('A training integrity gate failed; see stage log')
            if result.returncode:
                errors.append({'attempt':attempt,'returncode':result.returncode})
                failed=True;break
        if not failed:
            if not find_training(c,'B',a.seed):raise IntegrityError('Trainer returned without valid completion')
            return
    raise SelectionBlocked('Three attempts preserved; cannot complete this run automatically: '+json.dumps(errors))

def collect(a):
    items=[];pilot,pilot_dir=original_pilot(a)
    items.append({'model_size':'0.5B','parent_seed':17,'continuation_seed':42,'complete':True,
                  'reused_pilot':True,'run_directory':pilot_dir,'receipt':pilot})
    for size,parent in PROFILES:
        aa=SimpleNamespace(**{**vars(a),'model':size,'parent_seed':parent})
        c=load_context(aa);require_prepared(c)
        for seed in SEEDS:
            if (size,parent,seed)==('0.5B',17,42):continue
            found=find_training(c,'B',seed)
            items.append({'model_size':size,'parent_seed':parent,'continuation_seed':seed,
                          'complete':bool(found),'reused_pilot':False,
                          'run_directory':found[1] if found else '',
                          'receipt':found[0] if found else None})
    return sorted(items,key=lambda r:(r['model_size'],r['parent_seed'],r['continuation_seed']))

def write_status(a,state=None,override=None):
    root=control_root(a)
    if (root/'TRAINING_COMPLETE.json').is_file():return validate_final_marker(a)
    all_rows=collect(a);done=sum(r['complete'] and not r['reused_pilot'] for r in all_rows)
    report={'version':VERSION,'status':override or ('training_complete' if done==11 else 'in_progress'),
            'B_screened_complete_of_12':done+1,'new_B_training_complete_of_11':done,
            'pilot_reused':True,'all_requested_training_complete':done==11,
            'candidate_bank_rows':4955,'candidate_bank_sha256':BANK_SHA,
            'selected_rows_per_run':1000,'selected_supervised_tokens':ACTUAL_B_TOKENS,
            'target_supervised_tokens':TARGET,'selected_manifest_sha256':B_MANIFEST_SHA,
            'benchmark_evaluation_performed':False,'reserved_final_tests_performed':False,
            'original_matcher_flags_resolved':False,'control_directory':str(root),
            'updated_utc':utc(),'unit_state':state or {},
            'next_step':'Evaluate the eleven new B checkpoints on the frozen 500 GSM and 500 MATH development questions before comparing all methods.'}
    current_json(root/'STATUS.json',report)
    csvbuf=io.StringIO();fields=['model_size','parent_seed','continuation_seed','complete','reused_pilot','run_directory','adapter_sha256','bank_sha256','manifest_sha256','actual_tokens']
    writer=csv.DictWriter(csvbuf,fieldnames=fields);writer.writeheader()
    checkpoints=[]
    for row in all_rows:
        co=row['receipt'];writer.writerow({**{k:row[k] for k in fields if k in row},
            'adapter_sha256':co['adapter_sha256'] if co else '', 'bank_sha256':BANK_SHA,
            'manifest_sha256':B_MANIFEST_SHA,'actual_tokens':ACTUAL_B_TOKENS})
        if co:
            checkpoints.append({'key':f"B_screened_{row['model_size']}_p{row['parent_seed']}_t{row['continuation_seed']}",
                'method':'B_screened','model_size':row['model_size'],'parent_seed':row['parent_seed'],
                'continuation_seed':row['continuation_seed'],'adapter':str(Path(co['checkpoint'])/'adapter'),
                'checkpoint_sha256':co['adapter_sha256'],'parent_sha256':co['parent_sha256'],
                'manifest_sha256':B_MANIFEST_SHA,'bank_sha256':BANK_SHA,
                'actual_supervised_tokens':ACTUAL_B_TOKENS,'reused_original_pilot':row['reused_pilot'],
                'training_receipt':str(Path(row['run_directory'])/'completion.json')})
    # Controller summaries are replaceable; checkpoints and attempts are immutable.
    current_bytes=csvbuf.getvalue().encode();tmp=root/('TRAINING_RUNS.csv.'+uuid.uuid4().hex+'.partial')
    with tmp.open('xb') as f:f.write(current_bytes);f.flush();os.fsync(f.fileno())
    os.replace(tmp,root/'TRAINING_RUNS.csv')
    current_json(root/'SCREENED_B_CHECKPOINTS.json',{'version':VERSION,'complete':done==11,
                 'benchmark_evaluation_performed':False,'checkpoints':checkpoints})
    return report

def zip_bytes(members):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    return buffer.getvalue()

def ensure_handoff(a):
    root=control_root(a);mark=read(root/'TRAINING_COMPLETE.json')
    members={n:(root/n).read_bytes() for n in ['STATUS.json','TRAINING_COMPLETE.json','TRAINING_RUNS.csv','SCREENED_B_CHECKPOINTS.json',mark['results_zip']]}
    destination=root/('B_Screened_4955_Training_Handoff_'+root.name+'.zip')
    save_bytes(destination,zip_bytes(members));return destination

def finish(a,state):
    root=control_root(a);report=write_status(a,state,'training_complete')
    if not report['all_requested_training_complete']:raise IntegrityError('Cannot issue completion for missing runs')
    all_rows=collect(a);members={}
    for name in ['STATUS.json','TRAINING_RUNS.csv','SCREENED_B_CHECKPOINTS.json']:
        members[name]=(root/name).read_bytes()
    for path in [HERE/'protocol.json',HERE/'PACKAGE_PINS.json']:
        members['package/'+path.name]=path.read_bytes()
    # One selected training manifest, all profile freezes, and every training receipt.
    members['selection/B/train.jsonl']=(pilot_root(a.work)/'selection/B/train.jsonl').read_bytes()
    for path in root.glob('profiles/*/*/freeze.json'):members[str(path.relative_to(root))]=path.read_bytes()
    for path in root.glob('profiles/*/*/selection/B/selection.json'):members[str(path.relative_to(root))]=path.read_bytes()
    for r in all_rows:
        directory=Path(r['run_directory'])
        prefix='reused_pilot/B_seed42/' if r['reused_pilot'] else str(directory.relative_to(root))+'/'
        for name in ['completion.json','run_config.json','training_summary.json','training_log.jsonl']:
            members[prefix+name]=(directory/name).read_bytes()
    for path in (root/'submissions').glob('*.json'):members['submissions/'+path.name]=path.read_bytes()
    content=zip_bytes(members);digest_hex=hashlib.sha256(content).hexdigest()
    archive_name='B_training_receipts_'+digest_hex[:16]+'.zip';save_bytes(root/archive_name,content)
    marker={'version':VERSION,'B_total':12,'new_runs':11,'benchmark_evaluation_performed':False,
            'results_zip':archive_name,'results_zip_sha256':digest_hex,
            'status_report_sha256':sha(root/'STATUS.json'),'runs_csv_sha256':sha(root/'TRAINING_RUNS.csv'),
            'checkpoints_sha256':sha(root/'SCREENED_B_CHECKPOINTS.json'),'package_sha256':package_check()}
    save(root/'TRAINING_COMPLETE.json',marker);validate_final_marker(a)
    return ensure_handoff(a)

def validate_final_marker(a):
    root=control_root(a);m=read(root/'TRAINING_COMPLETE.json')
    if m.get('version')!=VERSION or m.get('new_runs')!=11 or m.get('B_total')!=12 or m.get('package_sha256')!=package_check():
        raise IntegrityError('Completion marker belongs to another package or experiment')
    for name,key in [('STATUS.json','status_report_sha256'),('TRAINING_RUNS.csv','runs_csv_sha256'),('SCREENED_B_CHECKPOINTS.json','checkpoints_sha256'),(m['results_zip'],'results_zip_sha256')]:
        check_hash(root/name,m[key])
    report=read(root/'STATUS.json')
    if (not report.get('all_requested_training_complete') or report.get('new_B_training_complete_of_11')!=11
        or report.get('benchmark_evaluation_performed') is not False):raise IntegrityError('Invalid training completeness claim')
    if sum(r['complete'] and not r['reused_pilot'] for r in collect(a))!=11:
        raise IntegrityError('Checkpoint validation no longer supports completion')
    return report

def preflight_profile(a):
    c=load_context(a);environment_check(c)
    from huggingface_hub import snapshot_download
    snapshot=Path(snapshot_download(c.entry['model'],revision=c.entry['revision'],local_files_only=True))
    for name in ['config.json','tokenizer.json']:
        if not (snapshot/name).is_file():raise IntegrityError('Missing pinned offline base/tokenizer file: '+name)
    if not list(snapshot.glob('*.safetensors')) and not list(snapshot.glob('pytorch_model*.bin')):
        raise IntegrityError('Pinned offline base-model weights missing')
    prepare(c)
    return {'profile':c.profile,'output':str(artifact(c)),'selected_rows':1000,
            'actual_tokens':ACTUAL_B_TOKENS,'candidate_rows':len(c.bank),
            'bank_sha256':BANK_SHA,'selection_sha256':B_MANIFEST_SHA,'GPU_training_performed':False}

def preflight(a):
    package_check()
    if not 1<=a.allocations<=8 or not 1800<=a.worker_seconds<=13800:
        raise IntegrityError('Use 1-8 allocations and 1800-13800 worker seconds')
    for cmd in ['sbatch','squeue','scontrol','scancel']:
        if shutil.which(cmd) is None:raise IntegrityError('Run on Sol with Slurm available: '+cmd)
    wanted=Path(a.b_root)/'venv_b_training_py312/bin/python'
    if Path(sys.executable).absolute()!=wanted.absolute() and Path(sys.prefix)!=wanted.parent.parent:
        raise IntegrityError('Use the existing B training Python: '+str(wanted))
    original_pilot(a)
    checked=[]
    for size,parent in PROFILES:
        aa=SimpleNamespace(**{**vars(a),'model':size,'parent_seed':parent})
        result=subprocess.run(child(aa,'preflight-profile'),capture_output=True,text=True)
        if result.returncode:raise IntegrityError(f'Preflight failed for {size} parent {parent}:\n'+result.stdout+result.stderr)
        checked.append(json.loads(result.stdout.strip().splitlines()[-1]))
    root=control_root(a);root.mkdir(parents=True,exist_ok=True)
    save(root/'PREFLIGHT.json',{'version':VERSION,'passed':True,'new_B_training_runs':11,
        'existing_B_pilot_reused':True,'benchmark_evaluation_requested':False,'profiles':checked})
    if not (root/'STATUS.json').exists():write_status(a,{},'prepared_not_submitted')
    print(json.dumps({'preflight_passed':True,'profiles':checked,'new_B_runs':11,
        'benchmark_evaluation':False,'control_directory':str(root)},indent=2),flush=True)

def queue_rows():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%r'],capture_output=True,text=True)
    if result.returncode:raise IntegrityError('Cannot establish active jobs; refusing duplicate submission: '+result.stderr)
    return {parts[0]:parts[1:] for line in result.stdout.splitlines() if len(parts:=line.split('|',2))==3}

def release_jobs(ledger,path,active):
    released=ledger.setdefault('released_job_ids',[])
    for job in ledger['jobs']:
        jid=job['job_id']
        if jid in released:continue
        details=active.get(jid)
        if details and details[0]=='PENDING' and details[1]=='JobHeldUser':
            subprocess.run(['scontrol','release',jid],check=True)
        elif details and details[0]=='PENDING' and 'Held' in details[1]:
            raise IntegrityError('Scheduler/admin hold requires review: '+jid+' '+details[1])
        elif not details:
            result=subprocess.run(['scontrol','show','job',jid,'-o'],capture_output=True,text=True)
            if result.returncode:
                raise IntegrityError('Cannot establish whether unreleased job is held: '+jid+'; preserve submission and check its scheduler state before retrying')
            if 'Reason=JobHeldUser' in result.stdout:
                subprocess.run(['scontrol','release',jid],check=True)
            elif any(term in result.stdout for term in ('Reason=JobHeldAdmin','Reason=JobHeld')):
                raise IntegrityError('Scheduler/admin hold requires review: '+jid)
        released.append(jid);current_json(path,ledger)

def submit(a):
    preflight(a);root=control_root(a)
    with locked(root/'SUBMIT.lock'):
        active=queue_rows()
        for path in sorted((root/'submissions').glob('submission_*.json'),reverse=True):
            old=read(path)
            if old.get('pending_sbatch_slot') is not None:
                raise IntegrityError('A prior sbatch outcome is uncertain. Preserve '+str(path)+' and check squeue before any new submission.')
            if any(j['job_id'] in active for j in old['jobs']):
                release_jobs(old,path,active)
                current_json(HERE/'RUN_LOCATION.json',{'control_directory':str(root),'submission_file':str(path),'jobs':old['jobs']})
                print(json.dumps({'reused_active_submission':True,'jobs':old['jobs'],'control_directory':str(root),
                    'all_jobs_released':len(old['released_job_ids'])==len(old['jobs'])},indent=2));return
        if (root/'TRAINING_COMPLETE.json').is_file():
            validate_final_marker(a);print('Already complete; no new job submitted.');return
        index=len(list((root/'submissions').glob('submission_*.json')))+1
        path=root/'submissions'/f'submission_{index:03d}.json'
        logs=root/'slurm_logs';logs.mkdir(parents=True,exist_ok=True)
        base=['sbatch','--parsable','--hold','--account',a.account,'--partition',a.partition,'--qos',a.qos,
              '--nodes','1','--ntasks','1','--cpus-per-task','4','--mem','32G','--gpus','a100:1',
              '--time','04:00:00','--signal','B:TERM@180','--chdir',str(Path(a.work)),
              '--job-name','bs4955-'+root.name[:8]+'-'+str(index),
              '--output',str(logs/'%j.out'),'--error',str(logs/'%j.err')]
        script=[str(HERE/'gpu_worker.sbatch'),str(HERE),str(Path(a.work)),str(Path(a.b_root))]
        test=subprocess.run([*base,'--test-only',*script,str(path),'0',str(a.worker_seconds)],text=True,capture_output=True)
        if test.returncode:raise IntegrityError('Slurm rejected resources before submitting jobs:\n'+test.stderr+test.stdout)
        ledger={'version':VERSION,'package_sha256':package_check(),'control_directory':str(root),
                'created_utc':utc(),'jobs':[],'released_job_ids':[],'submission_complete':False}
        current_json(path,ledger)
        for slot in range(a.allocations):
            command=list(base)
            if ledger['jobs']:command+=['--dependency','afterany:'+ledger['jobs'][-1]['job_id']]
            command+=script+[str(path),str(slot),str(a.worker_seconds)]
            ledger['pending_sbatch_slot']=slot;current_json(path,ledger)
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode:
                # No automatic resubmission after an uncertain submission outcome.
                ledger['submission_error']=result.stderr+result.stdout;current_json(path,ledger)
                raise IntegrityError('sbatch failed or its outcome is uncertain; submitted jobs remain held. See '+str(path))
            jid=result.stdout.strip().split(';')[0]
            if not jid.isdigit():
                ledger['submission_error']='Unrecognized sbatch response: '+result.stdout;current_json(path,ledger)
                raise IntegrityError('Unknown sbatch result; inspect the saved ledger and queue before retrying')
            ledger['jobs'].append({'slot':slot,'job_id':jid,'stdout':str(logs/(jid+'.out')),'stderr':str(logs/(jid+'.err'))})
            ledger.pop('pending_sbatch_slot',None);current_json(path,ledger)
        ledger['submission_complete']=True;current_json(path,ledger)
        current_json(HERE/'RUN_LOCATION.json',{'control_directory':str(root),'submission_file':str(path),'jobs':ledger['jobs']})
        release_jobs(ledger,path,queue_rows())
        print(json.dumps({'status':'submitted_training_not_yet_verified','jobs':ledger['jobs'],
            'all_jobs_released':len(ledger['released_job_ids'])==len(ledger['jobs']),
            'control_directory':str(root),'new_training_runs':11,'reused_screened_B_pilot':True,
            'benchmark_evaluation':False,'note':'After this cell returns with job IDs and all_jobs_released true, the laptop and notebook may disconnect.'},indent=2),flush=True)

def terminate_process(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def cancel_future(submission,slot):
    active=queue_rows()
    ids=[j['job_id'] for j in submission['jobs'] if j['slot']>slot and j['job_id'] in active and active[j['job_id']][0]=='PENDING']
    if ids:subprocess.run(['scancel',*ids],check=True)

def worker(a):
    root=control_root(a);submission=read(a.submission)
    if submission.get('package_sha256')!=package_check() or submission.get('control_directory')!=str(root):
        raise IntegrityError('Submission belongs to another package or output directory')
    jid=os.environ.get('SLURM_JOB_ID','')
    if not any(j['job_id']==jid and j['slot']==a.slot for j in submission['jobs']):
        raise IntegrityError('Worker must be the recorded Slurm allocation')
    interrupted={'stop':False,'process':None}
    def handler(signum,frame):
        interrupted['stop']=True
        if interrupted['process'] is not None:terminate_process(interrupted['process'])
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    with locked(root/'WORKER.lock'):
        if (root/'TRAINING_COMPLETE.json').is_file():
            validate_final_marker(a);ensure_handoff(a);cancel_future(submission,a.slot);return 0
        state=read(root/'UNIT_STATE.json') if (root/'UNIT_STATE.json').exists() else {}
        deadline=time.monotonic()+a.worker_seconds
        for unit in units():
            aa=profile_args(a,unit);c=load_context(aa);require_prepared(c)
            if find_training(c,'B',unit['seed']):
                state[unit['id']]={'status':'complete'};continue
            if interrupted['stop']:break
            if state.get(unit['id'],{}).get('status')=='blocked':continue
            if deadline-time.monotonic()<1800:
                print('Passing remaining training to the next pre-submitted allocation',flush=True);break
            log=root/'stage_logs'/(unit['id']+'_'+utc().replace(':','-')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            state[unit['id']]={'status':'running','job_id':jid,'log':str(log)}
            current_json(root/'UNIT_STATE.json',state);write_status(a,state)
            with log.open('x') as stream:
                process=subprocess.Popen(child(aa,'train-unit',seed=unit['seed']),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                interrupted['process']=process
                try:code=process.wait(timeout=max(1,deadline-time.monotonic()-120))
                except subprocess.TimeoutExpired:terminate_process(process);code=-signal.SIGTERM
                finally:interrupted['process']=None
            if code==0:
                if not find_training(c,'B',unit['seed']):raise IntegrityError('Stage returned without a valid checkpoint receipt')
                state[unit['id']]={'status':'complete','log':str(log),'job_id':jid}
            elif code in (65,66):
                state[unit['id']]={'status':'blocked','returncode':code,'log':str(log),'reason':'Training integrity gate or attempt limit failed; no automatic relaxation'}
            else:
                state[unit['id']]={'status':'interrupted' if code<0 else 'retry_pending','returncode':code,'log':str(log)}
            current_json(root/'UNIT_STATE.json',state);write_status(a,state)
            if interrupted['stop'] or code<0:break
        report=write_status(a,state)
        blocked=any(v.get('status')=='blocked' for v in state.values())
        if report['all_requested_training_complete']:
            handoff=finish(a,state);cancel_future(submission,a.slot)
            print('SCREENED B TRAINING COMPLETE; no benchmark evaluation. Handoff: '+str(handoff),flush=True);return 0
        pending=[u for u in units() if state.get(u['id'],{}).get('status') not in ('complete','blocked')]
        label='incomplete_blocked' if not pending and blocked else ('incomplete_allocation_chain_exhausted' if a.slot==max(j['slot'] for j in submission['jobs']) else 'continuing_in_next_allocation')
        write_status(a,state,label)
        if not pending:cancel_future(submission,a.slot)
        print('B training status: '+label,flush=True)
        return 2 if not pending and blocked else 0

def status(a):
    root=control_root(a);location=HERE/'RUN_LOCATION.json'
    if location.exists():
        info=read(location);ids=[j['job_id'] for j in info.get('jobs',[])]
        if info.get('control_directory')!=str(root):raise IntegrityError('RUN_LOCATION belongs to a different package/output directory')
        if ids:
            for cmd in [['squeue','-j',','.join(ids),'-o','%.12i %.12T %.12M %.45R'],['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,Timelimit,ExitCode']]:
                if shutil.which(cmd[0]):
                    result=subprocess.run(cmd,capture_output=True,text=True);print(result.stdout or result.stderr)
    if (root/'TRAINING_COMPLETE.json').exists():value=validate_final_marker(a)
    elif (root/'STATUS.json').exists():value=read(root/'STATUS.json')
    else:
        print('No training status yet. Run the notebook preflight and submission cells.');return
    print(json.dumps({k:value[k] for k in ['status','B_screened_complete_of_12','new_B_training_complete_of_11','candidate_bank_rows','selected_rows_per_run','benchmark_evaluation_performed']},indent=2))
    for key,s in value.get('unit_state',{}).items():
        if s.get('status')!='complete':print(key,json.dumps(s))
        if s.get('status')=='running' and s.get('log'):
            path=Path(s['log'])
            if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                print('Latest stage output:\n'+'\n'.join(path.read_text(errors='replace').splitlines()[-20:]))
    print('Output:',root);print('Run table:',root/'TRAINING_RUNS.csv')
    print('Completion marker:',root/'TRAINING_COMPLETE.json')
    print('Training handoff:',root/('B_Screened_4955_Training_Handoff_'+root.name+'.zip'))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','preflight-profile','submit','worker','train-unit','train-one','status'])
    p.add_argument('--work',default='/scratch/sshind38/capstone_c0_c1_work')
    p.add_argument('--b-root',default='/scratch/sshind38/capstone_approach_b')
    p.add_argument('--model',choices=['0.5B','1.5B'],default='0.5B')
    p.add_argument('--parent-seed',type=int,choices=[17,123],default=17)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--attempt',default='attempt01');p.add_argument('--smoke',action='store_true')
    p.add_argument('--allocations',type=int,default=2);p.add_argument('--worker-seconds',type=int,default=12600)
    p.add_argument('--account',default='class_fse57081354fall2026');p.add_argument('--partition',default='public');p.add_argument('--qos',default='class')
    p.add_argument('--submission');p.add_argument('--slot',type=int,default=0)
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    if a.action=='preflight':preflight(a)
    elif a.action=='preflight-profile':print(json.dumps(preflight_profile(a)))
    elif a.action=='submit':submit(a)
    elif a.action=='status':status(a)
    elif a.action=='worker':return worker(a)
    elif a.action=='train-unit':train_unit(a)
    elif a.action=='train-one':
        if (a.model,a.parent_seed,a.seed)==('0.5B',17,42):raise IntegrityError('Completed pilot is read-only and cannot be retrained by this package')
        c=load_context(a);require_prepared(c);train(c,'B',a.seed,a.attempt,smoke=a.smoke)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except SelectionBlocked as e:print('BLOCKED: '+str(e),file=sys.stderr);raise SystemExit(66)
    except (IntegrityError,ValueError,FileNotFoundError) as e:print('INTEGRITY CHECK FAILED: '+str(e),file=sys.stderr);raise SystemExit(65)

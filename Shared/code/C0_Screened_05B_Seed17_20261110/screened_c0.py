#!/usr/bin/env python3
"""Matched screened B/C0 pilot, Qwen2.5-0.5B parent 17, continuation 42.

This is a new protocol. The original bridge and its audit gates are immutable.
Only its verified input loading, serialization helpers and parent trainer are
reused; no original-protocol clearance is issued or bypassed.
"""
import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent
VERSION = "independent-screened-c0-pilot-v1"
PROFILE = "0.5B_seed17"
CONTINUATION_SEED = 42
TARGET = 318845
BANK_SHA = "ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f"
LEGACY_SHA = "47dd44c53526ef10ca8056b65ef215988e028d2047a04b549966c849db32bc6d"
PREVIEW_ID = "35f8a5ecbf57cd37aa6518d1cc65e0de8b72b5dfb9d549547b799ffda3996321"
ORIGINAL_ID = "053cf70a0bb7f987308adfecfaeb5d05b5f25b7ca5542d285110d6bab4a08aea"
MANIFESTS = {
    "B": ("04112f96b282ba8a58fcd8900f6ca91b656dbac9abbd17a4cdef2b02a9bb00e0", 317486),
    "C0": ("71e4863eba97c84e8f084ebfc58f6fae4ecde3e0678e6b1e927bf887cf312478", 320273),
}
BENCHMARKS = {
    "dev_gsm": "3cb2c71007d23e01cb70b6370c1ad82ad5bcea3527c7f3258b4814cf2e3d1cc6",
    "dev_math": "641460fdc55beff591daad0cc7530a3f933eb263c3dfdf8a0534fcfaf8196416",
}
LIMITATIONS = [
    "New conservative screen; the unavailable original 25-exclusion/572-quarantine audit is not resolved.",
    "Quarantine is not proof of contamination or mathematical invalidity; below-threshold repetitions can remain.",
    "Inherited parent/development exposure remains disclosed; these are exposure-limited development results.",
    "One parent seed (17) and one continuation seed (42); no estimate of training-seed variability.",
    "Reserved final benchmarks, C1 and other model sizes/seeds are outside this pilot.",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return expected


def save_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ValueError(f"Different output preserved: {path}; use a new attempt/version")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)


def save(path, value):
    save_bytes(path, (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode())


def save_rows(path, records):
    save_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                             for r in records).encode())


def unique(records):
    if not records:
        raise ValueError("Empty records")
    for key in ("row_id", "problem_group_id"):
        values = [r[key] for r in records]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f"Invalid/duplicate {key}")


def artifact(c, *parts):
    return c.out / c.profile / c.identity[:16] / Path(*parts)


def cell(row):
    return row["dataset_id"], row["length_bin"]


def bounds(target, numerator, denominator):
    return ((target * (denominator - numerator) + denominator - 1) // denominator,
            target * (denominator + numerator) // denominator)


def validate_population(original, retained, screen, decisions):
    unique(original)
    unique(retained)
    by_id = {r["row_id"]: r for r in original}
    held_ids = set(screen["held_out_ids"])
    if len(held_ids) != screen["held_out_rows"] or len(held_ids) != 1454:
        raise ValueError("Screen quarantine count differs")
    if len(decisions) != len(held_ids) or {d["row_id"] for d in decisions} != held_ids:
        raise ValueError("Quarantine decisions do not cover the declared IDs")
    for decision in decisions:
        row = by_id.get(decision["row_id"])
        if row is None or hashlib.sha256(row["problem"].encode()).hexdigest() != decision["question_sha256"]:
            raise ValueError("Quarantined question content changed")
        if decision.get("completion_sha256") and hashlib.sha256(row["completion"].encode()).hexdigest() != decision["completion_sha256"]:
            raise ValueError("Previously reviewed reference content changed")
    bad_groups = {by_id[rid]["problem_group_id"] for rid in held_ids}
    wanted = [r for r in original if r["problem_group_id"] not in bad_groups]
    if (retained != wanted or len(retained) != screen["retained_rows"]
            or len(retained) != 4955 or len(original) != 6409):
        raise ValueError("Retained bank differs from complete group-level quarantine")
    return True


def validate_selection(selected, bank, parent):
    unique(selected)
    by_id = {r["row_id"]: r for r in bank}
    if len(selected) != 1000 or any(by_id.get(r["row_id"]) != r for r in selected):
        raise ValueError("Selection membership/count/content differs")
    for key in ("row_id", "problem_group_id"):
        if {r[key] for r in selected} & {r[key] for r in parent}:
            raise ValueError(f"Selection intersects parent {key}")
    required = Counter(cell(r) for r in parent)
    if len(required) != 20 or set(required.values()) != {50} or Counter(cell(r) for r in selected) != required:
        raise ValueError("Selection must retain exactly 50 rows per source/length stratum")
    parent_tokens, selected_tokens = defaultdict(int), defaultdict(int)
    for r in parent:
        parent_tokens[cell(r)] += r["supervised_tokens"]
    for r in selected:
        selected_tokens[cell(r)] += r["supervised_tokens"]
    if sum(parent_tokens.values()) != TARGET:
        raise ValueError("Parent token anchor differs")
    for key, target in parent_tokens.items():
        low, high = bounds(target, 5, 100)
        if not low <= selected_tokens[key] <= high:
            raise ValueError("Selection stratum tokens exceed 5 percent")
    low, high = bounds(TARGET, 5, 1000)
    if not low <= sum(selected_tokens.values()) <= high:
        raise ValueError("Selection global tokens exceed 0.5 percent")
    def order(row):
        h = hashlib.sha256(("42|training-order|" + row["row_id"]).encode()).digest()
        return int.from_bytes(h[:4], "big"), row["row_id"]
    if selected != sorted(selected, key=order):
        raise ValueError("Selection training order differs")
    return sum(selected_tokens.values())


def load_legacy(work, b_root):
    bridge_path = work / "c0_c1_sol_bridge_v1/sol_bridge.py"
    check_hash(bridge_path, LEGACY_SHA)
    spec = importlib.util.spec_from_file_location("_immutable_original_bridge", bridge_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    original = legacy.load_context(SimpleNamespace(
        b_root=str(b_root), repo=str(work / "repo"),
        output=str(work / "outputs/C0_C1_B_bridge_v1"), model="0.5B", parent_seed=17))
    if original.identity != ORIGINAL_ID:
        raise ValueError("Original bridge identity changed; preserve evidence and inspect")
    proof = legacy.verify_proof(original)
    scores, receipt = legacy.load_scores(original)
    return legacy, original, proof, scores, receipt


def load_context(a):
    package_pins = read(HERE / "PACKAGE_PINS.json")
    for relative, expected in package_pins.items():
        check_hash(HERE / relative, expected)
    decision = read(HERE / "protocol_decision.json")
    if (decision.get("status") != "adopted_for_this_pilot" or decision.get("parent_seed") != 17
            or decision.get("model_size") != "0.5B" or decision.get("continuation_seed") != 42
            or decision.get("original_matcher_flags_resolved") is not False):
        raise ValueError("This package only implements the explicitly adopted pilot")
    work, b_root = Path(a.work).resolve(), Path(a.b_root).resolve()
    legacy, original, proof, old_scores, receipt = load_legacy(work, b_root)
    preview = work / "outputs/C0_C1_independent_screen_preview_v1" / PROFILE / PREVIEW_ID[:16]
    report = read(preview / "preview_report.json")
    expected_report = read(HERE / "expected_preview.json")
    for key, expected in expected_report.items():
        if key == "selections":
            for method, fields in expected.items():
                if any(report[key][method].get(k) != v for k, v in fields.items()):
                    raise ValueError(f"Actual {method} preview differs from the supplied successful report")
        elif report.get(key) != expected:
            raise ValueError(f"Actual preview differs from supplied successful report: {key}")
    check_hash(preview / "candidate_bank.jsonl", BANK_SHA)
    bank = rows(preview / "candidate_bank.jsonl")
    screen = read(HERE / "screening/screening_proposal.json")
    projection = [{k: r.get(k) for k in screen["projection_fields"]} for r in original.bank]
    if digest(projection) != screen["original_projection_sha256"]:
        raise ValueError("Original question projection differs from the screened evidence")
    decisions = read(HERE / "screening/known_exclusions.json")["decisions"]
    validate_population(original.bank, bank, screen, decisions)
    if (receipt["scores_sha256"] != expected_report["score_import"]["source_sha256"]
            or receipt["measurement_key"] != expected_report["score_import"]["source_measurement_key"]
            or receipt["parent_sha256"] != original.entry["expected_adapter_sha256"]):
        raise ValueError("Reference scores came from a different parent or measurement")
    original_by_id = {r["row_id"]: r for r in original.bank}
    for score in old_scores:
        if (score.get("status") != "ok" or score["row_sha256"] != digest(original_by_id[score["row_id"]])
                or score["measurement_key"] != receipt["measurement_key"]
                or score["parent_sha256"] != original.entry["expected_adapter_sha256"]):
            raise ValueError("Reference measurement/content identity differs")
    imported_path = preview / "imported_reference_measurements.jsonl"
    bank_ids = {r["row_id"] for r in bank}
    if rows(imported_path) != [s for s in old_scores if s["row_id"] in bank_ids]:
        raise ValueError("Imported scores differ from the unchanged verified measurements")
    selected = {}
    for method, (manifest_sha, tokens) in MANIFESTS.items():
        path = preview / method / "train.jsonl"
        check_hash(path, manifest_sha)
        choice = rows(path)
        if validate_selection(choice, bank, original.rebinned) != tokens:
            raise ValueError("Selection token total differs from the supplied report")
        selected_report = read(preview / method / "selection_preview.json")
        if (selected_report.get("manifest_sha256") != manifest_sha
                or selected_report.get("method") != method or selected_report.get("rows") != 1000
                or selected_report.get("actual_tokens") != tokens or selected_report.get("target_tokens") != TARGET
                or selected_report.get("solver", {}).get("passed") is not True
                or selected_report["solver"].get("tolerance_relaxed") is not False):
            raise ValueError("Selection solver receipt differs")
        selected[method] = (choice, selected_report)
    original_labels = legacy.artifact(original, "verification/actual_labels.jsonl")
    labels = rows(original_labels)
    wanted = original.parent + bank
    label_map = {r["row_id"]: r for r in labels}
    if len(label_map) != len(labels) or len(labels) != 7409 or proof.get("rows") != 7409:
        raise ValueError("Original actual-label proof lacks complete unique coverage")
    selected_labels = [label_map[r["row_id"]] for r in wanted]
    if any(p["supervised_tokens"] != r["supervised_tokens"] for p, r in zip(selected_labels, wanted)):
        raise ValueError("Imported actual-label counts differ from canonical rows")
    dependencies = {
        "version": VERSION, "original_identity": original.identity, "preview_identity": PREVIEW_ID,
        "runner_sha256": sha(__file__), "package_pins_sha256": sha(HERE / "PACKAGE_PINS.json"),
        "decision_sha256": sha(HERE / "protocol_decision.json"), "bank_sha256": BANK_SHA,
        "preview_report_sha256": sha(preview / "preview_report.json"),
        "score_import_sha256": sha(imported_path), "original_label_proof_sha256": digest(proof),
        "B_manifest_sha256": MANIFESTS["B"][0], "C0_manifest_sha256": MANIFESTS["C0"][0],
        "B_selection_sha256": sha(preview / "B/selection_preview.json"),
        "C0_selection_sha256": sha(preview / "C0/selection_preview.json"),
        "parent_sha256": original.entry["expected_adapter_sha256"],
    }
    c = SimpleNamespace(**{**vars(original), "a": a, "out": work / "outputs/C0_screened_pilot_v1",
                          "identity": digest(dependencies), "dependencies": dependencies,
                          "bank": bank, "legacy": legacy, "original": original, "preview": preview,
                          "selected": selected, "labels": selected_labels, "proof": proof,
                          "decision": decision, "work": work})
    return c


def frozen_files(c):
    result = {
        "candidate_bank.jsonl": c.preview / "candidate_bank.jsonl",
        "reference_measurements.jsonl": c.preview / "imported_reference_measurements.jsonl",
        "preview_report.json": c.preview / "preview_report.json",
        "protocol_decision.json": HERE / "protocol_decision.json",
        "screening/screening_proposal.json": HERE / "screening/screening_proposal.json",
        "screening/known_exclusions.json": HERE / "screening/known_exclusions.json",
        "screening/screening_flags.json": HERE / "screening/screening_flags.json",
    }
    for method in MANIFESTS:
        result[f"selection/{method}/train.jsonl"] = c.preview / method / "train.jsonl"
        result[f"selection/{method}/selection_preview.json"] = c.preview / method / "selection_preview.json"
    return result


def freeze_receipt(c):
    return {
        "version": VERSION, "identity": c.identity, "profile": PROFILE,
        "status": "frozen_under_adopted_independent_screen", "parent_seed": 17,
        "continuation_seed": 42, "bank_rows": 4955, "bank_sha256": BANK_SHA,
        "parent_sha256": c.entry["expected_adapter_sha256"], "dependencies": c.dependencies,
        "original_matcher_flags_resolved": False, "limitations": LIMITATIONS,
        "files": {name: sha(path) for name, path in frozen_files(c).items()},
        "inherited_label_proof": c.proof,
        "actual_label_counts_sha256": digest(c.labels),
        "tokenizer_execution": "Reuse original hash-bound proof for byte-identical retained rows; re-encode selected rows immediately before training.",
        "training_started_by_prepare": False,
    }


def prepare(c):
    for name, source in frozen_files(c).items():
        save_bytes(artifact(c, name), source.read_bytes())
    save_rows(artifact(c, "actual_labels.jsonl"), c.labels)
    save(artifact(c, "freeze.json"), freeze_receipt(c))
    location = {"identity": c.identity, "output_directory": str(artifact(c)),
                "status": "prepared_not_trained", "parent_seed": 17, "continuation_seed": 42}
    save(HERE / "RUN_LOCATION.json", location)
    print(json.dumps(location, indent=2), flush=True)


def require_prepared(c):
    path = artifact(c, "freeze.json")
    if read(path) != freeze_receipt(c):
        raise ValueError("Frozen protocol receipt differs; run prepare with exact current inputs")
    for name, expected in read(path)["files"].items():
        check_hash(artifact(c, name), expected)
    if rows(artifact(c, "actual_labels.jsonl")) != c.labels:
        raise ValueError("Frozen imported label proof changed")
    return sha(path)


def verify_proof(c):
    require_prepared(c)
    return c.proof


def load_selection(c, method, bonus=None):
    if method not in MANIFESTS:
        raise ValueError("This pilot supports B and C0 only")
    path = artifact(c, "selection", method, "train.jsonl")
    check_hash(path, MANIFESTS[method][0])
    selected = rows(path)
    validate_selection(selected, c.bank, c.rebinned)
    report = read(path.parent / "selection_preview.json")
    if (selected, report) != c.selected[method]:
        raise ValueError("Frozen selection differs")
    return selected, report


def training_source_for_method(source, method):
    if method not in MANIFESTS or source.count("'preliminary_document_aligned_B'") != 1:
        raise ValueError("Unexpected parent trainer label or method")
    return source.replace("'preliminary_document_aligned_B'",
                          repr("independent_conservative_screen_" + method + "_continuation"), 1)


def gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one allocated NVIDIA GPU is required; submit through Slurm")
    return torch


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


def train(c, method, attempt, smoke=False):
    seed = CONTINUATION_SEED
    verify_proof(c)
    selected, selection = load_selection(c, method)
    if seed != 42:
        raise ValueError("This pilot fixes continuation seed 42")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise ValueError("Use a simple new attempt name")
    protocol_hash = require_prepared(c)
    smoke_key = digest({"identity": c.identity, "selection": digest(selection), "method": method, "seed": seed})
    if not smoke:
        smoke_dir = artifact(c, "smokes", method, attempt)
        validate_completion(c, smoke_dir, method, smoke=True)
    torch = gpu()
    parent_record = read(c.runtime / c.entry["records_relative_path"] / "run_config.json")
    for name, version in parent_record["environment"]["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Training environment differs from original parent: {name}; keep the working B environment")
    p2 = c.common.load_person2()
    p2.select_model("0.5B")
    tok = p2.get_tokenizer()
    if digest(tok.backend_tokenizer.to_str()) != verify_proof(c)["tokenizer_sha256"]:
        raise ValueError("Training tokenizer differs from verification")
    core = p2.core
    dest = artifact(c, "smokes" if smoke else "runs", method, attempt if smoke else f"seed{seed}_" + attempt)
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
        "protocol_version": VERSION, "parent_training_seed": 17, "original_matcher_flags_resolved": False,
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
              "version": VERSION, "parent_seed": 17, "bank_sha256": BANK_SHA,
              "original_matcher_flags_resolved": False}
    save(dest / "completion.json", report)
    validate_completion(c, dest, method, smoke=smoke)
    print(json.dumps({**report, "run_directory": str(dest)}, indent=2))


def run_directory(c, method, attempt, smoke=False):
    return artifact(c, "smokes" if smoke else "runs", method,
                    attempt if smoke else f"seed42_{attempt}")


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


def validate_completion(c, directory, method, smoke=False):
    protocol_hash = require_prepared(c)
    selected, selection = load_selection(c, method)
    selected = selected[:32] if smoke else selected
    directory = Path(directory).resolve()
    if directory.parent != artifact(c, "smokes" if smoke else "runs", method).resolve():
        raise ValueError("Training completion is outside this frozen experiment")
    receipt = read(directory / "completion.json")
    expected = {
        "passed": True, "identity": c.identity, "method": method, "seed": 42,
        "smoke_only": smoke, "reload_probe_matched": True, "version": VERSION,
        "parent_seed": 17, "bank_sha256": BANK_SHA,
        "selection_sha256": digest(selection), "parent_sha256": c.entry["expected_adapter_sha256"],
        "screening_protocol_sha256": protocol_hash, "unique_examples": len(selected),
        "actual_tokens": sum(r["supervised_tokens"] for r in selected),
        "original_matcher_flags_resolved": False,
        "smoke_key": digest({"identity": c.identity, "selection": digest(selection), "method": method, "seed": 42}),
    }
    if any(receipt.get(k) != value for k, value in expected.items()) or not receipt.get("changed_adapter_tensors"):
        raise ValueError("Training completion has the wrong protocol, method, seed, selection or smoke status")
    check_hash(directory / "run_config.json", receipt["run_config_sha256"])
    check_hash(directory / "training_summary.json", receipt["summary_sha256"])
    config = read(directory / "run_config.json")
    summary = read(directory / "training_summary.json")
    wanted_config = {
        "model": c.entry["model"], "revision": c.entry["revision"], "seed": 42,
        "epochs": 1, "accumulation": 8, "learning_rate": 1e-4, "warmup_updates": 10,
        "microbatch_size": 1, "packing": False, "arm": method,
        "shared_universe_sha256": BANK_SHA, "parent_adapter_sha256": c.entry["expected_adapter_sha256"],
        "original_parent_adapter_sha256": c.entry["expected_adapter_sha256"],
        "bridge_identity": c.identity, "protocol_version": VERSION, "parent_training_seed": 17,
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
    if rows(inputs / "train.jsonl") != selected or (not smoke and config["manifest_sha256"] != MANIFESTS[method][0]):
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


def evaluation_target(c, run, attempt):
    if run == "parent":
        return (c.root / c.entry["adapter_relative_path"], c.entry["expected_adapter_sha256"],
                c.legacy.PARENT_SHA)
    if run not in MANIFESTS:
        raise ValueError("Evaluation run must be parent, B or C0")
    directory = run_directory(c, run, attempt)
    receipt = validate_completion(c, directory, run)
    return Path(receipt["checkpoint"]) / "adapter", receipt["adapter_sha256"], MANIFESTS[run][0]


def prediction_path(c, run, benchmark, attempt):
    return artifact(c, "development", attempt, run, benchmark + ".jsonl")


def validate_prediction_rows(predictions, questions, config, checkpoint_sha, manifest_sha):
    ids = [p["problem_id"] for p in predictions]
    question_map = {q["problem_id"]: q for q in questions}
    if len(questions) != 500 or len(question_map) != 500 or len(ids) != 500 or len(set(ids)) != 500 or set(ids) != set(question_map):
        raise ValueError("Evaluation must cover exactly the 500 frozen questions once each")
    for p in predictions:
        if (type(p.get("correct")) is not bool or p.get("checkpoint_sha256") != checkpoint_sha
                or p.get("train_manifest_sha256") != manifest_sha
                or p.get("eval_config_hash") != config["eval_config_hash"]
                or p.get("parser_version") != config["parser_version"]
                or p.get("benchmark") != question_map[p["problem_id"]]["benchmark"]):
            raise ValueError("Evaluation prediction has mixed checkpoint, question, parser or settings identity")
        if p.get("parse_status") == "verify_error":
            raise ValueError("Answer-checker execution errors cannot be counted as mathematical failures")
    return {p["problem_id"]: p["correct"] for p in predictions}


def validate_evaluation(c, run, benchmark, attempt):
    if benchmark not in BENCHMARKS:
        raise ValueError("Only the two frozen development benchmarks are permitted")
    _, checkpoint_sha, manifest_sha = evaluation_target(c, run, attempt)
    source = c.runtime / "vendor/evaluation/data" / (benchmark + ".jsonl")
    check_hash(source, BENCHMARKS[benchmark])
    path = prediction_path(c, run, benchmark, attempt)
    config = read(path.with_suffix(".config.json"))
    expected = {
        "model": c.entry["model"], "revision": c.entry["revision"], "precision": "fp32",
        "max_new_tokens": 1024, "batch_size": 16, "stop_at_question": False,
        "question_text": "p1_clean", "add_special_tokens": False, "decoding": "greedy",
        "eos_token_id": 151643, "adapter_mode": "merged", "parser_version": "p3-parser-v6",
        "prompt_sha256": hashlib.sha256(c.contract["prompt_template"].encode()).hexdigest(),
        "eval_file_sha256": BENCHMARKS[benchmark], "checkpoint_sha256": checkpoint_sha,
        "train_manifest_sha256": manifest_sha,
        "evaluate_py_sha256": sha(c.runtime / "vendor/evaluation/evaluate.py"),
        "answer_parser_sha256": sha(c.runtime / "vendor/evaluation/answer_parser.py"),
    }
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError("Executed evaluation configuration differs from the frozen development recipe")
    predictions = rows(path)
    outcomes = validate_prediction_rows(predictions, rows(source), config, checkpoint_sha, manifest_sha)
    return outcomes, config, predictions


def evaluate(c, run, benchmark, attempt):
    require_prepared(c)
    if benchmark not in BENCHMARKS:
        raise ValueError("This pilot does not execute reserved final tests")
    adapter, checkpoint_sha, manifest_sha = evaluation_target(c, run, attempt)
    source = c.runtime / "vendor/evaluation/data" / (benchmark + ".jsonl")
    check_hash(source, BENCHMARKS[benchmark])
    destination = prediction_path(c, run, benchmark, attempt)
    if destination.is_file():
        previous = rows(destination)
        config = read(destination.with_suffix(".config.json"))
        if (len(previous) != len({p["problem_id"] for p in previous})
                or config.get("checkpoint_sha256") != checkpoint_sha
                or config.get("eval_file_sha256") != BENCHMARKS[benchmark]):
            raise ValueError("Existing predictions have incompatible identities; preserve them")
        if len(previous) == 500:
            validate_evaluation(c, run, benchmark, attempt)
            print(f"Already complete and validated: {run} {benchmark}", flush=True)
            return
    gpu()
    command = [sys.executable, "-B", str(c.runtime / "vendor/evaluation/evaluate.py"),
               "--eval_file", str(source), "--adapter", str(adapter),
               "--expect_train_manifest_sha256", manifest_sha, "--out", str(destination),
               "--run_id", VERSION + "_" + attempt + "_" + run + "_" + benchmark,
               "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024",
               "--question_text", "p1_clean"]
    subprocess.run(command, check=True)
    outcomes, _, _ = validate_evaluation(c, run, benchmark, attempt)
    print(json.dumps({"run": run, "benchmark": benchmark, "correct": sum(outcomes.values()),
                      "total": len(outcomes), "predictions": str(destination)}, indent=2), flush=True)


def paired_difference(later, earlier, resamples=2000, seed=0):
    if not later or set(later) != set(earlier) or any(type(v) is not bool for v in list(later.values()) + list(earlier.values())):
        raise ValueError("Paired outcomes require identical nonempty IDs and boolean correctness")
    differences = [int(later[k]) - int(earlier[k]) for k in sorted(later)]
    rng = random.Random(seed)
    draws = sorted(100 * sum(rng.choice(differences) for _ in differences) / len(differences)
                   for _ in range(resamples))
    counts = Counter((earlier[k], later[k]) for k in earlier)
    return {
        "difference_pp": 100 * sum(differences) / len(differences),
        "ci95_pp": [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]],
        "questions": len(differences), "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "paired_outcomes": {"retained_success": counts[True, True], "improvement": counts[False, True],
                            "regression": counts[True, False], "persistent_failure": counts[False, False]},
        "uncertainty_scope": "paired development items; no training-seed uncertainty estimate",
    }


def summarize(c, attempt):
    protocol_hash = require_prepared(c)
    training = {method: validate_completion(c, run_directory(c, method, attempt), method)
                for method in MANIFESTS}
    results, evaluation_hashes = {}, {}
    compare_fields = (
        "model", "revision", "precision", "max_new_tokens", "batch_size", "stop_at_question",
        "question_text", "prompt_sha256", "add_special_tokens", "decoding", "eos_token_id",
        "adapter_mode", "gpu", "cuda", "torch", "transformers", "peft", "parser_version",
        "math_verify", "eval_file_sha256", "answer_parser_sha256", "evaluate_py_sha256", "cut_at_next_question",
    )
    for benchmark in BENCHMARKS:
        evaluated = {run: validate_evaluation(c, run, benchmark, attempt) for run in ("parent", "B", "C0")}
        parent_config = evaluated["parent"][1]
        if any(any(value[1].get(k) != parent_config.get(k) for k in compare_fields) for value in evaluated.values()):
            raise ValueError("Compared models used different evaluator, GPU, parser or generation settings")
        row = {"questions": 500}
        for run, (outcomes, config, predictions) in evaluated.items():
            row[run] = {"correct": sum(outcomes.values()), "accuracy_percent": 100 * sum(outcomes.values()) / 500,
                        "parse_status_counts": dict(Counter(p.get("parse_status", "missing") for p in predictions)),
                        "length_limited": sum(p.get("finish_reason") == "length" for p in predictions)}
            path = prediction_path(c, run, benchmark, attempt)
            evaluation_hashes[str(path)] = {"predictions_sha256": sha(path),
                                             "config_sha256": sha(path.with_suffix(".config.json"))}
        for later, earlier in (("B", "parent"), ("C0", "parent"), ("C0", "B")):
            row[f"{later}_minus_{earlier}"] = paired_difference(evaluated[later][0], evaluated[earlier][0])
        results[benchmark] = row
    report = {
        "version": VERSION, "identity": c.identity, "profile": PROFILE,
        "scope": "one screened B/C0 development pilot", "parent_seed": 17, "continuation_seed": 42,
        "screened_c0_pilot_complete": True, "all_project_experiments_complete": False,
        "reserved_final_evaluation_performed": False, "original_matcher_flags_resolved": False,
        "screening_protocol_sha256": protocol_hash, "bank_sha256": BANK_SHA,
        "training": training, "results": results, "evaluation_hashes": evaluation_hashes,
        "limitations": LIMITATIONS, "output_directory": str(artifact(c)),
    }
    destination = artifact(c, "reports", attempt)
    save(destination / "completion_report.json", report)
    lines = ["# Screened C0 pilot: 0.5B, parent seed 17", "",
             "Continuation seed: 42. Matched B/C0; one epoch each from the original parent.", "",
             "| Development benchmark | Parent % | B % | C0 % | C0 minus B (pp) | Paired 95% interval |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for benchmark, value in results.items():
        difference = value["C0_minus_B"]
        lines.append(f"| {benchmark} | {value['parent']['accuracy_percent']:.2f} | {value['B']['accuracy_percent']:.2f} | "
                     f"{value['C0']['accuracy_percent']:.2f} | {difference['difference_pp']:.2f} | {difference['ci95_pp']} |")
    lines += ["", "B: 1,000 examples / 317,486 supervised tokens. C0: 1,000 examples / 320,273 supervised tokens.",
              "", "Both satisfy the same 318,845-token target within 0.5%, and the original source/length constraints.",
              "", "## Scope and limitations", ""] + ["- " + item for item in LIMITATIONS]
    save_bytes(destination / "comparison.md", ("\n".join(lines) + "\n").encode())
    save_bytes(destination / "completion_stdout.json", (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode())
    save(artifact(c, "PILOT_COMPLETE_" + attempt + ".json"),
         {"screened_c0_pilot_complete": True, "identity": c.identity,
          "report": str(destination / "completion_report.json"),
          "report_sha256": sha(destination / "completion_report.json")})
    bundle = create_result_bundle(c, attempt)
    print(json.dumps({"screened_c0_pilot_complete": True,
                      "report": str(destination / "completion_report.json"),
                      "comparison": str(destination / "comparison.md"), "results_zip": str(bundle)}, indent=2), flush=True)


def create_result_bundle(c, attempt):
    import io
    import zipfile
    buffer = io.BytesIO()
    paths = [artifact(c, "freeze.json"), artifact(c, "protocol_decision.json"),
             artifact(c, "preview_report.json"), artifact(c, "PILOT_COMPLETE_" + attempt + ".json")]
    for method in MANIFESTS:
        directory = run_directory(c, method, attempt)
        for name in ("completion.json", "run_config.json", "training_summary.json", "training_log.jsonl", "seen_groups.csv"):
            paths.append(directory / name)
        paths.extend(artifact(c, "selection", method).glob("*.json*"))
    for run in ("parent", "B", "C0"):
        paths.extend(artifact(c, "development", attempt, run).glob("*.json*"))
    paths.extend(artifact(c, "reports", attempt).glob("*"))
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            info = zipfile.ZipInfo(str(Path(PROFILE) / path.relative_to(artifact(c))))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, path.read_bytes())
    destination = artifact(c, f"C0_05B_parent17_cont42_{attempt}_results.zip")
    save_bytes(destination, buffer.getvalue())
    return destination


def child_command(c, action, attempt, *arguments):
    return [sys.executable, "-B", str(HERE / "screened_c0.py"), action,
            "--work", str(c.work), "--b-root", str(c.root), "--attempt", attempt, *arguments]


def pipeline(c, attempt):
    require_prepared(c)
    stages = [("train", ["--method", method, "--smoke"], method, True)
              for method in MANIFESTS]
    stages += [("train", ["--method", method], method, False) for method in MANIFESTS]
    stages += [("evaluate", ["--run", run, "--benchmark", benchmark], run, False)
               for run in ("parent", "B", "C0") for benchmark in BENCHMARKS]
    stages += [("summarize", [], None, False)]
    for number, (action, arguments, method, smoke) in enumerate(stages, 1):
        print(f"\nStage {number}/{len(stages)}: {action} {' '.join(arguments)}", flush=True)
        if action == "train":
            directory = run_directory(c, method, attempt, smoke)
            if (directory / "completion.json").is_file():
                validate_completion(c, directory, method, smoke=smoke)
                print("Already completed and validated; preserving this stage", flush=True)
                continue
            if directory.exists():
                raise ValueError(f"Incomplete training attempt preserved: {directory}. "
                                 "Use --attempt pilot02 to restart both methods from the original parent; "
                                 "this runner never restores foreign or partial optimizer state.")
        subprocess.run(child_command(c, action, attempt, *arguments), check=True)
    print("SCREENED C0 PILOT COMPLETE: parent17, continuation42, B/C0, development evaluation.", flush=True)


def submit(c, attempt):
    require_prepared(c)
    complete = artifact(c, "PILOT_COMPLETE_" + attempt + ".json")
    if complete.is_file():
        print("This attempt already has a completion receipt. Run status to inspect it.", flush=True)
        return
    prior = sorted(artifact(c, "submissions", attempt).glob("*.json"))
    if prior:
        previous = read(prior[-1])
        query = subprocess.run(["squeue", "-h", "-j", previous["job_id"], "-o", "%T"],
                               check=True, capture_output=True, text=True)
        if query.stdout.strip():
            print(json.dumps({**previous, "scheduler_status": query.stdout.strip(),
                              "new_submission_created": False}, indent=2), flush=True)
            return
    logs = c.work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = ["sbatch", "--parsable", "--chdir", str(c.work),
               "--output", str(logs / "c0-screen-05b-p17.%j.out"),
               "--error", str(logs / "c0-screen-05b-p17.%j.err"),
               str(HERE / "gpu_pipeline.sbatch"), str(HERE), "--work", str(c.work),
               "--b-root", str(c.root), "--attempt", attempt]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise ValueError("Scheduler response was not a numeric job ID: " + result.stdout)
    receipt = {"job_id": job_id, "identity": c.identity, "attempt": attempt,
               "output_directory": str(artifact(c)),
               "stdout": str(logs / f"c0-screen-05b-p17.{job_id}.out"),
               "stderr": str(logs / f"c0-screen-05b-p17.{job_id}.err"),
               "status": "submitted_training_not_yet_verified", "command": command}
    save(artifact(c, "submissions", attempt, f"{int(job_id):020d}.json"), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def status(c, attempt):
    require_prepared(c)
    report = {"identity": c.identity, "output_directory": str(artifact(c)),
              "attempt": attempt, "parent_seed": 17, "continuation_seed": 42,
              "training": {}, "development": {}, "screened_c0_pilot_complete": False}
    for method in MANIFESTS:
        directory = run_directory(c, method, attempt)
        if (directory / "completion.json").is_file():
            receipt = validate_completion(c, directory, method)
            report["training"][method] = {"status": "complete_and_validated", "checkpoint": receipt["checkpoint"],
                                           "actual_tokens": receipt["actual_tokens"]}
        else:
            report["training"][method] = {"status": "in_progress_or_interrupted" if directory.exists() else "not_started"}
    for run in ("parent", "B", "C0"):
        report["development"][run] = {}
        for benchmark in BENCHMARKS:
            path = prediction_path(c, run, benchmark, attempt)
            count = len(rows(path)) if path.is_file() else 0
            if count == 500:
                validate_evaluation(c, run, benchmark, attempt)
            report["development"][run][benchmark] = {"saved_questions": count, "required_questions": 500,
                                                       "validated_complete": count == 500}
    marker = artifact(c, "PILOT_COMPLETE_" + attempt + ".json")
    if marker.is_file():
        saved = read(marker)
        if saved["identity"] != c.identity:
            raise ValueError("Completion marker belongs to another experiment")
        check_hash(saved["report"], saved["report_sha256"])
        if (any(value["status"] != "complete_and_validated" for value in report["training"].values())
                or any(not value["validated_complete"] for parts in report["development"].values() for value in parts.values())):
            raise ValueError("Completion marker exists but current training/evaluation evidence is incomplete")
        report["screened_c0_pilot_complete"] = True
        report["report"] = saved["report"]
        report["results_zip"] = str(artifact(c, f"C0_05B_parent17_cont42_{attempt}_results.zip"))
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "pipeline", "train", "evaluate", "summarize", "status"))
    parser.add_argument("--work", default="/scratch/sshind38/capstone_c0_c1_work")
    parser.add_argument("--b-root", default="/scratch/sshind38/capstone_approach_b")
    parser.add_argument("--attempt", default="pilot01")
    parser.add_argument("--method", choices=("B", "C0"), default="C0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", choices=("parent", "B", "C0"), default="C0")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="dev_gsm")
    args = parser.parse_args(argv)
    if not args.attempt or Path(args.attempt).name != args.attempt or args.attempt in (".", ".."):
        parser.error("Use a simple attempt name, such as pilot01")
    try:
        c = load_context(args)
        import fcntl
        lock_key = digest({"identity": c.identity, "action": args.action, "attempt": args.attempt,
                           "method": args.method, "smoke": args.smoke, "run": args.run,
                           "benchmark": args.benchmark})
        lock_path = c.out / ".locks" / (lock_key + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == "prepare":
                prepare(c)
            elif args.action == "submit":
                submit(c, args.attempt)
            elif args.action == "pipeline":
                pipeline(c, args.attempt)
            elif args.action == "train":
                train(c, args.method, args.attempt, smoke=args.smoke)
            elif args.action == "evaluate":
                evaluate(c, args.run, args.benchmark, args.attempt)
            elif args.action == "summarize":
                summarize(c, args.attempt)
            else:
                status(c, args.attempt)
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

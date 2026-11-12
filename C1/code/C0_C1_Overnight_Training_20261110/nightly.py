#!/usr/bin/env python3
"""Screened C0/C1 training, with durable Slurm continuation and no benchmark evaluation."""
import argparse
import contextlib
from collections import Counter
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

HERE = Path(__file__).resolve().parent
VERSION = "screened-c0-c1-training-only-v1"
BANK_SHA = "ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f"
TARGET = 318845
PROFILES = (("0.5B", 17), ("0.5B", 123), ("1.5B", 17), ("1.5B", 123))
SEEDS = (42, 137, 2026)
CHUNK_SIZE = 512
CHUNKS = 10
MAX_ATTEMPTS = 3
PILOT_ID = "be46c67690fb33ba680a5cd578bd418c6c06f4f2bcfadb39b329a10cb9fd5759"
PILOT_RUNNER_SHA = "fc83b1b2395c1d6292751c41544f5736534cc5ec7143d8d32cc7b8e0acebb0e6"
MANIFESTS = {"C0", "C1"}


class IntegrityError(ValueError):
    """Do not automatically change scientific inputs or reuse invalid evidence."""


class SelectionBlocked(IntegrityError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


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
        raise IntegrityError(f"Hash mismatch; preserved existing file: {path}")
    return expected


def save_bytes(path, content):
    """Create immutable artifacts; atomic hard-link prevents overwriting other attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise IntegrityError(f"Different existing evidence preserved: {path}")
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


def current_json(path, value):
    """Only controller status/index files are replaceable, never experimental artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def locked(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_check():
    pins = read(HERE / "PACKAGE_PINS.json")
    for relative, expected in pins.items():
        check_hash(HERE / relative, expected)
    return sha(HERE / "PACKAGE_PINS.json")


def control_root(a):
    return Path(a.work).resolve() / "outputs/C0_C1_overnight_training_v1" / digest({
        "package": package_check(), "protocol": sha(HERE / "protocol.json"),
        "work": str(Path(a.work).resolve()), "b_root": str(Path(a.b_root).resolve())})[:16]


def artifact(c, *parts):
    return c.out / c.profile / c.identity[:16] / Path(*parts)


def load_context(a):
    package_check()
    legacy = load_module("_overnight_immutable_bridge", HERE / "legacy/sol_bridge.py")
    old = legacy.load_context(SimpleNamespace(b_root=a.b_root, repo=str(Path(a.work) / "repo"),
         output=str(Path(a.work) / "outputs/C0_C1_B_bridge_v1"), model=a.model, parent_seed=a.parent_seed))
    preview = Path(a.work) / "outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17/35f8a5ecbf57cd37"
    check_hash(preview / "candidate_bank.jsonl", BANK_SHA)
    bank = rows(preview / "candidate_bank.jsonl")
    checks = load_module("_overnight_checks", HERE / "pilot_checks.py")
    screen = read(HERE / "screening/screening_proposal.json")
    decisions = read(HERE / "screening/known_exclusions.json")["decisions"]
    if digest([{k: r.get(k) for k in screen["projection_fields"]} for r in old.bank]) != screen["original_projection_sha256"]:
        raise IntegrityError("Original bank question projection changed")
    checks.validate_population(old.bank, bank, screen, decisions)
    dependencies = {"version": VERSION, "package_sha256": sha(HERE / "PACKAGE_PINS.json"),
        "protocol_sha256": sha(HERE / "protocol.json"), "original_identity": old.identity,
        "bank_sha256": BANK_SHA, "profile": old.profile,
        "parent_sha256": old.entry["expected_adapter_sha256"]}
    c = SimpleNamespace(**{**vars(old), "a": a, "out": control_root(a) / "profiles",
        "bank": bank, "bank_path": preview / "candidate_bank.jsonl", "original": old,
        "legacy": legacy, "checks": checks, "preview": preview, "work": Path(a.work).resolve(),
        "control": control_root(a), "identity": digest(dependencies), "dependencies": dependencies})
    return c


def freeze(c):
    return {"version": VERSION, "identity": c.identity, "dependencies": c.dependencies,
        "bank_rows": 4955, "bank_sha256": BANK_SHA, "parent_seed": c.a.parent_seed,
        "parent_sha256": c.entry["expected_adapter_sha256"], "model_size": c.a.model,
        "continuation_seeds": list(SEEDS), "methods": ["C0", "C1"],
        "target_tokens": TARGET, "global_tolerance": 0.005, "stratum_tolerance": 0.05,
        "benchmark_evaluation_performed": False, "original_matcher_flags_resolved": False,
        "inherited_parent_exposure_repaired": False,
        "candidate_generation_role": "Training selection only; question-only parent inference, not held-out evaluation",
        "C1_rule": "round(1000000 * fit) * (1 + confirmed_mistake); lambda=1",
        "C1_policy": "Frozen exploratory refinement; no improvement claim or benchmark tuning",
        "limitations": read(HERE / "protocol.json")["limitations"]}


def require_prepared(c):
    path = artifact(c, "freeze.json")
    if read(path) != freeze(c):
        raise IntegrityError("Frozen protocol/context changed")
    check_hash(artifact(c, "candidate_bank.jsonl"), BANK_SHA)
    return sha(path)


def verify_proof(c):
    require_prepared(c)
    proof = c.legacy.verify_proof(c)
    labels = rows(artifact(c, "verification/actual_labels.jsonl"))
    wanted = c.parent + c.bank
    if (proof.get("rows") != len(wanted) or len(labels) != len(wanted)
            or [r["row_id"] for r in labels] != [r["row_id"] for r in wanted]
            or any(l["supervised_tokens"] != r["supervised_tokens"] for l, r in zip(labels, wanted))):
        raise IntegrityError("Tokenizer proof does not cover this complete parent and screened bank")
    c.labels = labels
    return proof


def prepare(c):
    save_bytes(artifact(c, "candidate_bank.jsonl"), c.bank_path.read_bytes())
    save(artifact(c, "freeze.json"), freeze(c))
    if (artifact(c, "verification/verification.json")).is_file():
        verify_proof(c)
        return
    if c.profile == "0.5B_seed17":
        proof = c.legacy.verify_proof(c.original)
        originals = rows(c.legacy.artifact(c.original, "verification/actual_labels.jsonl"))
        by_id = {r["row_id"]: r for r in originals}
        labels = [by_id[r["row_id"]] for r in c.parent + c.bank]
        save_rows(artifact(c, "verification/actual_labels.jsonl"), labels)
        report = {"passed": True, "identity": c.identity, "rows": len(labels),
            "actual_parent_tokens": TARGET, "tokenizer_sha256": proof["tokenizer_sha256"],
            "labels_sha256": sha(artifact(c, "verification/actual_labels.jsonl")), "weights_loaded": False,
            "source_original_proof_sha256": digest(proof), "source_original_identity": c.original.identity}
        save(artifact(c, "verification/verification.json"), report)
    else:
        c.legacy.verify(c)
    verify_proof(c)


def load_scores(c):
    records, report = c.legacy.load_scores(c)
    if report.get("rows") != len(c.bank) or report.get("parent_sha256") != c.entry["expected_adapter_sha256"]:
        raise IntegrityError("Reference measurement belongs to another parent")
    by_id = {r["row_id"]: r for r in c.bank}
    for record in records:
        row = by_id[record["row_id"]]
        if (record.get("status") != "ok" or record.get("row_sha256") != digest(row)
                or record.get("parent_sha256") != c.entry["expected_adapter_sha256"]
                or record.get("measurement_key") != report["measurement_key"]
                or record.get("supervised_tokens") != row["supervised_tokens"]
                or not math.isfinite(record["mean_nll"]) or record["mean_nll"] < 0):
            raise IntegrityError("Reference measurement content, parent, tokens or runtime changed")
    return records, report


def score(c):
    verify_proof(c)
    if list(artifact(c, "reference").glob("*/completion.json")):
        return load_scores(c)
    if c.profile == "0.5B_seed17":
        previous, source = c.legacy.load_scores(c.original)
        allowed = {r["row_id"] for r in c.bank}
        records = [r for r in previous if r["row_id"] in allowed]
        dest = artifact(c, "reference/imported")
        save_rows(dest / "scores.jsonl", records)
        save(dest / "completion.json", {"complete": True, "identity": c.identity, "rows": len(records),
            "scores_sha256": sha(dest / "scores.jsonl"), "measurement_key": source["measurement_key"],
            "parent_sha256": c.entry["expected_adapter_sha256"], "imported_from_original_identity": c.original.identity,
            "source_scores_sha256": source["scores_sha256"], "new_GPU_scoring_performed": False})
    else:
        c.legacy.score(c)
    return load_scores(c)


def training_order(records):
    def key(row):
        h = hashlib.sha256(("42|training-order|" + row["row_id"]).encode()).digest()
        return int.from_bytes(h[:4], "big"), row["row_id"]
    return sorted(records, key=key)


def load_selection(c, method):
    if method not in MANIFESTS:
        raise IntegrityError("Only C0 and C1 training is authorized in this package")
    path = artifact(c, "selection", method, "train.jsonl")
    report = read(path.parent / "selection.json")
    check_hash(path, report["manifest_sha256"])
    selected = rows(path)
    tokens = c.checks.validate_selection(selected, c.bank, c.rebinned)
    _, measurements = load_scores(c)
    expected = {"identity": c.identity, "method": method, "rows": 1000, "actual_tokens": tokens,
                "target_tokens": TARGET, "scores_sha256": measurements["scores_sha256"],
                "bank_sha256": BANK_SHA, "parent_sha256": c.entry["expected_adapter_sha256"]}
    if any(report.get(k) != v for k, v in expected.items()):
        raise IntegrityError("Selection context changed")
    if method == "C1":
        outcomes, generation = load_generation(c)
        if report.get("generation_sha256") != digest(generation) or report.get("bonus") != "1":
            raise IntegrityError("C1 selection does not bind the complete candidate generations")
        if any(report["C1_diagnostics"]["degeneracy_flags"].values()):
            raise SelectionBlocked("C1 diagnostic gate failed")
    return selected, report


def solve_selection(c, method):
    verify_proof(c)
    dest = artifact(c, "selection", method)
    if (dest / "selection.json").is_file():
        return load_selection(c, method)
    from capstone_person4.b_select import solve
    from capstone_person4.selection import fit_priorities
    from capstone_person4_approachC1.generation_scoring import c1_utilities
    records, measurement = load_scores(c)
    ranked = fit_priorities(c.bank, records, 42)
    utilities = {r["row_id"]: int(round(r["fit"] * 1000000)) for r in ranked}
    generation = None
    diagnostics = None
    if method == "C0" and c.profile == "0.5B_seed17":
        source = c.preview / "C0/train.jsonl"
        check_hash(source, "71e4863eba97c84e8f084ebfc58f6fae4ecde3e0678e6b1e927bf887cf312478")
        selected = rows(source)
        solver = read(c.preview / "C0/selection_preview.json")["solver"]
        if solver.get("passed") is not True or solver.get("tolerance_relaxed") is not False:
            raise IntegrityError("Existing C0 selection has no conforming solver proof")
    else:
        if method == "C1":
            outcomes, generation = load_generation(c)
            mistakes = {r["row_id"]: r["mistake"] for r in outcomes}
            utilities = c1_utilities(ranked, mistakes, 1000000, Fraction(1))
        selected, solver = solve(c.rebinned, c.bank, 1, utilities=utilities)
        if selected is None:
            raise SelectionBlocked(f"No conforming {method} selection: {solver.get('status')}; no bounds relaxed")
        selected = training_order(selected)
    tokens = c.checks.validate_selection(selected, c.bank, c.rebinned)
    if method == "C1":
        c0, _ = load_selection(c, "C0")
        share = sum(o["mistake"] for o in outcomes) / len(outcomes)
        left, right = {r["row_id"] for r in c0}, {r["row_id"] for r in selected}
        overlap = len(left & right) / len(left | right)
        diagnostics = {"mistake_share": share, "C0_C1_jaccard": overlap,
            "degeneracy_flags": {"mistakes_rare": share < 0.05,
                "mistakes_near_universal": share > 0.95, "C1_almost_same_as_C0": overlap >= 0.95}}
        save(dest / "diagnostics.json", diagnostics)
        if any(diagnostics["degeneracy_flags"].values()):
            raise SelectionBlocked("C1 diagnostic gate failed; evidence preserved: " + json.dumps(diagnostics))
    save_rows(dest / "train.jsonl", selected)
    save(dest / "selection.json", {"identity": c.identity, "method": method, "bonus": "1" if method == "C1" else None,
        "rows": 1000, "actual_tokens": tokens, "target_tokens": TARGET, "manifest_sha256": sha(dest / "train.jsonl"),
        "scores_sha256": measurement["scores_sha256"], "bank_sha256": BANK_SHA,
        "parent_sha256": c.entry["expected_adapter_sha256"], "solver": solver,
        "generation_sha256": digest(generation) if generation else None, "C1_diagnostics": diagnostics,
        "utility_sha256": digest(utilities), "training_started_by_selection": False})
    return load_selection(c, method)


def candidate_rows(c, chunk):
    if not 0 <= chunk < CHUNKS:
        raise IntegrityError("Unknown candidate chunk")
    return c.bank[chunk * CHUNK_SIZE:(chunk + 1) * CHUNK_SIZE]


def candidate_root(c, chunk):
    return artifact(c, "generation", "chunks", f"{chunk:02d}")


def candidate_request(c, chunk):
    data = candidate_rows(c, chunk)
    request = [{"problem_id": r["row_id"], "benchmark": "math", "question": r["problem"],
                "gold_answer": r["final_answer"], "source": r["dataset_id"], "split": "training_candidate"}
               for r in data]
    path = candidate_root(c, chunk) / "questions.jsonl"
    save_rows(path, request)
    return path, request


def generation_contract(c, config, question_sha):
    expected = {"eval_file_sha256": question_sha, "checkpoint_sha256": c.entry["expected_adapter_sha256"],
        "model": c.entry["model"], "revision": c.entry["revision"], "precision": "fp32", "decoding": "greedy",
        "batch_size": 16, "max_new_tokens": 1024, "question_text": "p1_clean", "parser_version": "p3-parser-v6",
        "eos_token_id": 151643, "add_special_tokens": False, "adapter_mode": "merged", "stop_at_question": False,
        "train_manifest_sha256": c.legacy.PARENT_SHA}
    if any(config.get(k) != v for k, v in expected.items()):
        raise IntegrityError("Candidate generation uses a different parent or generation/grading contract")
    check_hash(c.runtime / "vendor/evaluation/evaluate.py", config["evaluate_py_sha256"])
    check_hash(c.runtime / "vendor/evaluation/answer_parser.py", config["answer_parser_sha256"])
    runtime_keys = ("model", "revision", "precision", "decoding", "batch_size", "max_new_tokens", "question_text",
        "parser_version", "eos_token_id", "add_special_tokens", "adapter_mode", "stop_at_question", "prompt_sha256",
        "transformers", "peft", "torch", "cuda", "gpu", "evaluate_py_sha256", "answer_parser_sha256", "math_verify")
    runtime = {k: config[k] for k in runtime_keys}
    path = artifact(c, "generation/runtime.json")
    if path.is_file():
        if read(path) != runtime:
            raise IntegrityError("Candidate generation platform/settings changed across chunks; no mixing allowed")
    else:
        save(path, runtime)
    return runtime


def classify_predictions(c, predictions, wanted, config, complete=True):
    from capstone_person4_approachC1.generation_scoring import classify
    expected = {r["row_id"] for r in wanted}
    ids = [r["problem_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) - expected or (complete and set(ids) != expected):
        raise IntegrityError("Candidate generations have duplicate/unknown/missing IDs")
    outcomes = []
    for p in predictions:
        if (p.get("checkpoint_sha256") != c.entry["expected_adapter_sha256"]
                or p.get("model_revision") != c.entry["model"] + "@" + c.entry["revision"]
                or p.get("parser_version") != "p3-parser-v6"
                or p.get("eval_config_hash") != config["eval_config_hash"]
                or p.get("benchmark") != "math" or p.get("train_manifest_sha256") != c.legacy.PARENT_SHA):
            raise IntegrityError("Candidate prediction has incompatible provenance")
        status, correct, flags = classify(p)
        if status == "execution_error":
            raise RuntimeError("Candidate checker execution error; retry generation before selection")
        outcomes.append({"row_id": p["problem_id"], "mistake": status == "incorrect",
                         "correct": correct, "status": status, "flags": flags})
    by_id = {r["row_id"]: r for r in outcomes}
    return [by_id[r["row_id"]] for r in wanted if r["row_id"] in by_id]


def load_chunk(c, chunk):
    root = candidate_root(c, chunk)
    report = read(root / "completion.json")
    if report.get("identity") != c.identity or report.get("chunk") != chunk or report.get("complete") is not True:
        raise IntegrityError("Candidate chunk completion has a different identity")
    path = root / report["predictions_relative_path"]
    check_hash(path, report["predictions_sha256"])
    config_path = path.with_suffix(".config.json")
    check_hash(config_path, report["config_sha256"])
    config = read(config_path)
    question_path, _ = candidate_request(c, chunk)
    generation_contract(c, config, sha(question_path))
    predictions = rows(path)
    outcomes = classify_predictions(c, predictions, candidate_rows(c, chunk), config)
    if digest(outcomes) != report["outcomes_sha256"]:
        raise IntegrityError("Candidate chunk outcomes changed")
    return outcomes, report


def generate_chunk(c, chunk):
    verify_proof(c)
    gpu()
    root = candidate_root(c, chunk)
    if (root / "completion.json").is_file():
        return load_chunk(c, chunk)
    question_path, _ = candidate_request(c, chunk)
    wanted = candidate_rows(c, chunk)
    output = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        path = root / f"attempt{attempt:02d}/predictions.jsonl"
        if not path.exists():
            if path.parent.exists():
                continue  # Preserve an interrupted model-load/config write without predictions.
            output = path
            break
        config_path = path.with_suffix(".config.json")
        try:
            config = read(config_path)
            previous = rows(path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # Preserve malformed partial attempts; regenerate this bounded chunk.
        generation_contract(c, config, sha(question_path))
        try:
            classify_predictions(c, previous, wanted, config, complete=False)
        except RuntimeError:
            continue
        output = path  # Native evaluator resumes compatible, unique complete records.
        break
    if output is None:
        raise SelectionBlocked("Candidate chunk exhausted three preserved attempts")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(c.runtime / "vendor/evaluation/evaluate.py"),
        "--eval_file", str(question_path), "--adapter", str(c.root / c.entry["adapter_relative_path"]),
        "--expect_train_manifest_sha256", c.legacy.PARENT_SHA, "--run_id", f"C1_training_candidates_{c.profile}_{chunk:02d}",
        "--out", str(output), "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024",
        "--question_text", "p1_clean"]
    subprocess.run(command, check=True)
    config = read(output.with_suffix(".config.json"))
    generation_contract(c, config, sha(question_path))
    outcomes = classify_predictions(c, rows(output), wanted, config)
    check_hash(c.root / c.entry["adapter_relative_path"] / "adapter_model.safetensors", c.entry["expected_adapter_sha256"])
    save(root / "completion.json", {"identity": c.identity, "chunk": chunk, "complete": True, "rows": len(outcomes),
        "questions_sha256": sha(question_path), "predictions_relative_path": str(output.relative_to(root)),
        "predictions_sha256": sha(output), "config_sha256": sha(output.with_suffix(".config.json")),
        "outcomes_sha256": digest(outcomes), "benchmark_evaluation_performed": False})
    return load_chunk(c, chunk)


def load_generation(c):
    outcomes, chunks = [], []
    for chunk in range(CHUNKS):
        classified, receipt = load_chunk(c, chunk)
        outcomes.extend(classified)
        chunks.append(receipt)
    if [r["row_id"] for r in outcomes] != [r["row_id"] for r in c.bank]:
        raise IntegrityError("Combined C1 outcomes do not cover this exact candidate bank")
    report = {"identity": c.identity, "complete": True, "rows": len(outcomes), "bank_sha256": BANK_SHA,
        "parent_sha256": c.entry["expected_adapter_sha256"], "chunks_sha256": digest(chunks),
        "runtime_sha256": sha(artifact(c, "generation/runtime.json")), "outcomes_sha256": digest(outcomes),
        "status_counts": dict(Counter(r["status"] for r in outcomes)), "benchmark_evaluation_performed": False}
    return outcomes, report


def finalize_generation(c):
    outcomes, report = load_generation(c)
    save_rows(artifact(c, "generation/outcomes.jsonl"), outcomes)
    save(artifact(c, "generation/completion.json"), report)
    return report


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


# The validated pilot training/checking functions are inserted here by the package builder.
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


def original_pilot(c):
    """Validate original evidence and checkpoint in place; do not relabel/retrain the pilot."""
    package = c.work / "C0_Screened_05B_Seed17_20261110"
    check_hash(package / "screened_c0.py", PILOT_RUNNER_SHA)
    pilot = load_module("_completed_screened_c0_pilot", package / "screened_c0.py")
    context = pilot.load_context(SimpleNamespace(work=str(c.work), b_root=str(c.root)))
    if context.identity != PILOT_ID:
        raise IntegrityError("Completed pilot identity differs")
    result = pilot.validate_completion(context,
        pilot.run_directory(context, "C0", "pilot01"), "C0")
    return result, str(pilot.run_directory(context, "C0", "pilot01"))


def find_training(c, method, seed):
    if method == "C0" and c.profile == "0.5B_seed17" and seed == 42:
        return original_pilot(c)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        directory = run_directory(c, method, seed, f"attempt{attempt:02d}")
        if (directory / "completion.json").exists():
            return validate_completion(c, directory, method, seed), str(directory)
    return None


def run_training(c, method, seed):
    completed = find_training(c, method, seed)
    if completed:
        return completed
    for attempt in range(1, MAX_ATTEMPTS + 1):
        name = f"attempt{attempt:02d}"
        directory = run_directory(c, method, seed, name)
        smoke_directory = run_directory(c, method, seed, name, smoke=True)
        if directory.exists():
            continue  # A partial optimizer state is never guessed or reused.
        if smoke_directory.exists() and not (smoke_directory / "completion.json").exists():
            continue
        if not (smoke_directory / "completion.json").exists():
            subprocess.run(child(c.a, "train-one", method=method, seed=seed, attempt=name, smoke=True), check=True)
        else:
            validate_completion(c, smoke_directory, method, seed, smoke=True)
        subprocess.run(child(c.a, "train-one", method=method, seed=seed, attempt=name), check=True)
        return find_training(c, method, seed)
    raise SelectionBlocked("Three partial training attempts preserved; further automatic attempts stopped")


def environment_check(c):
    expected = read(c.runtime / c.entry["records_relative_path"] / "run_config.json")["environment"]["packages"]
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise IntegrityError(f"Parent training environment differs: {name}; retain the existing B environment")


def preflight_profile(a):
    c = load_context(a)
    environment_check(c)
    if c.profile == "0.5B_seed17":
        original_pilot(c)
        c.legacy.verify_proof(c.original)
        c.legacy.load_scores(c.original)
    # Cache check uses metadata only, with network access disabled by load_context.
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(c.entry["model"], revision=c.entry["revision"], local_files_only=True))
    if not (snapshot / "config.json").is_file() or not (snapshot / "tokenizer.json").is_file():
        raise IntegrityError("Pinned base-model/tokenizer cache is incomplete")
    weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("pytorch_model*.bin"))
    if not weights or any(not p.is_file() for p in weights):
        raise IntegrityError("Pinned base-model weights missing from the existing offline cache")
    return {"profile": c.profile, "identity": c.identity, "parent_ready": True, "offline_cache": str(snapshot),
            "output_directory": str(artifact(c)), "GPU_training_performed": False}


def child(a, action, **parameters):
    command = [sys.executable, "-B", str(HERE / "nightly.py"), action, "--work", str(a.work), "--b-root", str(a.b_root),
               "--model", a.model, "--parent-seed", str(a.parent_seed)]
    for key, value in parameters.items():
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    return command


def units():
    plan = []
    def add(profile, stage, requires=(), **kwargs):
        key = profile + "__" + stage + "".join("__" + str(kwargs[k]) for k in sorted(kwargs))
        plan.append({"id": key, "profile": profile, "stage": stage, "requires": list(requires), **kwargs})
        return key
    roots = {}
    for size, parent_seed in PROFILES:
        profile = f"{size}_seed{parent_seed}"
        prep = add(profile, "prepare")
        scored = add(profile, "score", [prep])
        selected = add(profile, "select", [scored], method="C0")
        roots[profile] = selected
        for seed in SEEDS:
            if (size, parent_seed, seed) != ("0.5B", 17, 42):
                add(profile, "train", [selected], method="C0", seed=seed)
    for size, parent_seed in PROFILES:
        profile = f"{size}_seed{parent_seed}"
        chunks = [add(profile, "generate", [roots[profile]], chunk=chunk) for chunk in range(CHUNKS)]
        selected = add(profile, "select", chunks, method="C1")
        for seed in SEEDS:
            add(profile, "train", [selected], method="C1", seed=seed)
    return plan


def unit_path(a, unit):
    return control_root(a) / "unit_receipts" / (unit["id"] + ".json")


def file_evidence(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def execute_unit(a, unit):
    c = load_context(a)
    stage = unit["stage"]
    if stage == "prepare":
        prepare(c)
        evidence = [artifact(c, "freeze.json"), artifact(c, "candidate_bank.jsonl"),
                    artifact(c, "verification/verification.json"), artifact(c, "verification/actual_labels.jsonl")]
    elif stage == "score":
        _, report = score(c)
        path = next(artifact(c, "reference").glob("*/completion.json"))
        evidence = [path, path.parent / "scores.jsonl"]
    elif stage == "select":
        if unit["method"] == "C1":
            finalize_generation(c)
        solver = Path(a.b_root) / "venv_b_solver_v3/bin/python"
        command = child(a, "solve", method=unit["method"])
        command[0] = str(solver)
        subprocess.run(command, check=True)
        load_selection(c, unit["method"])
        evidence = [artifact(c, "selection", unit["method"], name) for name in ("train.jsonl", "selection.json")]
    elif stage == "generate":
        _, receipt = generate_chunk(c, unit["chunk"])
        path = candidate_root(c, unit["chunk"])
        prediction = path / receipt["predictions_relative_path"]
        evidence = [path / "completion.json", path / "questions.jsonl", prediction,
                    prediction.with_suffix(".config.json"), artifact(c, "generation/runtime.json")]
    elif stage == "train":
        receipt, directory = run_training(c, unit["method"], unit["seed"])
        directory = Path(directory)
        evidence = [directory / name for name in ("completion.json", "training_summary.json", "run_config.json", "training_log.jsonl")]
        evidence.append(Path(receipt["checkpoint"]) / "adapter/adapter_model.safetensors")
    else:
        raise IntegrityError("Unknown work unit")
    result = {"unit": unit, "identity": c.identity, "version": VERSION, "complete": True,
              "files": file_evidence(evidence), "benchmark_evaluation_performed": False}
    save(unit_path(a, unit), result)
    print(json.dumps({"unit_complete": unit["id"], "output_directory": str(artifact(c))}), flush=True)


def validate_unit_receipts(a):
    plan = units()
    identities = {}
    for size, parent_seed in PROFILES:
        profile_args = SimpleNamespace(**{**vars(a), "model": size, "parent_seed": parent_seed})
        c = load_context(profile_args)
        identities[c.profile] = c.identity
    done = set()
    for unit in plan:
        path = unit_path(a, unit)
        if not path.exists():
            continue
        receipt = read(path)
        if (receipt.get("unit") != unit or receipt.get("identity") != identities[unit["profile"]]
                or receipt.get("complete") is not True or receipt.get("version") != VERSION):
            raise IntegrityError("Work receipt belongs to different code/data/parent: " + str(path))
        for file, expected in receipt["files"].items():
            check_hash(file, expected)
        if any(dependency not in done for dependency in unit["requires"]):
            raise IntegrityError("Completed unit lacks validated dependency: " + unit["id"])
        done.add(unit["id"])
    return done


def report(a, state, status=None):
    plan = units()
    done = {u["id"] for u in plan if unit_path(a, u).is_file()}
    done_train = [u for u in plan if u["stage"] == "train" and u["id"] in done]
    runs = []
    # Only unit receipts count as completed new runs. The reused pilot has its own source receipt.
    for size, parent_seed in PROFILES:
        profile = f"{size}_seed{parent_seed}"
        for method in ("C0", "C1"):
            for seed in SEEDS:
                reused = (profile, method, seed) == ("0.5B_seed17", "C0", 42)
                unit = next((u for u in plan if u["profile"] == profile and u["stage"] == "train"
                             and u["method"] == method and u["seed"] == seed), None)
                completed = reused or (unit is not None and unit["id"] in done)
                path = None
                if reused:
                    path = str(Path(a.work) / f"outputs/C0_screened_pilot_v1/0.5B_seed17/{PILOT_ID[:16]}/runs/C0/seed42_pilot01")
                elif completed:
                    receipt = read(unit_path(a, unit))
                    path = str(Path(next(p for p in receipt["files"] if p.endswith("/completion.json"))).parent)
                runs.append({"model_size": size, "parent_seed": parent_seed, "continuation_seed": seed,
                             "method": method, "complete": completed, "reused_original_pilot": reused,
                             "run_directory": path, "adapter": str(Path(path) / "final/adapter") if path else None})
    all_complete = len(done_train) == 23 and len(done) == len(plan)
    counts = {method: sum(r["complete"] for r in runs if r["method"] == method) for method in ("C0", "C1")}
    value = {"version": VERSION, "updated_utc": utc(), "status": status or ("training_complete" if all_complete else "in_progress"),
        "all_requested_training_complete": all_complete, "C0_complete_of_12": counts["C0"], "C1_complete_of_12": counts["C1"],
        "new_training_runs_complete_of_23": len(done_train), "completed_units": len(done), "total_units": len(plan),
        "benchmark_evaluation_performed": False, "B_training_performed": False, "runs": runs,
        "unit_state": state, "control_directory": str(control_root(a)), "screened_bank_sha256": BANK_SHA,
        "original_matcher_flags_resolved": False,
        "results_scope": "Training artifacts only. C1 candidate generations are preparation, not benchmark evaluation.",
        "comparison_limitation": "Matching screened B controls beyond the existing pilot and all benchmark comparisons remain separate."}
    current_json(control_root(a) / "STATUS.json", value)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(runs[0]))
    writer.writeheader()
    writer.writerows(runs)
    path = control_root(a) / "TRAINING_RUNS.csv"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    temporary.write_text(output.getvalue())
    os.replace(temporary, path)
    return value


def build_bundle(a, final):
    root = control_root(a)
    target = root / "C0_C1_training_receipts.zip"
    temporary = root / ("bundle_" + uuid.uuid4().hex + ".zip.partial")
    wanted = [root / "STATUS.json", root / "TRAINING_RUNS.csv", HERE / "protocol.json", HERE / "PACKAGE_PINS.json"]
    wanted += list((root / "unit_receipts").glob("*.json")) + list((root / "submissions").glob("*.json"))
    for pattern in ("freeze.json", "selection/*/selection.json", "verification/verification.json",
                    "generation/completion.json", "generation/runtime.json", "selection/C1/diagnostics.json",
                    "runs/*/*/completion.json", "runs/*/*/run_config.json", "runs/*/*/training_summary.json"):
        wanted += list((root / "profiles").glob("*/*/" + pattern))
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(wanted)):
            if path.is_file():
                name = str(path.relative_to(root)) if path.is_relative_to(root) else "package/" + path.name
                archive.write(path, name)
    os.replace(temporary, target)
    if final["all_requested_training_complete"]:
        save(root / "TRAINING_COMPLETE.json", {"version": VERSION, "new_runs": 23, "C0_total": 12, "C1_total": 12,
            "status_report_sha256": sha(root / "STATUS.json"), "runs_csv_sha256": sha(root / "TRAINING_RUNS.csv"),
            "results_zip_sha256": sha(target), "benchmark_evaluation_performed": False})


def terminate_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def cancel_future(a, submission, slot):
    """Only this submission's later queued allocations are cancelled after terminal status."""
    ids = [j["job_id"] for j in submission["jobs"] if j["slot"] > slot]
    if ids:
        result = subprocess.run(["scancel", *ids], text=True, capture_output=True)
        current_json(control_root(a) / "cancellation.json", {"job_ids": ids, "returncode": result.returncode,
                                                               "reason": "No runnable work remains in this exact submission"})


def validate_final_marker(a):
    root = control_root(a)
    marker = read(root / "TRAINING_COMPLETE.json")
    for name, key in (("STATUS.json", "status_report_sha256"), ("TRAINING_RUNS.csv", "runs_csv_sha256"),
                      ("C0_C1_training_receipts.zip", "results_zip_sha256")):
        check_hash(root / name, marker[key])
    result = read(root / "STATUS.json")
    if (marker.get("new_runs") != 23 or result.get("all_requested_training_complete") is not True
            or result.get("new_training_runs_complete_of_23") != 23):
        raise IntegrityError("Final marker does not establish the requested 23 new training runs")
    return result


def worker(a):
    root = control_root(a)
    submission = read(Path(a.submission))
    if submission["package_sha256"] != package_check() or submission["control_directory"] != str(root):
        raise IntegrityError("Slurm submission belongs to another package/output directory")
    current_job = os.environ.get("SLURM_JOB_ID")
    if not any(j["slot"] == a.slot and j["job_id"] == current_job for j in submission["jobs"]):
        raise IntegrityError("Worker must be launched by its recorded Slurm job")
    stop = {"requested": False, "process": None}
    def interrupted(signum, frame):
        stop["requested"] = True
        if stop["process"] is not None:
            terminate_process(stop["process"])
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with locked(root / "WORKER.lock"):
        done = validate_unit_receipts(a)
        for size, parent_seed in PROFILES:
            c = load_context(SimpleNamespace(**{**vars(a), "model": size, "parent_seed": parent_seed}))
            if c.profile == "0.5B_seed17":
                original_pilot(c)
        if (root / "TRAINING_COMPLETE.json").is_file():
            validate_final_marker(a)
            cancel_future(a, submission, a.slot)
            print("All requested training was already completed and verified", flush=True)
            return 0
        state_path = root / "UNIT_STATE.json"
        state = read(state_path) if state_path.exists() else {}
        deadline = time.monotonic() + a.worker_seconds
        budget_min = {"prepare": 300, "score": 1200, "select": 900, "train": 1200, "generate": 2400}
        terminal_failure = False
        for unit in units():
            key = unit["id"]
            if key in done:
                state[key] = {"status": "complete"}
                continue
            if stop["requested"]:
                break
            prior = state.get(key, {})
            if prior.get("status") in ("blocked", "failed_permanently"):
                terminal_failure = True
                continue
            if any(state.get(d, {}).get("status") in ("blocked", "failed_permanently") for d in unit["requires"]):
                state[key] = {"status": "blocked", "reason": "Required preparation/selection did not pass"}
                terminal_failure = True
                continue
            if any(d not in done for d in unit["requires"]):
                continue
            if deadline - time.monotonic() < budget_min[unit["stage"]]:
                print("Passing unfinished work to the next pre-submitted allocation", flush=True)
                break
            failed = int(prior.get("failures", 0))
            params = {k: unit[k] for k in ("method", "seed", "chunk") if k in unit}
            size, seed_text = unit["profile"].split("_seed")
            profile_args = SimpleNamespace(**{**vars(a), "model": size, "parent_seed": int(seed_text)})
            command = child(profile_args, "unit", unit_id=key)
            log_dir = root / "stage_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = log_dir / (key + "_" + utc().replace(":", "-") + ".log")
            state[key] = {"status": "running", "job_id": current_job, "log": str(log), "failures": failed}
            current_json(state_path, state)
            report(a, state)
            print(f"START {key}; log: {log}", flush=True)
            with log.open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                stop["process"] = process
                try:
                    code = process.wait(timeout=max(1, deadline - time.monotonic() - 120))
                except subprocess.TimeoutExpired:
                    terminate_process(process)
                    code = -signal.SIGTERM
                finally:
                    stop["process"] = None
            if code == 0:
                if not unit_path(a, unit).is_file():
                    raise IntegrityError("Successful stage did not produce its completion receipt")
                done.add(key)
                state[key] = {"status": "complete", "log": str(log), "job_id": current_job}
            elif code in (65, 66):
                state[key] = {"status": "blocked", "returncode": code, "log": str(log),
                              "reason": "Integrity or C1 selection gate failed; no automatic relaxation"}
                terminal_failure = True
            elif code < 0 or stop["requested"]:
                state[key] = {"status": "interrupted", "log": str(log), "failures": failed}
                current_json(state_path, state)
                break
            else:
                failed += 1
                state[key] = {"status": "retry_pending" if failed < MAX_ATTEMPTS else "failed_permanently",
                              "log": str(log), "failures": failed, "returncode": code}
                if failed >= MAX_ATTEMPTS:
                    terminal_failure = True
            current_json(state_path, state)
            report(a, state)
            print(f"END {key}: {state[key]['status']}", flush=True)
        current_json(state_path, state)
        # Fresh hashes at the allocation boundary prevent a corrupt receipt being counted complete.
        validate_unit_receipts(a)
        original_pilot(load_context(SimpleNamespace(**{**vars(a), "model": "0.5B", "parent_seed": 17})))
        final = report(a, state)
        remaining = [u for u in units() if u["id"] not in done and state.get(u["id"], {}).get("status")
                     not in ("blocked", "failed_permanently")]
        if final["all_requested_training_complete"]:
            final = report(a, state, "training_complete")
            build_bundle(a, final)
            cancel_future(a, submission, a.slot)
        elif not remaining:
            final = report(a, state, "incomplete_blocked")
            build_bundle(a, final)
            cancel_future(a, submission, a.slot)
        elif a.slot == max(j["slot"] for j in submission["jobs"]):
            final = report(a, state, "incomplete_allocation_chain_exhausted")
            build_bundle(a, final)
        else:
            final = report(a, state, "continuing_in_next_allocation")
        print(json.dumps({k: final[k] for k in ("status", "C0_complete_of_12", "C1_complete_of_12",
                         "new_training_runs_complete_of_23", "control_directory")}, indent=2), flush=True)
        return 0 if final["all_requested_training_complete"] or remaining else 2


def preflight(a):
    if not (1 <= a.allocations <= 12):
        raise IntegrityError("Use 1 to 12 bounded allocations")
    if not 900 <= a.worker_seconds <= 13800:
        raise IntegrityError("Worker budget must leave at least ten minutes of a four-hour allocation")
    for command in ("sbatch", "scontrol", "squeue", "scancel"):
        if shutil.which(command) is None:
            raise IntegrityError("Run on Sol, where Slurm command is available: " + command)
    python = Path(a.b_root) / "venv_b_training_py312/bin/python"
    solver = Path(a.b_root) / "venv_b_solver_v3/bin/python"
    if Path(sys.executable).absolute() != python.absolute():
        # venv executable paths may be symlinks; don't compare resolved base-Python paths.
        if str(Path(sys.prefix)) != str(python.parent.parent):
            raise IntegrityError("Use the existing B training Python: " + str(python))
    if not solver.is_file():
        raise IntegrityError("Existing B solver environment is missing: " + str(solver))
    subprocess.run([str(solver), "-c", "import ortools; assert ortools.__version__ == '9.14.6206'; from ortools.sat.python import cp_model"], check=True)
    profiles = []
    for size, parent_seed in PROFILES:
        profile_a = SimpleNamespace(**{**vars(a), "model": size, "parent_seed": parent_seed})
        result = subprocess.run(child(profile_a, "preflight-profile"), capture_output=True, text=True)
        if result.returncode:
            raise IntegrityError(f"Preflight failed for {size}, parent {parent_seed}:\n{result.stdout}\n{result.stderr}")
        profiles.append(json.loads(result.stdout.strip().splitlines()[-1]))
    root = control_root(a)
    root.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "profiles": profiles, "remaining_C0_training": 11, "C1_training": 12,
              "benchmark_evaluation_requested": False, "planned_new_training_runs": 23,
              "candidate_generation_chunks": 40, "control_directory": str(root),
              "package_sha256": package_check()}
    save(root / "PREFLIGHT.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def submit(a):
    preflight(a)
    root = control_root(a)
    with locked(root / "SUBMIT.lock"):
        for path in sorted((root / "submissions").glob("*.json"), reverse=True):
            previous = read(path)
            ids = [j["job_id"] for j in previous["jobs"]]
            if ids:
                check = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T %r"], text=True, capture_output=True)
                if check.returncode:
                    raise IntegrityError("Unable to establish prior Slurm status; no duplicate submission")
                if check.stdout.strip():
                    released = previous.setdefault("released_job_ids", [])
                    queue = {line.split()[0]: line.split(maxsplit=2)[1:]
                             for line in check.stdout.splitlines() if len(line.split()) >= 2}
                    for job in previous["jobs"]:
                        if job["job_id"] in released:
                            continue
                        details = queue.get(job["job_id"], [])
                        if len(details) == 2 and details[1] == "JobHeldUser":
                            subprocess.run(["scontrol", "release", job["job_id"]], check=True)
                        # Running, completed and previously released dependencies need no release.
                        released.append(job["job_id"])
                        current_json(path, previous)
                    previous["submission_complete"] = True
                    current_json(path, previous)
                    current_json(HERE / "RUN_LOCATION.json", {"control_directory": str(root),
                        "submission_file": str(path), "jobs": previous["jobs"]})
                    print(json.dumps({"reused_active_submission": True, "submission_file": str(path),
                                      "jobs": previous["jobs"], "queue": check.stdout}, indent=2))
                    return
        validate_unit_receipts(a)
        if (root / "TRAINING_COMPLETE.json").is_file():
            validate_final_marker(a)
            print("All requested training is already complete: " + str(root / "TRAINING_COMPLETE.json"))
            return
        index = len(list((root / "submissions").glob("*.json"))) + 1
        path = root / "submissions" / f"submission_{index:03d}.json"
        log_root = root / "slurm_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        base = ["sbatch", "--parsable", "--hold", "--account", a.account, "--partition", a.partition,
            "--qos", a.qos, "--nodes", "1", "--ntasks", "1", "--cpus-per-task", "4", "--mem", "32G",
            "--gpus", "a100:1", "--time", "04:00:00", "--signal", "B:TERM@180", "--chdir", str(Path(a.work)),
            "--job-name", "c0c1-night-" + root.name[:8], "--output", str(log_root / "%j.out"),
            "--error", str(log_root / "%j.err")]
        script_args = [str(HERE / "gpu_worker.sbatch"), str(HERE), str(Path(a.work)), str(Path(a.b_root))]
        test = subprocess.run([*base, "--test-only", *script_args, str(path), "0", str(a.worker_seconds)],
                              text=True, capture_output=True)
        if test.returncode:
            raise IntegrityError("Sol rejected the requested resources before submission:\n" + test.stderr + test.stdout)
        ledger = {"version": VERSION, "package_sha256": package_check(), "control_directory": str(root),
            "created_utc": utc(), "allocations_requested": a.allocations, "submission_complete": False, "jobs": []}
        for slot in range(a.allocations):
            command = list(base)
            if ledger["jobs"]:
                command += ["--dependency", "afterany:" + ledger["jobs"][-1]["job_id"]]
            command += script_args + [str(path), str(slot), str(a.worker_seconds)]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                ledger["submission_error"] = result.stderr + result.stdout
                current_json(path, ledger)
                if not ledger["jobs"]:
                    raise IntegrityError("Slurm submission failed: " + ledger["submission_error"])
                print("Slurm accepted a shorter allocation chain; queued work remains resumable:\n" + ledger["submission_error"])
                break
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise IntegrityError("Unrecognized Slurm job ID: " + result.stdout)
            ledger["jobs"].append({"slot": slot, "job_id": job_id,
                "stdout": str(log_root / f"{job_id}.out"), "stderr": str(log_root / f"{job_id}.err")})
            current_json(path, ledger)
        ledger["submission_complete"] = True
        current_json(path, ledger)
        current_json(HERE / "RUN_LOCATION.json", {"control_directory": str(root), "submission_file": str(path),
                                                    "jobs": ledger["jobs"]})
        ledger["released_job_ids"] = []
        for job in ledger["jobs"]:
            subprocess.run(["scontrol", "release", job["job_id"]], check=True)
            ledger["released_job_ids"].append(job["job_id"])
            current_json(path, ledger)
        print(json.dumps({"status": "submitted_training_not_yet_verified", "new_C0_runs": 11, "new_C1_runs": 12,
            "benchmark_evaluation": False, "jobs": ledger["jobs"], "control_directory": str(root),
            "completion_report": str(root / "STATUS.json"), "completion_marker": str(root / "TRAINING_COMPLETE.json"),
            "note": "Notebook/laptop can disconnect now. Slurm queues the allocations serially; completion is established by the report."}, indent=2))


def status(a):
    root = control_root(a)
    location = HERE / "RUN_LOCATION.json"
    if location.exists():
        info = read(location)
        ids = [j["job_id"] for j in info["jobs"]]
        for command in (["squeue", "-j", ",".join(ids), "-o", "%.12i %.10T %.12M %.45R"],
                        ["sacct", "-j", ",".join(ids), "--format=JobID,State,Elapsed,Timelimit,ExitCode"]):
            result = subprocess.run(command, text=True, capture_output=True)
            print(result.stdout or result.stderr)
    if (root / "STATUS.json").exists():
        value = read(root / "STATUS.json")
        print(json.dumps({k: value[k] for k in ("status", "C0_complete_of_12", "C1_complete_of_12",
            "new_training_runs_complete_of_23", "completed_units", "total_units", "benchmark_evaluation_performed")}, indent=2))
        for key, state in value["unit_state"].items():
            if state["status"] not in ("complete",):
                print(key, json.dumps(state))
    else:
        print("Slurm has not written training status yet; jobs may still be pending.")
    print("Run table:", root / "TRAINING_RUNS.csv")
    print("Results ZIP:", root / "C0_C1_training_receipts.zip")
    print("Completion marker:", root / "TRAINING_COMPLETE.json")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "preflight-profile", "submit", "worker", "unit", "solve", "train-one", "status"))
    p.add_argument("--work", default="/scratch/sshind38/capstone_c0_c1_work")
    p.add_argument("--b-root", default="/scratch/sshind38/capstone_approach_b")
    p.add_argument("--model", choices=("0.5B", "1.5B"), default="0.5B")
    p.add_argument("--parent-seed", type=int, choices=(17, 123), default=17)
    p.add_argument("--method", choices=("C0", "C1"), default="C0")
    p.add_argument("--seed", type=int, choices=SEEDS, default=42)
    p.add_argument("--attempt", default="attempt01")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--unit-id")
    p.add_argument("--submission")
    p.add_argument("--slot", type=int, default=0)
    p.add_argument("--worker-seconds", type=int, default=13800)
    p.add_argument("--allocations", type=int, default=6)
    p.add_argument("--account", default="class_fse57081354fall2026")
    p.add_argument("--partition", default="public")
    p.add_argument("--qos", default="class")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.action == "preflight":
        preflight(a)
    elif a.action == "preflight-profile":
        print(json.dumps(preflight_profile(a)))
    elif a.action == "submit":
        submit(a)
    elif a.action == "worker":
        return worker(a)
    elif a.action == "status":
        status(a)
    elif a.action == "unit":
        unit = next(u for u in units() if u["id"] == a.unit_id)
        if unit["profile"] != f"{a.model}_seed{a.parent_seed}":
            raise IntegrityError("Work unit selected another model/parent")
        execute_unit(a, unit)
    elif a.action == "solve":
        solve_selection(load_context(a), a.method)
    elif a.action == "train-one":
        train(load_context(a), a.method, a.seed, a.attempt, smoke=a.smoke)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionBlocked as error:
        print("BLOCKED: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(66)
    except IntegrityError as error:
        print("INTEGRITY CHECK FAILED: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(65)
    except ValueError as error:
        print("SCIENTIFIC INPUT OR RECEIPT CHECK FAILED: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(65)
    except subprocess.CalledProcessError as error:
        print("Child stage failed: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(error.returncode if error.returncode in (65, 66) else 1)

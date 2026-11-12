#!/usr/bin/env python3
"""Run reference-fit C0/C1 on Suraj's existing B runtime, without modifying B."""

import argparse
import ast
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "SOURCE_PINS.json").read_text())
BANK_SHA = "806dd78f38b37774391efc37d98318f0610e95dbeb9d53f810b54a03baf435ae"
PARENT_SHA = "efa258c9c4a8bffa85bb3be2026c187e9035f3918fe044967c4dafadcfa7f303"
B_SHA = "378316a7e9485ba0f0e089f674aadf00efa3c2b3dc8c174911f089a6d0fb5579"
BINS_SHA = "024b651bc18084c715b0b77a1db78c82e9e20040dc5ec671e7597a2604dd51e6"
TARGET = 318845


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def save_bytes(path, data):
    """Create only, atomically; identical retries allowed, different data preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise ValueError(f"Different output already exists: {path}; use a new output root/attempt")
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)  # a competing writer cannot overwrite an existing result


def save(path, obj):
    save_bytes(path, (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def save_rows(path, records):
    save_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records).encode())


def check_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}: {actual} != {expected}")
    return actual


def unique(records):
    if not records:
        raise ValueError("Empty records")
    for key in ("row_id", "problem_group_id"):
        ids = [r[key] for r in records]
        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid/duplicate {key}")


def artifact(c, *parts):
    return c.out / c.profile / c.identity[:16] / Path(*parts)


def load_context(a):
    root = Path(a.b_root).resolve()
    repo = Path(a.repo).resolve()
    condition = root / "outputs/B_method_B_document_v3/mixed/budget_1x"
    runtime = condition / "runtime"
    shared = root / "outputs/B_method_B_document_v3/shared"
    if not (runtime / "config.json").is_file():
        raise FileNotFoundError(f"Missing actual B runtime: {runtime}/config.json")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(root / "hf_cache"),
                      TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    for relative, expected in PINS["vendor"].items():
        check_hash(runtime / "vendor" / relative, expected)
    for relative, expected in PINS["repo"].items():
        check_hash(repo / relative, expected)
    check_hash(runtime / "parent_registry.json", PINS["parent_registry_sha256"])
    for path in (repo, runtime):
        sys.path.insert(0, str(path))
    import common
    import workbench
    import continuation
    if Path(common.__file__).resolve() != (runtime / "common.py").resolve():
        raise ValueError("Another B runtime is imported; start a fresh Python process")
    common.sources_valid()  # the existing B provenance verifies vendor AND historical parent records
    cfg = common.config()
    continuation.validate_protocol(root, cfg)
    for key, expected in (("epochs", 1), ("microbatch_size", 1), ("gradient_accumulation", 8),
                          ("warmup_updates", 10), ("learning_rate", 1e-4)):
        if cfg["training"].get(key) != expected:
            raise ValueError(f"B training recipe differs: {key}")
    train_source = continuation.adapted_train_source()
    training_ast = digest_ast(train_source)
    if training_ast != PINS["training_ast_sha256"]:
        raise ValueError("Live B training adaptation differs from uploaded/reviewed code; inspect it before proceeding")
    bank_path = shared / "mixed_candidate_bank.jsonl"
    parent_path = root / "data/train_n1000.jsonl"
    bins_path = shared / "parent_length_bins.json"
    b_path = condition / "1x/train_B.jsonl"
    for p, h in ((bank_path, BANK_SHA), (parent_path, PARENT_SHA), (bins_path, BINS_SHA), (b_path, B_SHA)):
        check_hash(p, h)
    if cfg.get("common_universe_sha256") != BANK_SHA or cfg.get("parent_manifest_sha256") != PARENT_SHA:
        raise ValueError("B execution config uses different parent/common bank")
    bank, parent, b_rows = rows(bank_path), rows(parent_path), rows(b_path)
    unique(bank); unique(parent); unique(b_rows)
    if (len(bank), len(parent), len(b_rows)) != (6409, 1000, 1000):
        raise ValueError("Unexpected bank/parent/B row count")
    if sum(r["supervised_tokens"] for r in parent) != TARGET or sum(r["supervised_tokens"] for r in b_rows) != 317506:
        raise ValueError("Stored supervision totals differ")
    bank_map = {r["row_id"]: r for r in bank}
    if any(bank_map.get(r["row_id"]) != r for r in b_rows):
        raise ValueError("B selection differs from the canonical bank")
    if {r["problem_group_id"] for r in bank} & {r["problem_group_id"] for r in parent}:
        raise ValueError("Bank/parent recorded group overlap")
    from capstone_person4.b_select import parent_edges, rebin, edge_report, validate
    validate(parent); validate(bank)
    edges = parent_edges(parent)
    if read(bins_path) != edge_report(edges) or rebin(bank, edges) != bank:
        raise ValueError("B length bins do not reproduce")
    rebinned = rebin(parent, edges)
    counts = Counter((r["dataset_id"], r["length_bin"]) for r in rebinned)
    if len(counts) != 20 or set(counts.values()) != {50}:
        raise ValueError("Parent source/length strata differ")
    if a.model not in ("0.5B", "1.5B") or a.parent_seed not in (17, 123):
        raise ValueError("Unsupported parent")
    profile, entry = common.profile(a.model, a.parent_seed)
    parent_check = workbench.parent_check(root, profile, entry)
    if not parent_check["ready"]:
        raise ValueError(json.dumps(parent_check))
    contract_path = common.data_paths(root)["contract"]
    check_hash(contract_path, cfg["formatting_contract_sha256"])
    contract = read(contract_path)
    dependencies = {"runtime/" + p.name: sha(p) for p in sorted(runtime.glob("*.py"))}
    dependencies.update({"runtime/config.json": sha(runtime / "config.json"),
                         "runtime/execution_protocol.json": sha(runtime / "execution_protocol.json"),
                         "runner": sha(__file__), "pins": sha(HERE / "SOURCE_PINS.json"),
                         "parent_weights": entry["expected_adapter_sha256"],
                         "contract": sha(contract_path), "bank": BANK_SHA, "parent": PARENT_SHA,
                         "B": B_SHA, "bins": BINS_SHA})
    dependencies.update({"repo/" + k: sha(repo / k) for k in PINS["repo"]})
    identity = digest(dependencies)
    return SimpleNamespace(a=a, root=root, repo=repo, out=Path(a.output).resolve(), condition=condition,
                           runtime=runtime, shared=shared, common=common, wb=workbench, cont=continuation,
                           cfg=cfg, entry=entry, profile=profile, bank=bank, parent=parent, rebinned=rebinned,
                           bank_path=bank_path, b_path=b_path, contract=contract, dependencies=dependencies,
                           identity=identity, train_source=train_source)


def digest_ast(source):
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()


def training_source_for_method(source, method):
    """Change B's reporting label only; preserve initialization and every update."""
    if method not in ("C0", "C1") or source.count("'preliminary_document_aligned_B'") != 1:
        raise ValueError("Unexpected B reporting label or method")
    return source.replace("'preliminary_document_aligned_B'", repr(f"preliminary_{method}_reference_fit_continuation"), 1)


def preflight(c):
    report = {"passed": True, "identity": c.identity, "profile": c.profile,
              "B_runtime": str(c.runtime), "parent_adapter": str(c.root / c.entry["adapter_relative_path"]),
              "bank_rows": len(c.bank), "B_rows": 1000, "B_tokens": 317506, "parent_one_pass_tokens": TARGET,
              "same_B_training_function_ast": True, "dependencies": c.dependencies,
              "GPU_used": False, "training_started": False, "main_training_cleared": False,
              "protection_status": "Unresolved reported 25 exclusions and 572 quarantines; score cache may be computed before selection is frozen"}
    path = artifact(c, "preflight.json")
    save(path, report)
    print(json.dumps({**report, "report": str(path)}, indent=2))


def encoder(c):
    p2 = c.common.load_person2()
    p2.select_model(c.a.model)
    tok = p2.get_tokenizer()
    encoded = []
    dataset = c.parent + c.bank
    print(f"Verifying tokenizer labels for {len(dataset)} rows; weights are not loaded in this step", flush=True)
    for number, row in enumerate(dataset, 1):
        e = p2.encode(row, tok, c.contract["prompt_template"], c.contract["max_total_tokens"])
        if e["counts"]["supervised_tokens"] != row["supervised_tokens"]:
            raise ValueError("Actual label count mismatch: " + row["row_id"])
        if e["labels"][0] != -100 or e["labels"][-1] != tok.eos_token_id:
            raise ValueError("Prompt/EOS mask mismatch: " + row["row_id"])
        encoded.append(e)
        if number % 250 == 0 or number == len(dataset):
            print(f"Labels verified: {number}/{len(dataset)}", flush=True)
    return p2, tok, encoded


def verify(c):
    _, tok, encoded = encoder(c)
    counts = [{"row_id": row["row_id"], "supervised_tokens": e["counts"]["supervised_tokens"],
               "encoded_sha256": digest({k: e[k] for k in ("input_ids", "labels")})}
              for row, e in zip(c.parent + c.bank, encoded)]
    path = artifact(c, "verification")
    save_rows(path / "actual_labels.jsonl", counts)
    report = {"passed": True, "identity": c.identity, "rows": len(counts), "actual_parent_tokens":
              sum(e["counts"]["supervised_tokens"] for e in encoded[:len(c.parent)]),
              "tokenizer_sha256": digest(tok.backend_tokenizer.to_str()),
              "labels_sha256": sha(path / "actual_labels.jsonl"), "weights_loaded": False}
    if report["actual_parent_tokens"] != TARGET:
        raise ValueError("Actual parent replay anchor differs")
    save(path / "verification.json", report)
    print(json.dumps({**report, "report": str(path / "verification.json")}, indent=2))


def verify_proof(c):
    report = read(artifact(c, "verification/verification.json"))
    if report.get("passed") is not True or report.get("identity") != c.identity:
        raise ValueError("Run verify with the exact current inputs/code first")
    check_hash(artifact(c, "verification/actual_labels.jsonl"), report["labels_sha256"])
    return report


def gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("This stage requires exactly one allocated NVIDIA GPU; use Sol Interactive Apps or Slurm")
    return torch


def score(c, smoke=False):
    verify_proof(c)
    torch = gpu()
    from capstone_person4.reference_scoring import score_batch, diagnostic_subset, validate_score
    p2, tok, encoded = encoder(c)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    environment = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}
    env_key = digest({"identity": c.identity, "dtype": str(dtype), "environment": environment,
                      "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                      "scoring_batch_size": 1})
    dest = artifact(c, "reference", env_key[:16])
    selected = diagnostic_subset(c.bank, 8) if smoke else c.bank
    by_id = {r["row_id"]: e for r, e in zip(c.parent + c.bank, encoded)}
    records = {}
    if (dest / "scores.jsonl").is_file() and not smoke:
        index = rows(dest / "scores.jsonl")
        for r in index:
            if r.get("measurement_key") != env_key or r.get("parent_sha256") != c.entry["expected_adapter_sha256"]:
                raise ValueError("Score index identity differs")
            records[r["row_id"]] = r
    started = time.monotonic()
    model = p2.core.build_model(dtype, adapter=c.root / c.entry["adapter_relative_path"], trainable=False)
    model.eval()
    maximum = 0.0
    if smoke or not (dest / "smoke.json").is_file():
        sample = diagnostic_subset(c.bank, 8)
        for i in range(0, len(sample), 2):
            batch = sample[i:i + 2]
            grouped = score_batch(model, [by_id[r["row_id"]] for r in batch], tok.eos_token_id, dtype)
            for row, together in zip(batch, grouped):
                single = score_batch(model, [by_id[row["row_id"]]], tok.eos_token_id, dtype)[0]
                validate_score(single, row["supervised_tokens"])
                validate_score(together, row["supervised_tokens"])
                maximum = max(maximum, abs(single["mean_nll"] - together["mean_nll"]))
        if maximum > 0.05:
            raise ValueError(f"Scoring smoke mean NLL mismatch: {maximum}; no tolerance was relaxed")
        proof = {"passed": True, "measurement_key": env_key, "parent_sha256": c.entry["expected_adapter_sha256"],
                 "rows": 8, "maximum_padded_row_mean_difference_nats": maximum, "environment": environment,
                 "precision": str(dtype), "batch_size": 1, "training_started": False}
        save(dest / "smoke.json", proof)
    if smoke:
        print(json.dumps({"smoke_passed": True, "directory": str(dest), "training_started": False}, indent=2))
        return
    for i, row in enumerate(selected):
        cache = dest / "rows" / (hashlib.sha256(row["row_id"].encode()).hexdigest() + ".json")
        if cache.is_file():
            record = read(cache)
            validate_score(record, row["supervised_tokens"])
            if (record.get("measurement_key") != env_key or record.get("row_id") != row["row_id"]
                    or record.get("row_sha256") != digest(row)):
                raise ValueError("Stale/wrong cached reference score")
        else:
            values = score_batch(model, [by_id[row["row_id"]]], tok.eos_token_id, dtype)[0]
            validate_score(values, row["supervised_tokens"])
            record = {**values, "row_id": row["row_id"], "row_sha256": digest(row), "status": "ok",
                      "measurement_key": env_key, "parent_sha256": c.entry["expected_adapter_sha256"]}
            save(cache, record)
        records[row["row_id"]] = {k: v for k, v in record.items() if k != "tokens"}
        if (i + 1) % 100 == 0:
            print(f"Scored/verified {i + 1}/{len(selected)}", flush=True)
    if sha(c.root / c.entry["adapter_relative_path"] / "adapter_model.safetensors") != c.entry["expected_adapter_sha256"]:
        raise ValueError("Parent adapter changed during scoring")
    ordered = [records[r["row_id"]] for r in c.bank]
    save_rows(dest / "scores.jsonl", ordered)
    report = {"complete": True, "identity": c.identity, "measurement_key": env_key, "rows": len(ordered),
              "scores_sha256": sha(dest / "scores.jsonl"), "parent_sha256": c.entry["expected_adapter_sha256"],
              "scope": "reference solution NLL, completion plus EOS; model weights unchanged", "environment": environment}
    save(dest / "completion.json", report)
    save(dest / "sessions" / (uuid.uuid4().hex + ".json"), {"wall_seconds": time.monotonic() - started,
         "rows": len(ordered), "includes_loading_and_reused_cache_checks": True})
    print(json.dumps({**report, "scores": str(dest / "scores.jsonl")}, indent=2))


def load_scores(c):
    candidates = list(artifact(c, "reference").glob("*/completion.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one complete scoring precision/runtime for this parent; found {len(candidates)}")
    report_path = candidates[0]
    report = read(report_path)
    check_hash(report_path.parent / "scores.jsonl", report["scores_sha256"])
    records = rows(report_path.parent / "scores.jsonl")
    ids = [r["row_id"] for r in records]
    if (report.get("complete") is not True or report.get("identity") != c.identity
            or len(ids) != len(set(ids)) or set(ids) != {r["row_id"] for r in c.bank}):
        raise ValueError("Incomplete/duplicate/stale reference scores")
    return records, report


def selection_root(c, method, bonus="1"):
    rule = {"method": method, "bonus": str(Fraction(bonus)) if method == "C1" else None}
    return artifact(c, "selection", method + "_" + digest(rule)[:12])


def request_candidates(c):
    """Candidate answers use the mathematical grader for BOTH training sources."""
    verify_proof(c)
    request = [{"problem_id": r["row_id"], "benchmark": "math", "question": r["problem"],
                "gold_answer": r["final_answer"], "source": r["dataset_id"], "split": "candidate"} for r in c.bank]
    path = artifact(c, "generation/questions.jsonl")
    save_rows(path, request)
    meta = {"identity": c.identity, "questions_sha256": sha(path), "rows": len(request),
            "candidate_grader": "p3-parser-v6 mathematical equivalence for both sources",
            "gold_role": "grading only; inference uses question field only",
            "parent_sha256": c.entry["expected_adapter_sha256"]}
    save(path.parent / "request.json", meta)
    print(json.dumps({**meta, "questions": str(path)}, indent=2))


def validate_generation(c):
    root = artifact(c, "generation")
    request = read(root / "request.json")
    check_hash(root / "questions.jsonl", request["questions_sha256"])
    config = read(root / "parent_candidates.config.json")
    expected = {"eval_file_sha256": request["questions_sha256"], "checkpoint_sha256": c.entry["expected_adapter_sha256"],
                "model": c.entry["model"], "revision": c.entry["revision"], "precision": "fp32",
                "decoding": "greedy", "batch_size": 16, "max_new_tokens": 1024, "question_text": "p1_clean",
                "parser_version": "p3-parser-v6", "eos_token_id": 151643, "add_special_tokens": False,
                "adapter_mode": "merged", "stop_at_question": False}
    if request.get("identity") != c.identity or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Candidate-generation checkpoint, question file or evaluation protocol differs")
    check_hash(c.runtime / "vendor/evaluation/evaluate.py", config["evaluate_py_sha256"])
    check_hash(c.runtime / "vendor/evaluation/answer_parser.py", config["answer_parser_sha256"])
    predictions = rows(root / "parent_candidates.jsonl")
    ids = [r["problem_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {r["row_id"] for r in c.bank}:
        raise ValueError("Candidate generations must cover each bank row exactly once")
    from capstone_person4_approachC1.generation_scoring import classify
    outcomes = []
    for p in predictions:
        if (p.get("checkpoint_sha256") != c.entry["expected_adapter_sha256"]
                or p.get("parser_version") != expected["parser_version"]
                or p.get("eval_config_hash") != config["eval_config_hash"]
                or p.get("model_revision") != c.entry["model"] + "@" + c.entry["revision"]):
            raise ValueError("Mixed candidate-generation identities")
        status, correct, flags = classify(p)
        if status == "execution_error":
            raise ValueError("Retry candidate execution errors before selecting")
        outcomes.append({"row_id": p["problem_id"], "mistake": status == "incorrect", "correct": correct,
                         "status": status, "flags": flags})
    return outcomes, {"predictions_sha256": sha(root / "parent_candidates.jsonl"),
                      "generation_config_sha256": sha(root / "parent_candidates.config.json"),
                      "questions_sha256": request["questions_sha256"]}


def generate(c):
    gpu()
    request_candidates(c)
    dest = artifact(c, "generation")
    output = dest / "parent_candidates.jsonl"
    if output.exists():
        cfg = read(dest / "parent_candidates.config.json")
        if (cfg.get("eval_file_sha256") != sha(dest / "questions.jsonl")
                or cfg.get("checkpoint_sha256") != c.entry["expected_adapter_sha256"]):
            raise ValueError("Existing generation belongs to other inputs/checkpoint")
        previous = rows(output)
        ids = [r["problem_id"] for r in previous]
        if len(ids) != len(set(ids)) or set(ids) - {r["row_id"] for r in c.bank}:
            raise ValueError("Existing generations have duplicate/unknown IDs; preserve and recover before resuming")
    command = [sys.executable, str(c.runtime / "vendor/evaluation/evaluate.py"),
               "--eval_file", str(dest / "questions.jsonl"), "--adapter", str(c.root / c.entry["adapter_relative_path"]),
               "--expect_train_manifest_sha256", PARENT_SHA, "--run_id", "C1_candidates_" + c.profile,
               "--out", str(output), "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024",
               "--question_text", "p1_clean"]
    subprocess.run(command, check=True)
    outcomes, meta = validate_generation(c)
    save_rows(dest / "outcomes.jsonl", outcomes)
    report = {**meta, "identity": c.identity, "complete": True, "rows": len(outcomes),
              "status_counts": dict(Counter(o["status"] for o in outcomes)), "outcomes_sha256": sha(dest / "outcomes.jsonl")}
    save(dest / "completion.json", report)
    print(json.dumps(report, indent=2))


def load_generation(c):
    report = read(artifact(c, "generation/completion.json"))
    outcomes, meta = validate_generation(c)
    if report.get("identity") != c.identity or report.get("complete") is not True or any(report.get(k) != v for k, v in meta.items()):
        raise ValueError("Candidate-generation receipt is stale")
    check_hash(artifact(c, "generation/outcomes.jsonl"), report["outcomes_sha256"])
    if rows(artifact(c, "generation/outcomes.jsonl")) != outcomes:
        raise ValueError("Classified outcomes differ from saved predictions")
    return outcomes, report


def select(c, method, bonus="1"):
    verify_proof(c)
    from capstone_person4.b_select import solve, hash32
    from capstone_person4.selection import fit_priorities
    from capstone_person4_approachC1.generation_scoring import c1_utilities
    dest = selection_root(c, method, bonus)
    if (dest / "selection.json").is_file():
        _, report = load_selection(c, method, bonus)
        print(json.dumps({"reused": True, "selection": str(dest), **report}, indent=2))
        return
    reproduced, reproduction = solve(c.rebinned, c.bank, 1)
    if (reproduced is None or {r["row_id"] for r in reproduced} != {r["row_id"] for r in rows(c.b_path)}
            or reproduction["actual_tokens"] != 317506 or reproduction["objective"] != 912175166):
        raise ValueError("Shared solver failed to reproduce the actual B selection")
    scores, coverage = load_scores(c)
    priorities = fit_priorities(c.bank, scores, 42)
    utility = {r["row_id"]: int(round(r["fit"] * 1000000)) for r in priorities}
    generation = None
    if method == "C1":
        value = Fraction(bonus)
        if value < 0:
            raise ValueError("Negative mistake bonus")
        outcomes, generation = load_generation(c)
        mistakes = {r["row_id"]: r["mistake"] for r in outcomes}
        utility = c1_utilities(priorities, mistakes, 1000000, value)
    chosen, solver = solve(c.rebinned, c.bank, 1, utilities=utility)
    if chosen is None:
        raise ValueError(f"No conforming selection: {solver.get('status')}; bounds were not relaxed")
    solver.update(student_scores_used=True, random_utility=None, embeddings_used=False,
                  utility_definition="round(fit * 1000000)" if method == "C0"
                  else "round(fit * 1000000) * (den + num * confirmed_mistake)")
    chosen.sort(key=lambda r: (hash32("42|training-order|" + r["row_id"]), r["row_id"]))
    diagnostics = None
    if method == "C1":
        c0, _ = solve(c.rebinned, c.bank, 1, utilities={r["row_id"]: int(round(r["fit"] * 1000000)) for r in priorities})
        if c0 is None:
            raise ValueError("Cannot establish C0 comparison for C1 diagnostic")
        left, right = {r["row_id"] for r in c0}, {r["row_id"] for r in chosen}
        share = sum(mistakes.values()) / len(mistakes)
        overlap = len(left & right) / len(left | right)
        diagnostics = {"mistake_share": share, "C0_C1_jaccard": overlap,
                       "degeneracy_flags": {"mistakes_rare": share < 0.05, "mistakes_near_universal": share > 0.95,
                                            "C1_almost_same_as_C0": overlap >= 0.95}}
        if any(diagnostics["degeneracy_flags"].values()):
            raise ValueError("C1 declared diagnostic gate failed: " + json.dumps(diagnostics))
    save_rows(dest / "train.jsonl", chosen)
    report = {"identity": c.identity, "method": method, "bonus": str(Fraction(bonus)) if method == "C1" else None,
              "rows": len(chosen), "actual_tokens": sum(r["supervised_tokens"] for r in chosen), "target_tokens": TARGET,
              "manifest_sha256": sha(dest / "train.jsonl"), "scores_sha256": coverage["scores_sha256"],
              "generation": generation, "C1_diagnostics": diagnostics, "solver": solver,
              "B_reproduction": {"same_ids": True, "tokens": 317506, "objective": reproduction["objective"]},
              "training_status": "not run", "eligibility_status": "not cleared by selection feasibility"}
    save(dest / "selection.json", report)
    print(json.dumps({**report, "directory": str(dest)}, indent=2))


def load_selection(c, method, bonus="1"):
    dest = selection_root(c, method, bonus)
    report = read(dest / "selection.json")
    check_hash(dest / "train.jsonl", report["manifest_sha256"])
    selected = rows(dest / "train.jsonl")
    unique(selected)
    if (report.get("identity") != c.identity or report.get("method") != method
            or report.get("rows") != len(selected) or len(selected) != 1000
            or report.get("bonus") != (str(Fraction(bonus)) if method == "C1" else None)
            or report["actual_tokens"] != sum(r["supervised_tokens"] for r in selected)
            or abs(report["actual_tokens"] - TARGET) > TARGET * Fraction(5, 1000)):
        raise ValueError("Selected manifest identity/count/budget differs")
    bank = {r["row_id"]: r for r in c.bank}
    if any(bank.get(r["row_id"]) != r for r in selected):
        raise ValueError("Selected row changed from the bank")
    _, coverage = load_scores(c)
    if report["scores_sha256"] != coverage["scores_sha256"]:
        raise ValueError("Selection was made from other scores")
    if method == "C1" and report.get("generation") != load_generation(c)[1]:
        raise ValueError("C1 selection was made from other generation outcomes")
    return selected, report


def eligibility(c, path):
    """A review result, not permission to train: its cited evidence must exist."""
    if not path:
        raise ValueError("Main training needs --eligibility-report from the actual flagged-row review; scoring/smoke can proceed separately")
    path = Path(path).resolve()
    result = read(path)
    if (result.get("bank_sha256") != BANK_SHA or result.get("status") != "resolved"
            or result.get("unresolved_flags") != 0 or result.get("confirmed_ineligible_rows") != 0
            or not result.get("reviewer") or not result.get("reviewed_utc") or not result.get("evidence")):
        raise ValueError("Eligibility is unresolved or the bank needs revision across B/C0/C1")
    for evidence in result["evidence"]:
        check_hash(path.parent / evidence["path"], evidence["sha256"])
    return sha(path)


def train(c, method, seed, attempt, bonus="1", smoke=False, eligibility_report=None):
    verify_proof(c)
    selected, selection = load_selection(c, method, bonus)
    if seed not in (42, 137, 2026):
        raise ValueError("Continuation seed must be 42, 137 or 2026")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise ValueError("Use a simple new attempt name")
    review_hash = None if smoke else eligibility(c, eligibility_report)
    smoke_key = digest({"identity": c.identity, "selection": digest(selection), "method": method})
    if not smoke:
        proofs = list(artifact(c, "smokes", method).glob("*/completion.json"))
        if not any(read(p).get("smoke_key") == smoke_key and read(p).get("reload_probe_matched") is True for p in proofs):
            raise ValueError("Run a matching method training smoke first")
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
    dest = artifact(c, "smokes" if smoke else "runs", method, attempt if smoke else f"seed{seed}_" + attempt)
    if dest.exists():
        raise ValueError("Existing training attempt preserved; choose a new attempt")
    training_rows = selected[:32] if smoke else selected
    inputs = artifact(c, "training_inputs", method, digest(selection)[:12], "smoke" if smoke else "full")
    save_rows(inputs / "train.jsonl", training_rows)
    for name in ("prompt_template.json", "tokenization_policy.json"):
        save_bytes(inputs / name, (c.condition / "training_inputs" / name).read_bytes())
    # Retain B's initialization/update/save loop. Change its B-specific reporting
    # label so C0/C1 checkpoints are not mislabeled as B experiments.
    execution_source = training_source_for_method(c.train_source, method)
    exec(compile(execution_source, str(c.runtime / "continuation.py") + "::reused_train", "exec"), core.__dict__)
    original_parent = c.root / c.entry["adapter_relative_path"]
    parent_groups = {r["problem_group_id"] for r in c.parent}
    counts = [p2.encode(r, tok, c.contract["prompt_template"], c.contract["max_total_tokens"])["counts"]["supervised_tokens"]
              for r in training_rows]
    expected_tokens = sum(counts)
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
        "smoke_only": smoke, "eligibility_report_sha256": review_hash, "continuation_training_seed": seed,
        "shared_universe_sha256": BANK_SHA, "initial_optimizer_state": "empty", "fresh_warmup_updates": 10,
        "original_parent_adapter_sha256": c.entry["expected_adapter_sha256"]})
    core.train(args)
    result = read(dest / "training_summary.json")
    if (not result.get("complete") or result["processed_tokens"] != expected_tokens
            or set(result["seen_ids"]) != {r["row_id"] for r in training_rows}
            or set(result["seen_groups"]) != {r["problem_group_id"] for r in training_rows}):
        raise ValueError("Training consumption/IDs did not reconcile")
    check_hash(original_parent / "adapter_model.safetensors", c.entry["expected_adapter_sha256"])
    from capstone_person4.continuation import verify_adapter_update
    checkpoint = Path(result["checkpoint"])
    changed = verify_adapter_update(original_parent / "adapter_model.safetensors", checkpoint / "adapter/adapter_model.safetensors")
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
              "eligibility_report_sha256": review_hash, "training_seconds": result["training_seconds"]}
    save(dest / "completion.json", report)
    print(json.dumps({**report, "run_directory": str(dest)}, indent=2))


def evaluate(c, run, attempt):
    gpu()
    if not run:
        raise ValueError("Supply --run parent or --run /absolute/completed/run_directory")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise ValueError("Use a simple evaluation attempt")
    if run == "parent":
        adapter = c.root / c.entry["adapter_relative_path"]
        manifest, label = PARENT_SHA, "parent"
        expected_adapter = c.entry["expected_adapter_sha256"]
    else:
        run = Path(run).resolve()
        cfg = read(run / "run_config.json")
        summary = read(run / "training_summary.json")
        if (cfg.get("model") != c.entry["model"] or cfg.get("revision") != c.entry["revision"]
                or cfg.get("parent_adapter_sha256") != c.entry["expected_adapter_sha256"]
                or cfg.get("shared_universe_sha256") != BANK_SHA or summary.get("complete") is not True):
            raise ValueError("Run uses another model/parent/bank, or is incomplete")
        if (run / "completion.json").is_file():
            completion = read(run / "completion.json")
            if completion.get("identity") != c.identity or completion.get("smoke_only") is not False or completion.get("passed") is not True:
                raise ValueError("Only reconciled full bridge runs can be evaluated")
            method = completion["method"]
            _, selected = load_selection(c, method, c.a.bonus)
            if completion["selection_sha256"] != digest(selected):
                raise ValueError("Evaluated run is bound to another selection")
            check_hash(run / "run_config.json", completion["run_config_sha256"])
            check_hash(run / "training_summary.json", completion["summary_sha256"])
            expected_adapter = completion["adapter_sha256"]
        elif run.is_relative_to(c.condition / "runs"):
            receipt = read(run / "B_completion_audit.json")
            if receipt.get("passed") is not True or cfg.get("manifest_sha256") != B_SHA or summary.get("processed_tokens") != 317506:
                raise ValueError("B run lacks compatible completed exposure evidence")
            expected_adapter = sha(Path(summary["checkpoint"]) / "adapter/adapter_model.safetensors")
        else:
            raise ValueError("Unknown run lineage")
        adapter = Path(summary["checkpoint"]) / "adapter"
        manifest, label = cfg["manifest_sha256"], run.name
    check_hash(adapter / "adapter_model.safetensors", expected_adapter)
    expected_questions = {"dev_gsm": "3cb2c71007d23e01cb70b6370c1ad82ad5bcea3527c7f3258b4814cf2e3d1cc6",
                          "dev_math": "641460fdc55beff591daad0cc7530a3f933eb263c3dfdf8a0534fcfaf8196416"}
    for benchmark, expected in expected_questions.items():
        source = c.runtime / "vendor/evaluation/data" / (benchmark + ".jsonl")
        check_hash(source, expected)
        dest = artifact(c, "development", label, attempt, benchmark + ".jsonl")
        if dest.is_file():
            previous = rows(dest)
            ids = [p["problem_id"] for p in previous]
            old = read(dest.with_suffix(".config.json"))
            if (len(ids) != len(set(ids)) or old.get("eval_file_sha256") != expected
                    or old.get("checkpoint_sha256") != expected_adapter):
                raise ValueError("Existing evaluation IDs/config differ; preserve and recover")
        command = [sys.executable, str(c.runtime / "vendor/evaluation/evaluate.py"), "--eval_file", str(source),
                   "--adapter", str(adapter), "--expect_train_manifest_sha256", manifest, "--out", str(dest),
                   "--run_id", label + "_" + benchmark, "--precision", "fp32", "--batch_size", "16",
                   "--max_new_tokens", "1024", "--question_text", "p1_clean"]
        subprocess.run(command, check=True)
        preds = rows(dest)
        if len(preds) != 500 or {p["problem_id"] for p in preds} != {p["problem_id"] for p in rows(source)}:
            raise ValueError("Incomplete/duplicate benchmark coverage")
        print(json.dumps({"predictions": str(dest), "correct": sum(p["correct"] is True for p in preds), "total": 500}))


def paired_counts(earlier, later):
    if not earlier or set(earlier) != set(later) or any(type(v) is not bool for v in list(earlier.values()) + list(later.values())):
        raise ValueError("Paired outcomes require the same nonempty IDs and boolean correctness")
    counts = Counter((earlier[i], later[i]) for i in earlier)
    return {"retained_success": counts[True, True], "improvement": counts[False, True],
            "regression": counts[True, False], "persistent_failure": counts[False, False]}


def compare(c, earlier, later):
    if not earlier or not later:
        raise ValueError("Supply --earlier and --later native evaluator JSONL files")
    left, right = rows(earlier), rows(later)
    lc, rc = read(Path(earlier).with_suffix(".config.json")), read(Path(later).with_suffix(".config.json"))
    fields = ("model", "revision", "precision", "max_new_tokens", "batch_size", "stop_at_question", "question_text",
              "prompt_sha256", "add_special_tokens", "decoding", "eos_token_id", "adapter_mode", "gpu", "cuda",
              "parser_version", "math_verify", "eval_file_sha256", "answer_parser_sha256", "evaluate_py_sha256")
    if any(lc.get(k) != rc.get(k) for k in fields) or lc.get("model") != c.entry["model"] or lc.get("revision") != c.entry["revision"]:
        raise ValueError("Paired predictions used different questions/models/evaluation settings")
    for pred, cfg in ((left, lc), (right, rc)):
        if len(pred) != len({p["problem_id"] for p in pred}):
            raise ValueError("Duplicate prediction IDs")
        if any(p.get("checkpoint_sha256") != cfg["checkpoint_sha256"] or p.get("parser_version") != cfg["parser_version"]
               or p.get("eval_config_hash") != cfg["eval_config_hash"] for p in pred):
            raise ValueError("Mixed prediction identities")
    from capstone_person4.evaluation import paired_difference
    lm, rm = {p["problem_id"]: p["correct"] for p in left}, {p["problem_id"]: p["correct"] for p in right}
    counts = paired_counts(lm, rm)
    report = {"earlier": str(Path(earlier).resolve()), "later": str(Path(later).resolve()),
              "earlier_sha256": sha(earlier), "later_sha256": sha(later), "paired_outcomes": counts,
              "earlier_correct": sum(lm.values()), "later_correct": sum(rm.values()), "questions": len(lm),
              "later_minus_earlier": paired_difference(rm, lm),
              "scope": "paired item uncertainty for saved checkpoints; training-control lineage must also be checked; no seed-variation interval"}
    dest = artifact(c, "comparisons", digest(report)[:16] + ".json")
    save(dest, report)
    print(json.dumps({**report, "report": str(dest)}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("preflight", "verify", "score-smoke", "score", "select", "candidate-request", "generate", "train", "evaluate", "compare"))
    ap.add_argument("--b-root", default="/scratch/sshind38/capstone_approach_b")
    ap.add_argument("--repo", default=str(HERE.parent / "repo"))
    ap.add_argument("--output", default=str(HERE.parent / "outputs/C0_C1_B_bridge_v1"))
    ap.add_argument("--model", choices=("0.5B", "1.5B"), default="0.5B")
    ap.add_argument("--parent-seed", type=int, choices=(17, 123), default=17)
    ap.add_argument("--method", choices=("C0", "C1"), default="C0")
    ap.add_argument("--bonus", default="1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attempt", default="pilot01")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eligibility-report")
    ap.add_argument("--run")
    ap.add_argument("--earlier")
    ap.add_argument("--later")
    a = ap.parse_args(argv)
    try:
        c = load_context(a)
        import fcntl
        lock_path = c.out / ".locks" / (digest({"identity": c.identity, "action": a.action,
                    "method": a.method, "seed": a.seed, "attempt": a.attempt, "bonus": a.bonus}) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This exact stage is already running; duplicate writer refused")
        if a.action == "preflight": preflight(c)
        elif a.action == "verify": verify(c)
        elif a.action in ("score-smoke", "score"): score(c, a.action == "score-smoke")
        elif a.action == "select": select(c, a.method, a.bonus)
        elif a.action == "candidate-request": request_candidates(c)
        elif a.action == "generate": generate(c)
        elif a.action == "train": train(c, a.method, a.seed, a.attempt, a.bonus, a.smoke, a.eligibility_report)
        elif a.action == "evaluate": evaluate(c, a.run, a.attempt)
        elif a.action == "compare": compare(c, a.earlier, a.later)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

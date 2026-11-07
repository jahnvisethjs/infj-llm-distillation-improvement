"""Manifest selection and continuation through Person 2's existing update loop."""

from collections import Counter
import csv
import io
from pathlib import Path
from types import SimpleNamespace

from .allocation import AllocationError, allocate
from .audit import gate, parent_exposure
from .config import artifact_root, stage_fingerprint
from .data import load_prepared
from .person2_adapter import Person2Adapter, continuation_source, notebook_sources
from .provenance import PipelineError, digest, read_json, read_rows, sha, source_hashes, unique_rows, write_bytes, write_json, write_rows
from .reference_scoring import load_scores
from .selection import fit_priorities, random_priorities, stratum


def verify_adapter_update(parent_path, continued_path):
    """Check tensor values; serialization metadata alone must not count as an update."""
    from safetensors import safe_open
    import torch
    changed = []
    with safe_open(str(parent_path), framework="pt", device="cpu") as parent, \
            safe_open(str(continued_path), framework="pt", device="cpu") as continued:
        names = set(parent.keys())
        if names != set(continued.keys()) or not names or any("lora_" not in name for name in names):
            raise PipelineError("Continued adapter tensor identities differ from the original LoRA parent")
        for name in sorted(names):
            before, after = parent.get_tensor(name), continued.get_tensor(name)
            if before.shape != after.shape or not torch.isfinite(after).all():
                raise PipelineError("Continued adapter has incompatible or nonfinite tensors")
            if not torch.equal(before, after):
                changed.append(name)
    if not changed:
        raise PipelineError("Continuation did not change any parent adapter tensor values")
    return changed


def budget_and_coverage(cfg, candidate_rows):
    parent_rows, _ = parent_exposure(cfg)
    allocator = cfg["allocation"]
    if allocator.get("budget_kind") == "replay":
        adapter = Person2Adapter(cfg)
        tokenizer = adapter.tokenizer()
        target = sum(adapter.encode(r, tokenizer)["counts"]["supervised_tokens"] for r in parent_rows)
        if allocator.get("expected_replay_tokens", target) != target:
            raise PipelineError("Re-encoded parent budget differs from declared replay anchor")
        origin = "verified unique parent rows re-encoded once; historical repeats excluded"
    elif allocator.get("budget_kind") == "pilot":
        target = allocator.get("target_tokens")
        origin = "explicit provisional pilot token budget; not a verified replay anchor"
    else:
        raise PipelineError("Resolve allocation.budget_kind as replay or pilot")
    proportions = allocator.get("stratum_proportions")
    if proportions is None:
        proportions = dict(Counter(stratum(r) for r in parent_rows))
    if set(proportions) != {stratum(r) for r in candidate_rows}:
        raise PipelineError("Parent coverage differs from available candidate strata; explicitly declare shared stratum_proportions")
    return target, proportions, origin


def select(cfg, method):
    gate(cfg, "select")
    if method not in ("B", "C0"):
        raise PipelineError("This phase implements only B and C0")
    rows, preparation = load_prepared(cfg)
    seed = cfg["allocation"]["seed"]
    coverage = None
    if method == "C0":
        scores, coverage = load_scores(cfg)
        priorities = fit_priorities(rows, scores, seed)
    else:
        priorities = random_priorities(rows, seed)
    target, proportions, origin = budget_and_coverage(cfg, rows)
    settings = cfg["allocation"]
    root = artifact_root(cfg, "select") / "selection" / method
    try:
        selected, report = allocate(priorities, target, proportions, settings["tolerance"],
                                    settings["boundary_window"], settings["max_swaps"], settings["max_row_counts"])
    except AllocationError as exc:
        write_json(root / "allocation_failure.json", exc.report)
        raise
    by_id = {row["row_id"]: row for row in rows}
    manifest = [by_id[r["row_id"]] for r in selected]
    write_rows(root / "manifest.jsonl", manifest)
    table = [{k: r.get(k) for k in ("row_id", "problem_group_id", "stratum", "utility", "fit", "mean_nll", "tie_key", "priority_rank", "supervised_tokens")} for r in selected]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    write_bytes(root / "selection_table.csv", buffer.getvalue().encode("utf-8"))
    report.update({"synthetic": False, "method": method, "version": "reference-fit-v1", "budget_origin": origin,
                   "stratum_proportions": proportions, "manifest_sha256": sha(root / "manifest.jsonl"),
                   "selected_data_hash": digest(manifest), "config_fingerprint": cfg["config_fingerprint"],
                   "stage_fingerprint": stage_fingerprint(cfg, "select"),
                   "candidate_sha256": preparation["candidate_sha256"], "score_fingerprint": coverage["measurement_key"] if coverage else None,
                   "parent_fingerprint": cfg["parent"]["adapter_sha256"], "code": source_hashes(),
                   "prompt_tokens": sum(r["prompt_tokens"] for r in manifest),
                   "sequence_tokens": sum(r["total_tokens"] for r in manifest),
                   "source_rows": dict(Counter(r["dataset_id"] for r in manifest)),
                   "source_tokens": {key: sum(r["supervised_tokens"] for r in manifest if r["dataset_id"] == key)
                                     for key in sorted({r["dataset_id"] for r in manifest})}})
    write_json(root / "allocation_report.json", report)
    write_json(root / "ordered_ids.json", [r["row_id"] for r in manifest])
    other_path = root.parent / ("C0" if method == "B" else "B") / "allocation_report.json"
    if other_path.exists():
        other = read_json(other_path)
        if other.get("stage_fingerprint") != report["stage_fingerprint"]:
            raise PipelineError("Comparison arm selection belongs to different dependencies")
        write_json(root.parent / "pair_budget_report.json", {
            "target_tokens": target, "absolute_pairwise_difference": abs(report["actual_tokens"] - other["actual_tokens"]),
            "B_tokens": report["actual_tokens"] if method == "B" else other["actual_tokens"],
            "C0_tokens": report["actual_tokens"] if method == "C0" else other["actual_tokens"],
            "stage_fingerprint": report["stage_fingerprint"]})
    return report


def load_selection(cfg, method):
    if method not in ("B", "C0"):
        raise PipelineError("Expected B or C0 selection")
    root = artifact_root(cfg, "select") / "selection" / method
    report = read_json(root / "allocation_report.json")
    rows = read_rows(root / "manifest.jsonl")
    unique_rows(rows)
    if (report.get("synthetic") is not False or report.get("stage_fingerprint") != stage_fingerprint(cfg, "select")
            or report.get("method") != method
            or report.get("manifest_sha256") != sha(root / "manifest.jsonl")
            or report.get("selected_data_hash") != digest(rows)
            or report.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]):
        raise PipelineError("Selected manifest is synthetic, stale or belongs to another parent/method")
    candidates, prep = load_prepared(cfg)
    common = {r["row_id"]: r for r in candidates}
    if report["candidate_sha256"] != prep["candidate_sha256"] or any(common.get(r["row_id"]) != r for r in rows):
        raise PipelineError("Selected data differs from its frozen candidate universe")
    total = sum(r["supervised_tokens"] for r in rows)
    if total != report["actual_tokens"] or abs(total - report["target_tokens"]) > report["target_tokens"] * report["tolerance"]:
        raise PipelineError("Selected token totals fail budget reconciliation")
    if dict(Counter(stratum(r) for r in rows)) != {k: v for k, v in report["quotas"].items() if v}:
        raise PipelineError("Selected coverage differs from allocation quotas")
    return rows, report


def train(cfg, method, attempt, resume=None, smoke=False):
    gate(cfg, "train")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise PipelineError("Use a simple new attempt name, not a path")
    rows, selection = load_selection(cfg, method)
    if not smoke:
        proofs = []
        for path in (artifact_root(cfg) / "training_smoke").glob("*/*/token_reconciliation.json"):
            proof = read_json(path)
            if (proof.get("training_smoke_only") is True and proof.get("stage_fingerprint") == stage_fingerprint(cfg, "train")
                    and proof.get("parent_fingerprint") == cfg["parent"]["adapter_sha256"]
                    and proof.get("run_config_sha256") == sha(path.parent / "run_config.json")):
                proofs.append(proof)
        if not proofs:
            raise PipelineError("Run a matching train --smoke before full continuation; no complete disposable smoke evidence found")
    if method == "C0":
        _, coverage = load_scores(cfg)
        if selection["score_fingerprint"] != coverage["measurement_key"]:
            raise PipelineError("Selection reference cache changed")
    adapter = Person2Adapter(cfg, continuation=True)
    core = adapter.core
    tokenizer = adapter.tokenizer()
    if adapter.tokenizer_fingerprint != load_prepared(cfg)[1]["tokenizer_fingerprint"]:
        raise PipelineError("Training tokenizer differs from prepared/scored tokenizer")
    encoded = [adapter.encode(r, tokenizer) for r in rows]
    if sum(e["counts"]["supervised_tokens"] for e in encoded) != selection["actual_tokens"]:
        raise PipelineError("Actual trainer labels differ from selected token budget")
    root = artifact_root(cfg)
    # A smoke has a separate tiny manifest and can never be labeled a controlled arm.
    if smoke:
        if resume:
            raise PipelineError("Training smoke checks do not resume; use a new disposable attempt")
        rows = rows[:min(4, len(rows))]
        encoded = encoded[:len(rows)]
    inputs = root / "training_inputs" / method / ("smoke" if smoke else "full")
    write_rows(inputs / "manifest.jsonl", rows)
    write_json(inputs / "template.json", {"template": adapter.format["prompt_template"]})
    write_json(inputs / "policy.json", {"confirmed_with_person1": True, "add_special_tokens": False,
               "append_eos": True, "max_total_tokens": cfg["data"]["max_total_tokens"],
               "provenance": {"format_sha256": load_prepared(cfg)[1]["evidence"]["release"]["format_sha256"]}})
    write_bytes(inputs / "reused_continuation_engine.py", continuation_source(notebook_sources(cfg["integration"]["notebook"])["trainer_core.py"]).encode())
    recipe = cfg["training"]
    args = SimpleNamespace(data=inputs / "manifest.jsonl", template=inputs / "template.json", policy=inputs / "policy.json",
                           mode="continuation", arm=method, epochs=1, seed=recipe["seed"], accumulation=recipe["accumulation"],
                           lr=recipe["learning_rate"], save_every=recipe["save_every"], parent=Path(cfg["parent"]["adapter"]),
                           resume=Path(resume).resolve() if resume else None, stop_after=None, clearance=None,
                           verification=inputs / "verification.json", warmup_updates=recipe["warmup_updates"],
                           compute_dtype=cfg["scoring"]["dtype"],
                           out=root / ("training_smoke" if smoke else "runs") / method / attempt)
    if args.out.exists():
        raise PipelineError("Use a new attempt name; existing training output is preserved")
    current = {"manifest_sha256": sha(args.data), "template_sha256": sha(args.template), "policy_sha256": sha(args.policy),
               "model": cfg["parent"]["model"], "revision": cfg["parent"]["revision"],
               "code_sha256": stage_fingerprint(cfg, "train"), "notebook_sha256": sha(cfg["integration"]["notebook"]),
               "config_fingerprint": stage_fingerprint(cfg, "train"), "parent_fingerprint": cfg["parent"]["adapter_sha256"],
               "selection_sha256": digest(selection), "training_smoke_only": smoke}
    core.fingerprint = lambda unused: dict(current)
    # core.train checks this again before loading weights. All real gates already ran above.
    def guard(unused, actual_rows, fingerprint):
        verification = read_json(args.verification)
        if verification.get("fingerprint") != fingerprint or verification.get("passed") is not True or actual_rows != rows:
            raise PipelineError("Continuation token/provenance verification is stale")
        return None
    core.provenance_guard = guard
    write_json(args.verification, {"fingerprint": current, "passed": True,
               "actual_supervised_tokens": sum(e["counts"]["supervised_tokens"] for e in encoded)})
    # This call creates a new AdamW and uses the adapted parent initialization;
    # only an explicit resume restores this same arm's own optimizer and RNG state.
    parent_before = sha(Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors")
    core.train(args)
    summary = read_json(args.out / "training_summary.json")
    actual_total = sum(e["counts"]["supervised_tokens"] for e in encoded)
    if (summary.get("complete") is not True or summary.get("processed_tokens") != actual_total
            or set(summary.get("seen_ids", [])) != {r["row_id"] for r in rows}
            or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in rows}
            or sha(Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors") != parent_before):
        raise PipelineError("Continuation finished without valid token/exposure/parent integrity reconciliation")
    checkpoint_sha256 = sha(Path(summary["checkpoint"]) / "adapter/adapter_model.safetensors")
    changed_tensors = verify_adapter_update(Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors",
                                           Path(summary["checkpoint"]) / "adapter/adapter_model.safetensors")
    report = {"valid_controlled_run": not smoke, "training_smoke_only": smoke, "synthetic": False,
              "method": method, "actual_tokens": actual_total, "expected_tokens": actual_total,
              "target_tokens": selection["target_tokens"] if not smoke else actual_total,
              "budget_tolerance": selection["tolerance"], "candidate_sha256": selection["candidate_sha256"],
              "training_seconds": summary.get("training_seconds"), "peak_allocated_GB": summary.get("peak_allocated_GB"),
              "parent_unchanged": True, "parent_fingerprint": parent_before, "config_fingerprint": cfg["config_fingerprint"],
              "stage_fingerprint": stage_fingerprint(cfg, "train"), "evaluation_status": "pending",
              "changed_adapter_tensors": changed_tensors,
              "code": source_hashes(),
              "selection_sha256": digest(selection), "run_config_sha256": sha(args.out / "run_config.json"),
              "training_summary_sha256": sha(args.out / "training_summary.json"),
              "checkpoint": summary["checkpoint"], "checkpoint_sha256": checkpoint_sha256}
    write_json(args.out / "resolved_config.json", cfg)
    write_json(args.out / "token_reconciliation.json", report)
    return {**report, "run_directory": str(args.out)}

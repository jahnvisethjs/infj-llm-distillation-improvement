"""Reconcile the completed C0 arm (and B when trained here) into an unevaluated handoff."""

from pathlib import Path

from .audit import gate
from .bindings import verified
from .config import artifact_root, stage_fingerprint
from .continuation import load_selection
from .data import load_prepared
from .provenance import PipelineError, digest, read_json, sha, write_json
from .reference_scoring import load_scores


def finalize(cfg, b_run, c0_run):
    gate(cfg, "select")
    _, preparation = load_prepared(cfg)
    _, coverage = load_scores(cfg)
    arms = {}
    for method, run in (("B", b_run), ("C0", c0_run)):
        if run is None:
            continue
        root = Path(run).resolve()
        rows, selection = load_selection(cfg, method)
        report = read_json(root / "token_reconciliation.json")
        summary = read_json(root / "training_summary.json")
        run_config = read_json(root / "run_config.json")
        expected_tokens = sum(r["supervised_tokens"] for r in rows)
        if (report.get("valid_controlled_run") is not True or report.get("training_smoke_only") is not False
                or report.get("synthetic") is not False or report.get("method") != method
                or report.get("stage_fingerprint") != stage_fingerprint(cfg, "train")
                or report.get("selection_sha256") != digest(selection) or report.get("parent_unchanged") is not True
                or report.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]
                or report.get("actual_tokens") != expected_tokens or summary.get("processed_tokens") != expected_tokens
                or summary.get("complete") is not True or set(summary.get("seen_ids", [])) != {r["row_id"] for r in rows}
                or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in rows}
                or run_config.get("seed") != cfg["training"]["seed"] or run_config.get("arm") != method
                or run_config.get("mode") != "continuation"
                or run_config.get("optimizer_initialization") != "fresh_from_verified_parent_weights"
                or run_config.get("parent_adapter_sha256") != cfg["parent"]["adapter_sha256"]):
            raise PipelineError(f"{method} is not a valid reconciled full arm")
        verified(root / "run_config.json", report["run_config_sha256"])
        verified(root / "training_summary.json", report["training_summary_sha256"])
        checkpoint = Path(report["checkpoint"]) / "adapter/adapter_model.safetensors"
        verified(checkpoint, report["checkpoint_sha256"])
        arms[method] = {"run_directory": str(root), "reconciliation_sha256": sha(root / "token_reconciliation.json"),
                        "checkpoint": str(checkpoint.parent), "checkpoint_sha256": report["checkpoint_sha256"],
                        "actual_tokens": expected_tokens, "selection_sha256": selection["manifest_sha256"]}
    if "B" in arms and arms["B"]["run_directory"] == arms["C0"]["run_directory"]:
        raise PipelineError("B and C0 must be independent training jobs")
    _, c0_selection = load_selection(cfg, "C0")
    external_b = None if "B" in arms else {
        "status": "trained outside this pipeline",
        "requirement": "same parent, same frozen candidate universe, same recipe, supervised tokens within tolerance of the target",
        "target_tokens": c0_selection["target_tokens"], "tolerance": c0_selection["tolerance"],
        "matched_budget_B_manifest": str(artifact_root(cfg, "select") / "selection" / "B" / "manifest.jsonl")}
    verified(Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors", cfg["parent"]["adapter_sha256"])
    report = {"approach_complete": True, "synthetic": False, "evaluation_status": "pending",
              "stage_fingerprint": stage_fingerprint(cfg, "train"), "parent": cfg["parent"],
              "candidate_sha256": preparation["candidate_sha256"], "measurement_key": coverage["measurement_key"],
              "recipe": cfg["training"], "arms": arms,
              "pairwise_token_difference": abs(arms["B"]["actual_tokens"] - arms["C0"]["actual_tokens"]) if "B" in arms else None,
              "external_B": external_b,
              "interpretation": "Completed continuation checkpoints; accuracy and independent correctness unmeasured"}
    root = artifact_root(cfg) / "handoffs" / digest(arms)[:16]
    write_json(root / "approach_completion.json", report)
    write_json(root / "evaluation_handoff.json", {**report, "protected_question_sources": cfg.get("protection", {}).get("question_manifests", [])})
    return {**report, "report_directory": str(root)}

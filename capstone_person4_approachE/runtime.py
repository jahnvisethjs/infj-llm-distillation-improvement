"""Reuse Person 2's continuation through isolated, restored E bindings."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
from pathlib import Path
from unittest.mock import patch

from capstone_person4 import continuation
from capstone_person4.audit import gate
from capstone_person4.bindings import verified
from capstone_person4.config import stage_fingerprint
from capstone_person4.provenance import PipelineError, digest, read_json, sha, write_json
from . import selection as e


def attempt_name(value):
    if not value or Path(value).name != value or value in (".", "..") or "\\" in value:
        raise PipelineError("Use a simple new attempt name, not a path")
    return value


def with_seed(cfg, seed):
    if type(seed) is not int or seed < 0:
        raise PipelineError("Continuation seeds must be nonnegative integers")
    current = deepcopy(cfg)
    current["training"]["seed"] = seed
    current.pop("config_fingerprint", None)
    current["config_fingerprint"] = digest(current)
    return current


def run_directory(cfg, attempt, smoke=False):
    return e.root(cfg) / ("training_smoke" if smoke else "runs") / "E" / attempt_name(attempt)


def validate_run(cfg, method, directory, smoke=False):
    """Portable checkpoint lookup; saved records are verified, never rewritten."""
    directory = Path(directory).resolve()
    if method == "E":
        rows, selection = e.load_selection(cfg)
        expected_stage = e.fingerprint(cfg, "train")
    else:
        rows, selection = e.load_control(cfg, method)
        expected_stage = stage_fingerprint(cfg, "train")
    if smoke:
        rows = rows[:min(4, len(rows))]
    report = read_json(directory / "token_reconciliation.json")
    summary = read_json(directory / "training_summary.json")
    run = read_json(directory / "run_config.json")
    tokens = sum(r["supervised_tokens"] for r in rows)
    expected = {"method": method, "synthetic": False, "valid_controlled_run": not smoke,
                "training_smoke_only": smoke, "stage_fingerprint": expected_stage,
                "selection_sha256": digest(selection), "parent_unchanged": True,
                "parent_fingerprint": cfg["parent"]["adapter_sha256"], "continuation_seed": cfg["training"]["seed"],
                "actual_tokens": tokens, "expected_tokens": tokens}
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise PipelineError(f"{method} run is not a matching reconciled {'smoke' if smoke else 'full arm'}: {directory}")
    for name, key in (("run_config.json", "run_config_sha256"), ("training_summary.json", "training_summary_sha256")):
        verified(directory / name, report.get(key))
    seen = summary.get("seen_ids", [])
    groups = summary.get("seen_groups", [])
    expected_manifest = (sha(e.root(cfg) / "training_inputs/E/smoke/manifest.jsonl")
                         if smoke else selection["manifest_sha256"])
    if (summary.get("complete") is not True or summary.get("processed_tokens") != tokens
            or len(seen) != len(rows) or set(seen) != {r["row_id"] for r in rows}
            or len(groups) != len(rows) or set(groups) != {r["problem_group_id"] for r in rows}
            or run.get("seed") != cfg["training"]["seed"] or run.get("arm") != method
            or run.get("mode") != "continuation" or run.get("epochs") != 1
            or run.get("optimizer_initialization") != "fresh_from_verified_parent_weights"
            or run.get("parent_adapter_sha256") != cfg["parent"]["adapter_sha256"]
            or run.get("manifest_sha256") != expected_manifest
            or run.get("code_sha256") != expected_stage):
        raise PipelineError(f"{method} training exposure or parent/recipe mismatch: {directory}")
    for field in ("learning_rate", "accumulation", "warmup_updates", "microbatch_size", "packing"):
        if run.get(field) != cfg["training"][field]:
            raise PipelineError(f"{method} continuation recipe differs: {field}")
    checkpoint = directory / "final/adapter/adapter_model.safetensors"
    verified(checkpoint, report.get("checkpoint_sha256"))
    if checkpoint.resolve() == (Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors").resolve():
        raise PipelineError("Continuation checkpoint must be separate from the original parent")
    if method == "E":
        provenance = read_json(directory / "e_provenance.json")
        if (provenance.get("stage_fingerprint") != expected_stage
                or provenance.get("selection_sha256") != digest(selection)
                or provenance.get("bindings") != e.binding(cfg)
                or provenance.get("reconciliation_sha256") != sha(directory / "token_reconciliation.json")):
            raise PipelineError("E run provenance is stale")
    return {"directory": str(directory), "report": report, "run_config": run,
            "checkpoint": str(checkpoint.parent), "selection": selection}


def require_smoke(cfg):
    key = e.fingerprint(cfg, "train", seedless=True)
    for path in (Path(cfg["output_root"]) / cfg["active_profile"]).glob("e-*/training_smoke/E/*/token_reconciliation.json"):
        report = read_json(path)
        if report.get("method") == "E" and report.get("smoke_fingerprint") == key:
            # A seed-42 smoke may validate a seed-137 run only under the same E selection/recipe.
            current = with_seed(cfg, report.get("continuation_seed"))
            validate_run(current, "E", path.parent, smoke=True)
            return
    raise PipelineError("Run a matching E train --smoke first; C0/C1 smoke evidence cannot unlock E")


@contextmanager
def trainer_bindings():
    """Single-process CLI adapter; all shared globals are restored even on exceptions."""
    original_loader = continuation.load_selection

    def load(cfg, method):
        return e.load_selection(cfg) if method == "E" else original_loader(cfg, method)

    with patch.object(continuation, "load_selection", load), \
            patch.object(continuation, "artifact_root", e.root), \
            patch.object(continuation, "stage_fingerprint", e.fingerprint), \
            patch.object(continuation, "seed_agnostic_fingerprint", lambda cfg: e.fingerprint(cfg, "train", seedless=True)):
        yield


def train(cfg, attempt, smoke=False):
    attempt_name(attempt)
    e.load_selection(cfg)
    if not smoke:
        require_smoke(cfg)
    with trainer_bindings():
        result = continuation.train(cfg, "E", attempt, smoke=smoke)
    directory = Path(result["run_directory"])
    write_json(directory / "e_provenance.json", {
        "stage_fingerprint": e.fingerprint(cfg, "train"), "bindings": e.binding(cfg),
        "generation_coverage": e.inputs(cfg)[3],
        "selection_sha256": result["selection_sha256"],
        "reconciliation_sha256": sha(directory / "token_reconciliation.json")})
    validate_run(cfg, "E", directory, smoke)
    return result


def preflight(cfg, seeds=(42, 137, 2026), c0_attempt="pilot01", c1_attempt="c1pilot01"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise PipelineError("Provide unique continuation seeds")
    settings = cfg["allocation"]
    if (cfg["data"].get("universe_mode") != "explicit" or cfg["data"].get("candidate_count") != 6409
            or settings.get("selector") != "method-b-cpsat-v3"
            or settings.get("expected_replay_tokens") != 318845 or settings.get("budget_kind") != "replay"):
        raise PipelineError("E requires the completed C1 shared 6,409-row bank and verified 318,845-token replay recipe")
    gate(cfg, "select")
    verified(Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors", cfg["parent"]["adapter_sha256"])
    rows, _, outcomes, _ = e.inputs(cfg)
    e.controls(cfg, rows)
    try:
        solver_version = importlib.metadata.version("ortools")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PipelineError("E selection requires OR-Tools 9.14.6206 in the existing SOL environment") from exc
    if solver_version != e.b_select.ORTOOLS_VERSION:
        raise PipelineError(f"Expected OR-Tools {e.b_select.ORTOOLS_VERSION}, found {solver_version}")
    for seed in seeds:
        current = with_seed(cfg, seed)
        for method, attempt in (("C0", c0_attempt), ("C1", c1_attempt)):
            directory = e.upstream_root(current) / "runs" / method / attempt_name(attempt)
            validate_run(current, method, directory)
    return {"ready_for_E_selection": True, "model": cfg["active_profile"], "parent_seed": cfg["parent"]["training_seed"],
            "continuation_seeds": list(seeds), "candidate_rows": len(rows),
            "confirmed_mistakes": sum(o["mistake"] for o in outcomes), "bindings": e.binding(cfg),
            "gpu_execution_verified": False}

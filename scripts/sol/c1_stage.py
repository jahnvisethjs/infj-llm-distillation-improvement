"""C1 batch gates and handoffs; reuse the scientific pipeline without changing C0 identities."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from capstone_person4.bindings import verified
from capstone_person4.config import artifact_root, load_config, stage_fingerprint
from capstone_person4.continuation import load_selection, select, train
from capstone_person4.data import load_prepared
from capstone_person4.handoff import finalize
from capstone_person4.provenance import PipelineError, canonical, digest, read_json, sha, write_json
from capstone_person4.reference_scoring import load_scores
from capstone_person4_approachC1 import generation_scoring as gs


def attempt_name(value):
    if not value or Path(value).name != value or value in (".", "..") or "\\" in value:
        raise PipelineError("Use a simple attempt name, not a path")
    return value


def c0_run(cfg, attempt):
    """Read-only prerequisite check. Final handoff performs the complete arm reconciliation."""
    root = artifact_root(cfg) / "runs/C0" / attempt_name(attempt)
    report = read_json(root / "token_reconciliation.json")
    _, selection = load_selection(cfg, "C0")
    expected = {"method": "C0", "valid_controlled_run": True, "training_smoke_only": False,
                "synthetic": False, "parent_unchanged": True,
                "stage_fingerprint": stage_fingerprint(cfg, "train"),
                "parent_fingerprint": cfg["parent"]["adapter_sha256"],
                "selection_sha256": digest(selection), "continuation_seed": cfg["training"]["seed"]}
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise PipelineError(f"Matching completed C0 run required: {root}")
    for name, key in (("run_config.json", "run_config_sha256"),
                      ("training_summary.json", "training_summary_sha256")):
        verified(root / name, report[key])
    verified(Path(report["checkpoint"]) / "adapter/adapter_model.safetensors", report["checkpoint_sha256"])
    summary = read_json(root / "training_summary.json")
    config = read_json(root / "run_config.json")
    if (summary.get("complete") is not True or summary.get("processed_tokens") != selection["actual_tokens"]
            or config.get("seed") != cfg["training"]["seed"] or config.get("arm") != "C0"):
        raise PipelineError(f"C0 training evidence does not reconcile: {root}")
    return root


def gate_bindings(cfg):
    _, preparation = load_prepared(cfg)
    _, scores = load_scores(cfg)
    _, generation = gs.load_generation(cfg)
    _, c0 = load_selection(cfg, "C0")
    return {"candidate_sha256": preparation["candidate_sha256"], "measurement_key": scores["measurement_key"],
            "generation_key": generation["generation_key"], "outcomes_sha256": generation["outcomes_sha256"],
            "c0_selection_sha256": digest(c0), "mistake_bonus": str(gs.mistake_bonus(cfg)),
            "stage_fingerprint": stage_fingerprint(cfg, "select"),
            "generation_code_sha256": sha(gs.__file__), "batch_code_sha256": sha(__file__)}


def gate_path(cfg):
    return artifact_root(cfg, "select") / "selection/C1/batch_gate.json"


def require_gate(cfg):
    record = read_json(gate_path(cfg))
    _, selection = load_selection(cfg, "C1")
    if record["bindings"] != gate_bindings(cfg) or record["selection_sha256"] != digest(selection):
        raise PipelineError("C1 batch diagnostic gate is stale; rerun C1 selection")
    diagnostics = record["diagnostics"]
    if diagnostics.get("proceed_to_training") is not True or any(diagnostics["degeneracy_flags"].values()):
        raise PipelineError("C1 diagnostic gate failed; training is blocked")
    return record


def select_c1(cfg):
    rows, _ = load_prepared(cfg)
    scores, _ = load_scores(cfg)
    diagnostics = gs.correctness_diagnostics(cfg, rows, scores)
    bindings = gate_bindings(cfg)
    diagnostic_path = artifact_root(cfg, "select") / "diagnostics" / (digest(bindings)[:16] + "_c1_batch.json")
    write_json(diagnostic_path, {"bindings": bindings, "diagnostics": diagnostics})
    if diagnostics["proceed_to_training"] is not True or any(diagnostics["degeneracy_flags"].values()):
        raise PipelineError(f"C1 diagnostic gate failed; evidence preserved at {diagnostic_path}")
    result = select(cfg, "C1")
    write_json(gate_path(cfg), {"bindings": bindings, "diagnostics": diagnostics,
                              "selection_sha256": digest(result)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "select", "train", "finalize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("0.5B", "1.5B"))
    parser.add_argument("--c0-attempt", default="pilot01")
    parser.add_argument("--attempt", default="c1pilot01")
    parser.add_argument("--smoke-attempt", default="c1smoke01")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    parser.add_argument("--predictions")
    parser.add_argument("--report")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        attempt_name(args.attempt)
        attempt_name(args.c0_attempt)
        attempt_name(args.smoke_attempt)
        cfg = load_config(args.config, args.profile, args.training_seed)
        if cfg["data"].get("universe_mode") != "explicit" or cfg["data"].get("candidate_count") != 6409:
            raise PipelineError("C1 batch requires the explicit shared 6,409-row Method B bank")
        if args.action == "preflight":
            if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
                raise PipelineError("Continuation seeds must be unique nonnegative integers")
            for path in (args.predictions, args.report):
                if not path or not Path(path).is_file():
                    raise PipelineError("Supply existing Person 3 predictions and run config files")
            load_prepared(cfg)
            load_scores(cfg)
            load_selection(cfg, "B")
            gs.mistake_bonus(cfg)
            for seed in args.seeds:
                c0_run(load_config(args.config, args.profile, seed), args.c0_attempt)
            # Request export is deliberately separate; never submit an ingest job without one.
            _, preparation = load_prepared(cfg)
            request = read_json(gs.generation_root(cfg) / "request" / gs.request_name(cfg) / "request.json")
            if request.get("generation_key") != gs.generation_key(cfg, preparation):
                raise PipelineError("Export a generation request for this parent and bank first")
            result = {"ready_for_submission": True, "training_seeds": args.seeds,
                      "generation_validation": "ingest job validates complete predictions before downstream jobs"}
        elif args.action == "select":
            result = select_c1(cfg)
        elif args.action == "train":
            require_gate(cfg)
            c0_run(cfg, args.c0_attempt)
            result = train(cfg, "C1", args.attempt, smoke=args.smoke)
        else:
            result = []
            seeds = [args.training_seed] if args.training_seed is not None else args.seeds
            for seed in seeds:
                current = load_config(args.config, args.profile, seed)
                require_gate(current)
                c0 = c0_run(current, args.c0_attempt)
                c1 = artifact_root(current) / "runs/C1" / args.attempt
                if not (c1 / "token_reconciliation.json").is_file():
                    raise PipelineError(f"Completed C1 run required; cannot issue a C0-only handoff: {c1}")
                result.append(finalize(current, None, c0, c1))
        print(canonical(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

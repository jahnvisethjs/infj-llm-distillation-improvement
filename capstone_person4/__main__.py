"""Run explicit, gated stages with python -m capstone_person4."""

import argparse
from pathlib import Path
import sys

from .provenance import PipelineError, canonical


def dry_run(output):
    from .allocation import allocate
    from .provenance import write_json, write_rows
    from .selection import fit_priorities, random_priorities
    rows = [{"row_id": f"synthetic-{i}", "problem_group_id": f"synthetic-group-{i}",
             "dataset_id": "orca" if i < 12 else "numina", "length_bin": i % 2,
             "problem": f"SYNTHETIC fixture {i}", "completion": "SYNTHETIC reference",
             "supervised_tokens": 10, "synthetic": True} for i in range(24)]
    scores = [{"row_id": r["row_id"], "mean_nll": (i % 6 + 1) / 10, "supervised_tokens": 10,
               "status": "ok", "synthetic": True} for i, r in enumerate(rows)]
    proportions = {"orca|0": 1, "orca|1": 1, "numina|0": 1, "numina|1": 1}
    result = {"synthetic": True, "warning": "MOCK data/scores; not model measurements or training", "methods": {}}
    for method, priorities in (("B", random_priorities(rows, 42)), ("C0", fit_priorities(rows, scores, 42))):
        selected, report = allocate(priorities, 80, proportions)
        selected = [{**r, "synthetic": True} for r in selected]
        write_rows(Path(output) / method / "synthetic_manifest.jsonl", selected)
        result["methods"][method] = {**report, "synthetic": True, "selected_ids": [r["row_id"] for r in selected]}
    write_json(Path(output) / "SYNTHETIC_DRY_RUN.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run", help="model-independent, visibly synthetic allocator demonstration")
    dry.add_argument("--output", required=True)
    for name in ("audit", "protect", "prepare", "score-reference", "diagnose", "select", "train", "finalize", "evaluate", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--profile", choices=("0.5B", "1.5B"))
        if name == "audit":
            command.add_argument("--target", choices=("inputs", "approach", "evaluation", "all"), default="approach")
        if name == "finalize":
            command.add_argument("--b-run", help="optional: omit when B is trained outside this pipeline")
            command.add_argument("--c0-run", required=True)
        if name in ("select", "train", "evaluate"):
            command.add_argument("--method", required=True, choices=("parent", "B", "C0") if name == "evaluate" else ("B", "C0"))
        if name in ("score-reference", "train"):
            command.add_argument("--smoke", action="store_true")
        if name in ("train", "evaluate", "compare"):
            command.add_argument("--attempt", required=True)
        if name in ("train", "finalize", "evaluate", "compare"):
            command.add_argument("--training-seed", type=int, help="continuation seed replication (42, 137, 2026); selection stays fixed")
        if name == "train":
            command.add_argument("--resume", help="this arm's own periodic or paused checkpoint, never the parent optimizer")
        if name == "evaluate":
            command.add_argument("--run")
            command.add_argument("--split", choices=("development", "final"), default="development")
        if name == "compare":
            command.add_argument("--reports", nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            result = dry_run(args.output)
        else:
            from .config import load_config
            cfg = load_config(args.config, args.profile, getattr(args, "training_seed", None))
            if args.command == "audit":
                from .audit import audit
                result = audit(cfg, write=True, target=args.target)
                print(canonical({"report_directory": result["report_directory"], "stages": result["stages"],
                                 "checks": [{"check": c["check"], "status": c["status"], "remedy": c["remedy"]} for c in result["checks"]]}))
                return 0 if result["ready"] else 2
            if args.command == "protect":
                from .protection import protect
                result = protect(cfg)
            elif args.command == "finalize":
                from .handoff import finalize
                result = finalize(cfg, args.b_run, args.c0_run)
            elif args.command == "prepare":
                from .data import prepare
                result = prepare(cfg)
            elif args.command in ("score-reference", "diagnose"):
                from .reference_scoring import diagnose, score_reference
                result = diagnose(cfg) if args.command == "diagnose" else score_reference(cfg, args.smoke)
            elif args.command in ("select", "train"):
                from .continuation import select, train
                result = select(cfg, args.method) if args.command == "select" else train(cfg, args.method, args.attempt, args.resume, args.smoke)
            elif args.command in ("evaluate", "compare"):
                from .evaluation import compare, evaluate
                result = evaluate(cfg, args.method, args.attempt, args.run, args.split) if args.command == "evaluate" else compare(cfg, args.reports, args.attempt)
        print(canonical(result))
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError, SyntaxError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit stages; no network/model work occurs during inspect/opportunity."""

import argparse
import json
import subprocess
import sys

from .common import RoutingError, destination, load_config


def parser():
    ap = argparse.ArgumentParser(description="Question-only routing, matched checkpoints, seed controls, and live answering")
    commands = ap.add_subparsers(dest="command", required=True)
    arguments = {
        "inspect": ("config",), "opportunity": ("config", "output"),
        "patterns": ("opportunity", "output"), "prepare-pool": ("config", "registry", "output"),
        "label": ("config", "registry", "pool", "split", "output"),
        "embed": ("pool", "output"), "train": ("registry", "pool", "embeddings", "train_labels", "validation_labels", "output"),
        "freeze": ("training", "output"), "evaluate-test": ("frozen", "pool", "embeddings", "test_labels", "output"),
        "report": ("opportunity", "output"), "route": ("frozen", "question"),
        "answer": ("config", "frozen", "question")}
    for command, names in arguments.items():
        sub = commands.add_parser(command)
        for name in names:
            sub.add_argument("--" + name.replace("_", "-"), required=True,
                             **({"choices": ("train", "validation", "test")} if name == "split" else {}))
        if command in ("label", "answer"):
            sub.add_argument("--authorize-gpu", action="store_true")
        if command == "label":
            sub.add_argument("--frozen")
            sub.add_argument("--import-index")
            sub.add_argument("--reuse-labels", help="Reuse sealed chunks from an earlier interrupted label attempt")
        if command == "prepare-pool":
            sub.add_argument("--download", action="store_true")
        if command == "report":
            sub.add_argument("--test")
        if command in ("route", "answer"):
            sub.add_argument("--experiment", default="primary_p17_t42")
    return ap


def run(args):
    cfg = load_config(args.config) if hasattr(args, "config") else None
    if hasattr(args, "output"):
        inputs = [getattr(args, k, None) for k in ("registry", "pool", "embeddings", "train_labels", "validation_labels",
                                                   "training", "frozen", "test_labels", "opportunity", "test", "reuse_labels")]
        if cfg:
            def paths(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k in ("path", "parser_path", "checkpoint", "a", "b", "c1", "run_config", "training_summary",
                                 "parent_checkpoint", "parent_run_config", "predictions", "config", "receipt") and isinstance(v, str):
                            yield v
                        else:
                            yield from paths(v)
                elif isinstance(value, list):
                    for item in value:
                        yield from paths(item)
            inputs.extend(paths(cfg))
        destination(args.output, inputs)
    if args.command == "inspect":
        from .inputs import inspect
        return inspect(cfg)
    if args.command == "opportunity":
        from .analysis import opportunity
        return opportunity(cfg, args.output)
    if args.command == "patterns":
        from .analysis import patterns
        return patterns(args.opportunity, args.output)
    if args.command == "prepare-pool":
        from .pool import prepare_pool
        return prepare_pool(cfg, args.registry, args.output, args.download)
    if args.command == "label":
        from .labeling import label
        return label(cfg, args.registry, args.pool, args.split, args.output, args.authorize_gpu, args.frozen, args.import_index, args.reuse_labels)
    if args.command == "embed":
        from .router import embed
        return embed(args.pool, args.output)
    if args.command == "train":
        from .router import train
        return train(args.registry, args.pool, args.embeddings, args.train_labels, args.validation_labels, args.output)
    if args.command == "freeze":
        from .router import freeze
        return freeze(args.training, args.output)
    if args.command == "evaluate-test":
        from .evaluation import evaluate_test
        return evaluate_test(args.frozen, args.pool, args.embeddings, args.test_labels, args.output)
    if args.command == "report":
        from .evaluation import report
        return report(args.opportunity, args.output, args.test)
    if args.command == "route":
        from .router import route
        return route(args.question, args.frozen, args.experiment)
    if args.command == "answer":
        from .live import answer
        return answer(args.question, args.frozen, cfg["evaluator"], args.authorize_gpu, args.experiment)
    raise RoutingError("Unknown stage")


def main():
    try:
        value = run(parser().parse_args())
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc), "complete": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

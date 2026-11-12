"""E-only stages; the original C0/C1 CLI and source fingerprints stay unchanged."""

import argparse
import sys

from capstone_person4.config import load_config
from capstone_person4.provenance import PipelineError, canonical
from . import selection, runtime, handoff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "select", "train", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, help="the unchanged SOL config used for C1")
        command.add_argument("--profile", choices=("0.5B", "1.5B"))
        if name in ("preflight", "finalize"):
            command.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
            command.add_argument("--c0-attempt", default="pilot01")
            command.add_argument("--c1-attempt", default="c1pilot01")
        if name in ("train", "finalize"):
            command.add_argument("--attempt", default="epilot01")
        if name == "train":
            command.add_argument("--training-seed", type=int, default=42)
            command.add_argument("--smoke", action="store_true")
    package = commands.add_parser("package")
    package.add_argument("--run-index", nargs="+", required=True, help="one or more finalized E CSV indexes")
    package.add_argument("--output", required=True, help="new destination directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            result = handoff.package(args.run_index, args.output)
        else:
            cfg = load_config(args.config, args.profile, getattr(args, "training_seed", None))
            if args.command == "preflight":
                result = runtime.preflight(cfg, args.seeds, args.c0_attempt, args.c1_attempt)
            elif args.command == "select":
                result = selection.select(cfg)
            elif args.command == "train":
                result = runtime.train(cfg, args.attempt, args.smoke)
            else:
                result = handoff.finalize(cfg, args.attempt, args.seeds, args.c0_attempt, args.c1_attempt)
        print(canonical(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

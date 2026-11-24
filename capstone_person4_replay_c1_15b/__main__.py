"""Sol replay/C1 stages. Configuration/preflight do not submit or run GPU jobs."""

import argparse
from pathlib import Path
import subprocess
import sys

from . import METHOD
from .common import SEEDS, PilotError, canonical, load_config


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    configure = commands.add_parser("configure", help="write an explicit Sol-local configuration; never submit")
    configure.add_argument("--config-output", required=True)
    configure.add_argument("--parent-seed", type=int, choices=(17, 123), required=True)
    configure.add_argument("--work", default="/scratch/sshind38/capstone_c0_c1_work")
    configure.add_argument("--b-root", default="/scratch/sshind38/capstone_approach_b")
    configure.add_argument("--native-runner")
    configure.add_argument("--a-eval-root", default="/scratch/msures15/eval_results")
    configure.add_argument("--a-policy", help="original A token policy, if bounded hash discovery cannot find it")
    configure.add_argument("--output-root")
    configure.add_argument("--account", default="class_fse57081354fall2026")
    configure.add_argument("--partition", default="public")
    configure.add_argument("--qos", default="class")
    for name in ("preflight", "select", "_solve", "train", "evaluate", "finalize", "submit"):
        command = commands.add_parser(name, help="internal isolated solver stage" if name == "_solve" else None)
        command.add_argument("--config", required=True)
        if name in ("train", "evaluate", "finalize", "submit"):
            command.add_argument("--attempt", default="attempt01")
        if name in ("evaluate", "finalize", "submit"):
            command.add_argument("--evaluation-attempt", default="eval01")
        if name in ("train", "evaluate", "submit"):
            command.add_argument("--authorize-gpu", action="store_true", help="explicitly authorize this GPU action")
        if name == "train":
            command.add_argument("--training-seed", choices=SEEDS, type=int, default=42)
            command.add_argument("--smoke", action="store_true")
        if name == "evaluate":
            command.add_argument("--methods", nargs="+", choices=("A", "C1", METHOD), default=["A", "C1", METHOD])
            command.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "configure":
            from .sol import configure
            value = configure(args)
        else:
            cfg = load_config(args.config)
            from . import inputs
            if args.command in ("train", "evaluate", "submit") and not args.authorize_gpu:
                raise PilotError("GPU execution requires an explicit --authorize-gpu; implementation is not execution approval")
            if args.command == "preflight":
                value = inputs.preflight(cfg)
            elif args.command == "select":
                inputs.preflight(cfg)
                subprocess.run([cfg["solver_python"], "-B", "-m", __package__, "_solve", "--config", cfg["config_path"]],
                               cwd=Path(__file__).resolve().parents[1], check=True)
                from .selection import load_selection
                bundle = inputs.load(cfg)
                value = load_selection(bundle)[1]
            else:
                if args.command in ("train", "evaluate", "submit"):
                    if args.command == "submit":
                        inputs.preflight(cfg)
                    else:
                        inputs.check_environment(cfg)
                bundle = inputs.load(cfg)
                if args.command == "_solve":
                    from .selection import select
                    value = select(bundle)
                elif args.command == "train":
                    from .runtime import train
                    value = train(bundle, args.training_seed, args.attempt, args.smoke)
                elif args.command == "evaluate":
                    from .evaluation import evaluate
                    value = evaluate(bundle, args.methods, args.seeds, args.attempt, args.evaluation_attempt)
                elif args.command == "finalize":
                    from .evaluation import finalize
                    value = finalize(bundle, args.attempt, args.evaluation_attempt)
                else:
                    from .sol import submit
                    value = submit(bundle, args.attempt, args.evaluation_attempt)
        print(canonical(value), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

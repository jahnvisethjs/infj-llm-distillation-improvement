"""Inspect or recover parent17 evaluation without editing the v1 training package."""

import argparse
import json
import subprocess
import sys

from . import PARENT17_IDENTITY
from . import recovery


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run", "submit"):
        command = commands.add_parser(name)
        command.add_argument("--repo", default="/scratch/jseth2/replay_c1_15b_code_v1")
        command.add_argument("--config", default="/scratch/jseth2/replay15b_parent17_config.json")
        command.add_argument("--expected-identity", default=PARENT17_IDENTITY)
        command.add_argument("--training-attempt", default="attempt01")
        command.add_argument("--source-evaluation-attempt", default="eval01")
        command.add_argument("--evaluation-attempt", default="eval02_recovery")
        if name == "inspect":
            command.add_argument("--summary-only", action="store_true")
        else:
            command.add_argument("--authorize-gpu", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        bundle, common, evaluation = recovery.load_original(args)
        if args.command == "inspect":
            _, result = recovery.inspect_sources(bundle, common, evaluation, args)
            status = 0 if result["ready"] else 2
            if args.summary_only:
                result = {key: value for key, value in result.items() if key != "actions"}
        elif args.command == "submit":
            result = recovery.submit(bundle, common, evaluation, args)
            status = 0
        else:
            result = recovery.run(bundle, common, evaluation, args)
            status = 0
        print(json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")), flush=True)
        return status
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

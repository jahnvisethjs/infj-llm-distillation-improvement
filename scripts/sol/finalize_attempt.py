"""Finalize the C0 attempt (and a same-named B attempt when it exists) using actual stage paths."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from capstone_person4.config import artifact_root, load_config
from capstone_person4.handoff import finalize
from capstone_person4.provenance import PipelineError, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--profile", choices=("0.5B", "1.5B"))
    parser.add_argument("--training-seed", type=int)
    args = parser.parse_args()
    try:
        if not args.attempt or Path(args.attempt).name != args.attempt or args.attempt in (".", ".."):
            raise PipelineError("Use a simple training attempt name")
        cfg = load_config(args.config, args.profile, args.training_seed)
        root = artifact_root(cfg) / "runs"
        b_run = root / "B" / args.attempt
        c1_run = root / "C1" / args.attempt
        print(canonical(finalize(cfg, b_run if b_run.is_dir() else None, root / "C0" / args.attempt,
                                 c1_run if c1_run.is_dir() else None)))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

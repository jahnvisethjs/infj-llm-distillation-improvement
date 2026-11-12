"""Check every ten minutes; submit the 1.5B/parent123 E chain once below four jobs."""

import os
from pathlib import Path
import subprocess
import time


def main():
    repo = Path(os.environ.get("C0_REPO", Path(__file__).resolve().parents[2]))
    command = ["bash", "scripts/sol/submit_e_pipeline.sh",
               "configs/c0.local.sol.1.5b.seed123.json", "q15-p123-e",
               "pilot01", "c1pilot01", "epilot01", "esmoke01"]
    if not (repo / command[1]).is_file() or not (repo / command[2]).is_file():
        raise SystemExit("Cannot find the submitter/config; set C0_REPO to your SOL checkout.")
    if not (os.environ.get("C0_PYTHON") or os.environ.get("C0_ENV")):
        raise SystemExit("Keep C0_PYTHON or C0_ENV exported, along with your working HF cache settings.")
    while True:
        print("Waiting 10 minutes before checking the queue...", flush=True)
        time.sleep(600)
        result = subprocess.run(
            ["squeue", "-h", "-r", "-u", "jseth2", "-o", "%j"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        jobs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        print(f"Active jobs: {len(jobs)}", flush=True)
        if any(name.startswith("q15-p123-e-") for name in jobs):
            print("This E chain is already queued/running; no submission made.", flush=True)
            return 0
        if len(jobs) < 4:
            print("Below four jobs; submitting the E chain once.", flush=True)
            # Inherit the same Python, cache and scheduler exports as successful E runs.
            env = {**os.environ, "C0_REPO": str(repo.resolve())}
            result = subprocess.run(command, cwd=repo, env=env)
            if result.returncode:
                print("Submission failed or was partial. Inspect squeue before retrying.", flush=True)
            return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"Stopped: {exc}", flush=True)
        if getattr(exc, "stderr", None):
            print(exc.stderr, flush=True)
        raise SystemExit(1)

"""Explicit Sol configuration and a bounded afterok chain; never auto-install or cancel."""

import getpass
from pathlib import Path
import re
import subprocess
import sys

from . import VERSION
from .common import PARENTS, SEEDS, PilotError, attempt_name, canonical, read, write


def configure(args):
    work = Path(args.work).resolve()
    b_root = Path(args.b_root).resolve()
    cfg = {"version": VERSION, "model": "1.5B", "parent_seed": args.parent_seed,
           "selection_seed": 42, "continuation_seeds": list(SEEDS),
           "work": str(work), "b_root": str(b_root),
           "native_runner": str(Path(args.native_runner).resolve() if args.native_runner else
                                work / "C0_C1_Overnight_Training_20261110/nightly.py"),
           "c1_profile": str(work / f"outputs/C0_C1_overnight_training_v1/7376bbd91d80e726/profiles/1.5B_seed{args.parent_seed}" / PARENTS[args.parent_seed]["native_identity_prefix"]),
           "training_python": str(b_root / "venv_b_training_py312/bin/python"),
           "solver_python": str(b_root / "venv_b_solver_v3/bin/python"),
           "a_eval_root": str(Path(args.a_eval_root).resolve()),
           "a_policy": str(Path(args.a_policy).resolve()) if args.a_policy else None,
           "c1_eval_root": str(work / "outputs/C0_C1_development_evaluation_v1/931e8fb821c7d70f"),
           "output_root": str(Path(args.output_root or f"/scratch/{getpass.getuser()}/replay_c1_15b_4955_pilot").resolve()),
           "slurm": {"account": args.account, "partition": args.partition, "qos": args.qos}}
    write(args.config_output, cfg)
    return {"configuration": str(Path(args.config_output).resolve()), "jobs_submitted": False}


def stages(attempt, evaluation_attempt):
    common = ["--attempt", attempt]
    return [("select", ["select"], False),
            ("smoke", ["train", "--smoke", "--training-seed", "42", *common, "--authorize-gpu"], True),
            *[(f"train{seed}", ["train", "--training-seed", str(seed), *common, "--authorize-gpu"], True) for seed in SEEDS],
            ("evaluate", ["evaluate", *common, "--evaluation-attempt", evaluation_attempt, "--authorize-gpu"], True),
            ("finalize", ["finalize", *common, "--evaluation-attempt", evaluation_attempt], False)]


def submit(bundle, attempt, evaluation_attempt):
    attempt_name(attempt)
    attempt_name(evaluation_attempt)
    root = bundle.root / "submissions" / f"{attempt}_{evaluation_attempt}"
    if root.exists():
        records = [read(p) for p in sorted(root.glob("job_*.json"))]
        ids = [r["job_id"] for r in records]
        if ids:
            active = subprocess.run(["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i"],
                                    capture_output=True, text=True, check=True).stdout.strip()
            if active:
                raise PilotError("Existing active submission preserved: " + active.replace("\n", ", "))
        if (root / "submission.json").is_file():
            return {"reused_submission": True, **read(root / "submission.json")}
        raise PilotError("Interrupted submission preserved; inspect its recorded jobs before choosing new attempt names")
    root.mkdir(parents=True, exist_ok=False)  # Atomic exclusion of competing submissions.
    repo = Path(__file__).resolve().parents[1]
    worker = repo / "scripts/sol/replay15b_worker.sh"
    dependencies = []
    records = []
    settings = bundle.cfg.get("slurm", {})
    for key in ("account", "partition", "qos"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", settings.get(key, "")):
            raise PilotError("Set a valid Slurm " + key)
    for number, (name, arguments, gpu) in enumerate(stages(attempt, evaluation_attempt)):
        command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=32G",
                   "--time=04:00:00", f"--job-name=replay15b-p{bundle.cfg['parent_seed']}-" + name,
                   "--account=" + settings["account"], "--partition=" + settings["partition"],
                   "--qos=" + settings["qos"], "--chdir=" + str(repo),
                   "--output=" + str(root / (name + ".%j.out")), "--error=" + str(root / (name + ".%j.err"))]
        if gpu:
            command.append("--gpus=a100:1")
        if dependencies:
            command.append("--dependency=afterok:" + dependencies[-1])
        command += [str(worker), bundle.cfg["training_python"], str(repo), bundle.cfg["config_path"], *arguments]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise PilotError("Unrecognized sbatch response; inspect scheduler before retrying: " + result.stdout)
        record = {"stage": name, "job_id": job_id, "gpu": gpu, "command": command}
        write(root / f"job_{number:02d}_{name}.json", record)
        records.append(record)
        dependencies.append(job_id)
    report = {"identity": bundle.identity, "model": bundle.cfg["model"], "parent_seed": bundle.cfg["parent_seed"], "jobs": records, "submission_directory": str(root),
              "failure_policy": "afterok stops dependent jobs; preserve all attempts; no automatic retries/cancellations"}
    write(root / "submission.json", report)
    return report

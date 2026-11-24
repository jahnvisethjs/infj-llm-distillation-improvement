"""Preserve v1 training identity and recover evaluation in a new attempt directory."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

from . import PARENT17_IDENTITY, VERSION
from .frozen import FROZEN_CODE
from .validation import ALIASES, CheckerExecutionError, parser_status_support


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_original(repo):
    repo = Path(repo).resolve()
    files = list((repo / "capstone_person4_replay_c1_15b").glob("*.py"))
    files += list((repo / "scripts/sol").glob("replay15b_*"))
    actual = {p.relative_to(repo).as_posix(): file_sha(p) for p in files if p.is_file()}
    if actual != FROZEN_CODE:
        raise ValueError("Original 1.5B v1 code changed; preserve the training deployment and restore its exact files")
    return repo


def tool_code():
    root = Path(__file__).resolve().parents[1]
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((root / "scripts/sol").glob("recover_replay15b_*.sh"))
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(files)}


def load_original(args):
    repo = verify_original(args.repo)
    sys.path.insert(0, str(repo))
    package = importlib.import_module("capstone_person4_replay_c1_15b")
    if Path(package.__file__).resolve() != repo / "capstone_person4_replay_c1_15b/__init__.py":
        raise ValueError("Another copy of the original package was already imported")
    common = importlib.import_module(package.__name__ + ".common")
    # The pinned native context must establish upstream imports before any
    # capstone_person4 modules are imported by recovery validation/reporting.
    inputs = importlib.import_module(package.__name__ + ".inputs")
    cfg = common.load_config(args.config)
    if cfg["parent_seed"] != 17:
        raise ValueError("This recovery is restricted to parent17; preserve the completed parent123 report")
    inputs.check_environment(cfg)
    bundle = inputs.load(cfg)
    if args.expected_identity != PARENT17_IDENTITY or bundle.identity != args.expected_identity:
        raise ValueError("Recovery does not match the existing parent17 training identity")
    runtime = importlib.import_module(package.__name__ + ".runtime")
    evaluation = importlib.import_module(package.__name__ + ".evaluation")
    for seed in common.SEEDS:
        runtime.validate_run(bundle, runtime.run_directory(bundle, seed, args.training_attempt), seed)
    common.checked(bundle.context.runtime / "vendor/evaluation/evaluate.py", common.EVALUATOR_SHA)
    common.checked(bundle.context.runtime / "vendor/evaluation/answer_parser.py", common.PARSER_SHA)
    return bundle, common, evaluation


@dataclass
class Action:
    method: str
    seed: int
    benchmark: str
    expected: dict
    kind: str
    reason: str
    records: object = None
    config: object = None
    origin: object = None

    def public(self):
        value = {k: getattr(self, k) for k in ("method", "seed", "benchmark", "kind", "reason", "origin")}
        if self.records is not None:
            value["rows"] = len(self.records)
            value["parse_status_counts"] = dict(Counter(r.get("parse_status") for r in self.records))
        return value


def source_action(bundle, common, evaluation, method, seed, benchmark, training_attempt, source_attempt):
    expected = evaluation.target(bundle, method, seed, training_attempt)
    common.checked(Path(expected["checkpoint"]) / "adapter_model.safetensors", expected["checkpoint_sha256"])
    _, ids = evaluation.question_file(bundle, benchmark)
    source = evaluation.result_root(bundle, method, seed, benchmark, source_attempt)
    if (source / "completion.json").exists():
        # Existing canonical receipts are authoritative: corruption must stop,
        # rather than silently substituting a raw attempt or rerunning a model.
        origin = {"source": str(source), "source_kind": "verified_receipt", "reused": True,
                  "files": {str(source / name): common.sha(source / name)
                            for name in ("completion.json", "predictions.jsonl", "predictions.config.json")}}
        evaluation.validate_result(bundle, method, seed, benchmark, training_attempt, source_attempt)
        records, config = common.rows(source / "predictions.jsonl"), common.read(source / "predictions.config.json")
        checked_origins(common, origin)
        return Action(method, seed, benchmark, expected, "reuse", "validated canonical evaluation", records, config, origin)
    if method == "C1":
        records, config, native_origin = evaluation.native_c1_result(bundle, expected, benchmark, ids)
        source_file = Path(native_origin["source"])
        source_files = [source_file, source_file.parent / "completion.json", *source_file.parent.glob("*.config.json")]
        origin = {**native_origin, "source_kind": "native_C1",
                  "files": {str(p): common.sha(p) for p in source_files}}
        return Action(method, seed, benchmark, expected, "reuse", "verified native C1 evaluation", records, config,
                      origin)
    raw, sidecar = source / "native/raw.jsonl", source / "native/raw.config.json"
    origin = {"source": str(raw), "source_kind": "incomplete_raw", "reused": False,
              "files": {str(p): common.sha(p) for p in (raw, sidecar) if p.is_file()}}
    if not raw.is_file() or not sidecar.is_file():
        return Action(method, seed, benchmark, expected, "generate", "raw predictions or sidecar absent", origin=origin)
    config = common.read(sidecar)  # A malformed sidecar is an integrity stop.
    try:
        records = common.rows(raw)
    except json.JSONDecodeError:
        return Action(method, seed, benchmark, expected, "generate", "interrupted JSONL record", origin=origin)
    try:
        evaluation.validate_predictions(records, config, expected, benchmark, ids)
    except CheckerExecutionError as error:
        origin["checker_errors"] = error.records
        return Action(method, seed, benchmark, expected, "generate", "checker execution error; clean benchmark retry", origin=origin)
    except ValueError as error:
        # Only a strict subset of expected unique IDs is an interrupted run.
        # Wrong checkpoints, malformed outcomes, duplicates and unknown IDs stop.
        actual_ids = [r.get("problem_id") for r in records]
        if (str(error) == "Predictions must cover the exact 500 frozen questions once each"
                and len(actual_ids) < len(ids) and len(set(actual_ids)) == len(actual_ids)
                and set(actual_ids) <= ids):
            return Action(method, seed, benchmark, expected, "generate", "incomplete question coverage", origin=origin)
        raise
    origin.update(source_kind="validated_raw", reused=True)
    return Action(method, seed, benchmark, expected, "reuse", "complete raw predictions passed all checks", records, config, origin)


def inspect_sources(bundle, common, evaluation, args):
    common.attempt_name(args.training_attempt)
    common.attempt_name(args.source_evaluation_attempt)
    common.attempt_name(args.evaluation_attempt)
    if args.source_evaluation_attempt == args.evaluation_attempt or args.evaluation_attempt == "eval01":
        raise ValueError("Use a new recovery evaluation attempt; the original eval01 files are immutable")
    actions, blocked = [], []
    with parser_status_support(evaluation):
        for method in ("A", "C1", "REPLAY_C1_50_50"):
            for seed in common.SEEDS:
                for benchmark in common.QUESTIONS:
                    try:
                        actions.append(source_action(bundle, common, evaluation, method, seed, benchmark,
                                                     args.training_attempt, args.source_evaluation_attempt))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        blocked.append({"method": method, "seed": seed, "benchmark": benchmark, "error": str(error)})
    plan = {"version": VERSION, "identity": bundle.identity, "parent_seed": 17, "read_only": True,
            "ready": not blocked, "source_evaluation_attempt": args.source_evaluation_attempt,
            "evaluation_attempt": args.evaluation_attempt, "training_attempt": args.training_attempt,
            "reuse_benchmark_files": sum(a.kind == "reuse" for a in actions),
            "generate_benchmark_files": sum(a.kind == "generate" for a in actions),
            "actions": [a.public() for a in actions], "blocked": blocked,
            "training_restarted": False, "gpu_execution_performed": False}
    return actions, plan


def protocol(bundle, common, args):
    return {"version": VERSION, "identity": bundle.identity, "parent_seed": 17,
            "config_sha256": common.sha(args.config), "training_attempt": args.training_attempt,
            "source_evaluation_attempt": args.source_evaluation_attempt, "evaluation_attempt": args.evaluation_attempt,
            "original_code": FROZEN_CODE, "recovery_code": tool_code(),
            "parser_sha256": common.PARSER_SHA, "evaluator_sha256": common.EVALUATOR_SHA,
            "validation_aliases": ALIASES, "saved_records_modified": False,
            "checker_errors": "reject; generate one clean benchmark attempt, never count as wrong",
            "training_restarted": False}


def recovery_root(bundle, args):
    return bundle.root / "recovery" / args.evaluation_attempt


def checked_origins(common, origin):
    for path, expected in origin.get("files", {}).items():
        common.checked(path, expected)


def generate(bundle, common, evaluation, action, destination):
    question_path, ids = evaluation.question_file(bundle, action.benchmark)
    raw = destination / "native/raw.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=False)
    evaluator = common.checked(bundle.context.runtime / "vendor/evaluation/evaluate.py", common.EVALUATOR_SHA)
    checkpoint = Path(action.expected["checkpoint"]) / "adapter_model.safetensors"
    common.checked(checkpoint, action.expected["checkpoint_sha256"])
    command = [sys.executable, "-B", str(evaluator), "--eval_file", str(question_path),
               "--adapter", action.expected["checkpoint"], "--expect_train_manifest_sha256", action.expected["manifest_sha256"],
               "--run_id", f"recovery_1.5B_p17_{action.method}_s{action.seed}_{action.benchmark}", "--out", str(raw),
               "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024", "--question_text", "p1_clean"]
    subprocess.run(command, check=True)
    records, config = common.rows(raw), common.read(raw.with_suffix(".config.json"))
    evaluation.validate_predictions(records, config, action.expected, action.benchmark, ids)
    common.checked(checkpoint, action.expected["checkpoint_sha256"])
    return records, config, {"source": str(raw), "source_kind": "new_evaluation", "reused": False,
                             "files": {str(p): common.sha(p) for p in (raw, raw.with_suffix(".config.json"))},
                             "prior_attempt": action.origin}


def save_result(bundle, common, evaluation, action, records, config, origin, args, recovery_protocol):
    _, ids = evaluation.question_file(bundle, action.benchmark)
    evaluation.validate_predictions(records, config, action.expected, action.benchmark, ids)
    checked_origins(common, origin)
    dest = evaluation.result_root(bundle, action.method, action.seed, action.benchmark, args.evaluation_attempt)
    common.write_rows(dest / "predictions.jsonl", records)
    common.write(dest / "predictions.config.json", config)
    receipt = {"identity": bundle.identity, "target": action.expected, "benchmark": action.benchmark,
               "evaluation_attempt": args.evaluation_attempt, "complete": True, "origin": origin,
               "predictions_sha256": common.sha(dest / "predictions.jsonl"),
               "config_sha256": common.sha(dest / "predictions.config.json"), "reserved_final_tests_performed": False,
               "recovery_version": VERSION, "recovery_protocol_sha256": common.digest(recovery_protocol)}
    common.write(dest / "completion.json", receipt)
    evaluation.validate_result(bundle, action.method, action.seed, action.benchmark,
                               args.training_attempt, args.evaluation_attempt)
    return receipt


def verified_completion(bundle, common, evaluation, args, recovery_protocol):
    root = recovery_root(bundle, args)
    result = common.read(root / "completion.json")
    if (result.get("complete") is not True or result.get("identity") != bundle.identity
            or result.get("recovery_protocol_sha256") != common.digest(recovery_protocol)):
        raise ValueError("Recovery completion belongs to changed code, inputs or another attempt")
    for path, expected in result["reports"].items():
        common.checked(path, expected)
    with parser_status_support(evaluation):
        for method in ("A", "C1", "REPLAY_C1_50_50"):
            for seed in common.SEEDS:
                for benchmark in common.QUESTIONS:
                    _, receipt = evaluation.validate_result(bundle, method, seed, benchmark,
                                                             args.training_attempt, args.evaluation_attempt)
                    if receipt.get("recovery_protocol_sha256") != common.digest(recovery_protocol):
                        raise ValueError("Mixed recovery evaluation policies")
    return result


def run(bundle, common, evaluation, args):
    common.attempt_name(args.evaluation_attempt)
    recovery_protocol = protocol(bundle, common, args)
    root = recovery_root(bundle, args)
    if (root / "completion.json").exists():
        return verified_completion(bundle, common, evaluation, args, recovery_protocol)
    actions, plan = inspect_sources(bundle, common, evaluation, args)
    if not plan["ready"]:
        raise ValueError("Recovery blocked by invalid source records: " + common.canonical(plan["blocked"]))
    if plan["generate_benchmark_files"] and not args.authorize_gpu:
        raise ValueError("Missing/unusable evaluations require --authorize-gpu on a scheduled GPU node")
    common.write(root / "protocol.json", recovery_protocol)
    # A persistent claim prevents duplicate execution. Failures are preserved;
    # choose a fresh recovery attempt and optionally reuse this attempt as source.
    try:
        (root / "execution_claim").mkdir()
    except FileExistsError:
        raise ValueError("Recovery execution already exists; inspect logs and choose a new evaluation attempt") from None
    common.write(root / "plan.json", plan)
    receipts = []
    try:
        with parser_status_support(evaluation):
            for action in actions:
                dest = evaluation.result_root(bundle, action.method, action.seed, action.benchmark, args.evaluation_attempt)
                if dest.exists():
                    raise ValueError("Recovery destination already exists; preserve it and use a new evaluation attempt")
                checked_origins(common, action.origin)
                if action.kind == "generate":
                    records, config, origin = generate(bundle, common, evaluation, action, dest)
                else:
                    records, config, origin = action.records, action.config, action.origin
                receipts.append(save_result(bundle, common, evaluation, action, records, config, origin,
                                            args, recovery_protocol))
            report = evaluation.finalize(bundle, args.training_attempt, args.evaluation_attempt)
        report_dir = Path(report["report_directory"])
        common.write(report_dir / "recovery.json", {"protocol": recovery_protocol, "plan": plan,
                     "evaluation_receipts_sha256": common.digest(receipts), "training_restarted": False})
        verify_original(args.repo)
        result = {"complete": True, "version": VERSION, "identity": bundle.identity,
                  "report_directory": str(report_dir), "training_restarted": False,
                  "reused_benchmark_files": plan["reuse_benchmark_files"],
                  "generated_benchmark_files": plan["generate_benchmark_files"],
                  "recovery_protocol_sha256": common.digest(recovery_protocol),
                  "reports": {str(report_dir / name): common.sha(report_dir / name) for name in (
                      "accuracy.csv", "paired_differences.csv", "run_index.csv", "summary.json", "provenance.json", "recovery.json")}}
        common.write(root / "completion.json", result)
        return verified_completion(bundle, common, evaluation, args, recovery_protocol)
    except Exception as error:
        common.write(root / "failure.json", {"complete": False, "identity": bundle.identity,
                     "recovery_protocol_sha256": common.digest(recovery_protocol), "error": str(error),
                     "training_restarted": False, "completed_benchmark_files": len(receipts)})
        raise


def submit(bundle, common, evaluation, args):
    actions, plan = inspect_sources(bundle, common, evaluation, args)
    if not plan["ready"]:
        raise ValueError("Recovery blocked by invalid source records: " + common.canonical(plan["blocked"]))
    if not args.authorize_gpu:
        raise ValueError("Submitting the recovery GPU job requires --authorize-gpu")
    root = recovery_root(bundle, args)
    recovery_protocol = protocol(bundle, common, args)
    if (root / "completion.json").exists():
        return {**verified_completion(bundle, common, evaluation, args, recovery_protocol), "already_complete": True}
    if (root / "submission.json").exists():
        if common.read(root / "protocol.json") != recovery_protocol:
            raise ValueError("Existing recovery submission belongs to changed code or inputs")
        return {**common.read(root / "submission.json"), "reused_submission": True}
    settings = bundle.cfg.get("slurm", {})
    for key in ("account", "partition", "qos"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", settings.get(key, "")):
            raise ValueError("Set a valid Slurm " + key)
    common.write(root / "protocol.json", recovery_protocol)
    try:
        (root / "submission_claim").mkdir()
    except FileExistsError:
        raise ValueError("Interrupted submission preserved; inspect the scheduler before using a new recovery attempt") from None
    tool_root = Path(__file__).resolve().parents[1]
    command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=32G",
               "--time=04:00:00", "--gpus=a100:1", "--job-name=recover15b-p17-eval",
               "--account=" + settings["account"], "--partition=" + settings["partition"], "--qos=" + settings["qos"],
               "--chdir=" + str(tool_root), "--output=" + str(root / "recover.%j.out"),
               "--error=" + str(root / "recover.%j.err"), str(tool_root / "scripts/sol/recover_replay15b_worker.sh"),
               bundle.cfg["training_python"], str(tool_root), str(Path(args.repo).resolve()),
               str(Path(args.config).resolve()), args.training_attempt, args.source_evaluation_attempt,
               args.evaluation_attempt, args.expected_identity]
    common.write(root / "submission_request.json", {"command": command, "protocol": recovery_protocol, "plan": plan})
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Unrecognized sbatch response; inspect scheduler before retrying: " + result.stdout)
    receipt = {"job_id": job, "identity": bundle.identity, "command": command,
               "evaluation_attempt": args.evaluation_attempt, "training_restarted": False,
               "report_directory": str(bundle.root / "reports" / args.evaluation_attempt),
               "log_directory": str(root)}
    common.write(root / "submission.json", receipt)
    return receipt

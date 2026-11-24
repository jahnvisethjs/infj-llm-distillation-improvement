"""Common-runtime development evaluation and paired, seed-matched reporting."""

import csv
import gzip
import hashlib
import importlib.metadata
import io
from pathlib import Path
import random
import statistics
import subprocess
import sys

from . import METHOD, VERSION
from .common import (MODEL, REVISION, EVALUATOR_SHA, EVAL_PACKAGES, PARSER_SHA, PROMPT_SHA, QUESTIONS, SEEDS,
                     PilotError, checked, digest, read, rows, sha, write, write_bytes, write_rows)
from .runtime import run_directory, validate_run
from .selection import load_selection


def target(bundle, method, seed, attempt):
    if method == METHOD:
        directory = run_directory(bundle, seed, attempt)
        receipt = validate_run(bundle, directory, seed)
        config = read(directory / "run_config.json")
        return {"method": METHOD, "seed": seed, "checkpoint": str(directory / "final/adapter"),
                "checkpoint_sha256": receipt["adapter_sha256"], "manifest_sha256": config["manifest_sha256"]}
    matches = [c for c in bundle.controls if c["method"] == method and c["seed"] == seed]
    if len(matches) != 1:
        raise PilotError("Expected one verified control for this method/seed")
    return matches[0]


def question_file(bundle, benchmark):
    path = checked(Path(bundle.cfg["c1_eval_root"]) / "questions" / (benchmark + ".jsonl"), QUESTIONS[benchmark])
    questions = rows(path)
    ids = [r.get("problem_id", r.get("row_id")) for r in questions]
    if len(ids) != 500 or len(set(ids)) != 500 or any(not i for i in ids):
        raise PilotError("Frozen development question coverage differs")
    return path, set(ids)


def validate_predictions(records, config, expected, benchmark, question_ids):
    settings = {"precision": "fp32", "max_new_tokens": 1024, "batch_size": 16, "stop_at_question": False,
                "question_text": "p1_clean", "prompt_sha256": PROMPT_SHA, "add_special_tokens": False,
                "decoding": "greedy", "eos_token_id": 151643, "adapter_mode": "merged",
                "parser_version": "p3-parser-v6", "cut_at_next_question": True, "math_verify": "0.9.0",
                "torch": "2.6.0+cu124", "transformers": "4.57.1", "peft": "0.17.1", "cuda": "12.4",
                "gpu": "NVIDIA A100-SXM4-80GB", "eval_file_sha256": QUESTIONS[benchmark],
                "evaluate_py_sha256": EVALUATOR_SHA, "answer_parser_sha256": PARSER_SHA,
                "checkpoint_sha256": expected["checkpoint_sha256"], "train_manifest_sha256": expected["manifest_sha256"],
                "model": MODEL, "revision": REVISION,
                "train_seed": expected["seed"]}
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in settings.items()):
        differences = [k for k, v in settings.items() if type(config.get(k)) is not type(v) or config.get(k) != v]
        raise PilotError("Evaluation checkpoint/questions/runtime/contract mismatch: " + ", ".join(differences))
    expected_arm = "REPEAT" if expected["method"] == "A" else expected["method"]
    if config.get("train_arm") != expected_arm:
        raise PilotError("Evaluator loaded another training arm")
    ids = [r.get("problem_id") for r in records]
    if len(ids) != len(set(ids)) or set(ids) != question_ids or len(ids) != 500:
        raise PilotError("Predictions must cover the exact 500 frozen questions once each")
    from capstone_person4_approachC1.generation_scoring import classify
    for row in records:
        if type(row.get("correct")) is not bool:
            raise PilotError("Evaluation correctness must be explicit booleans")
        status, correct, _ = classify(row)
        if status == "execution_error":
            raise PilotError("Checker execution errors block evaluation; retry into a new attempt")
        if row.get("parser_version") != "p3-parser-v6":
            raise PilotError("Prediction row parser version changed")
        if type(row.get("generated_tokens")) is not int or row["generated_tokens"] < 0:
            raise PilotError("Invalid generated-token count")
    return {r["problem_id"]: r["correct"] for r in records}


def prediction_bytes(path):
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def native_c1_result(bundle, expected, benchmark, question_ids):
    # The publisher preserves native receipt/config bytes but reorganizes runs
    # and gzip-compresses predictions. Inspect only the two known locations;
    # never substitute unrelated results or search all of scratch.
    candidates = [("native", Path(bundle.cfg["c1_eval_root"]) / "results" /
                   f"C1_1.5B_p{bundle.cfg['parent_seed']}_t{expected['seed']}" / benchmark)]
    if bundle.cfg.get("work"):
        published = Path(bundle.cfg["work"]).parent / "github_screened_4955_latest/C1/evaluation"
        candidates.append(("published", published / f"1.5B_parent{bundle.cfg['parent_seed']}_seed{expected['seed']}" / benchmark))
    root = None
    layout = None
    for candidate_layout, candidate in candidates:
        try:
            (candidate / "completion.json").stat()
        except FileNotFoundError:
            continue
        # Preserve permission errors, and validate an existing receipt strictly
        # rather than silently falling back after a failed integrity check.
        root, layout = candidate, candidate_layout
        break
    if root is None:
        locations = "; ".join(str(p / "completion.json") for _, p in candidates)
        raise PilotError("C1 evaluation receipt not found in the native or published layout. Checked: " + locations)
    receipt = read(root / "completion.json")
    if (receipt.get("complete") is not True or receipt.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or receipt.get("eval_file_sha256") != QUESTIONS[benchmark]
            or receipt.get("benchmark") != benchmark or receipt.get("reserved_final_tests_performed") is not False):
        raise PilotError("Native C1 evaluation receipt belongs to another experiment")
    candidates = list(root.glob("*.config.json"))
    # The canonical native result sidecar is normally predictions.config.json;
    # source_config is preserved in the receipt when raw attempts are retained.
    matched = [p for p in candidates if sha(p) == receipt["config_sha256"]]
    if len(matched) != 1:
        raise PilotError("Expected one canonical C1 evaluation sidecar bound by its receipt")
    config_path = matched[0]
    stem = config_path.name.removesuffix(".config.json")
    predictions = [root / (stem + suffix) for suffix in (".jsonl", ".jsonl.gz")]
    predictions = [p for p in predictions if p.is_file()]
    if len(predictions) != 1:
        raise PilotError("Missing or ambiguous canonical C1 prediction file")
    raw = prediction_bytes(predictions[0])
    if hashlib.sha256(raw).hexdigest() != receipt["predictions_sha256"]:
        raise PilotError("Native C1 prediction bytes changed")
    config = read(config_path)
    records = rows(predictions[0])
    validate_predictions(records, config, expected, benchmark, question_ids)
    return records, config, {"completion_sha256": sha(root / "completion.json"),
                             "source": str(predictions[0]), "source_layout": layout, "reused": True}


def result_root(bundle, method, seed, benchmark, evaluation_attempt):
    from .common import attempt_name
    return bundle.root / "evaluation" / attempt_name(evaluation_attempt) / f"{method}_seed{seed}" / benchmark


def validate_result(bundle, method, seed, benchmark, attempt, evaluation_attempt):
    expected = target(bundle, method, seed, attempt)
    checked(Path(expected["checkpoint"]) / "adapter_model.safetensors", expected["checkpoint_sha256"])
    _, ids = question_file(bundle, benchmark)
    dest = result_root(bundle, method, seed, benchmark, evaluation_attempt)
    receipt = read(dest / "completion.json")
    for filename, key in (("predictions.jsonl", "predictions_sha256"), ("predictions.config.json", "config_sha256")):
        checked(dest / filename, receipt[key])
    binding = {"identity": bundle.identity, "target": expected, "benchmark": benchmark,
               "evaluation_attempt": evaluation_attempt, "complete": True}
    if any(receipt.get(k) != v for k, v in binding.items()):
        raise PilotError("Pilot evaluation receipt is stale")
    values = validate_predictions(rows(dest / "predictions.jsonl"), read(dest / "predictions.config.json"), expected, benchmark, ids)
    return values, receipt


def evaluate(bundle, methods=("A", "C1", METHOD), seeds=SEEDS, attempt="attempt01", evaluation_attempt="eval01"):
    for package, version in EVAL_PACKAGES.items():
        if importlib.metadata.version(package) != version:
            raise PilotError("Use the native C1 evaluator environment: " + package)
    c = bundle.context
    evaluator = checked(c.runtime / "vendor/evaluation/evaluate.py", EVALUATOR_SHA)
    checked(c.runtime / "vendor/evaluation/answer_parser.py", PARSER_SHA)
    results = []
    for method in methods:
        for seed in seeds:
            expected = target(bundle, method, seed, attempt)
            checkpoint = Path(expected["checkpoint"]) / "adapter_model.safetensors"
            checked(checkpoint, expected["checkpoint_sha256"])
            for benchmark in QUESTIONS:
                question_path, ids = question_file(bundle, benchmark)
                dest = result_root(bundle, method, seed, benchmark, evaluation_attempt)
                if (dest / "completion.json").is_file():
                    results.append(validate_result(bundle, method, seed, benchmark, attempt, evaluation_attempt)[1])
                    continue
                if dest.exists():
                    raise PilotError("Incomplete evaluation preserved; choose a new --evaluation-attempt")
                if method == "C1":
                    records, config, origin = native_c1_result(bundle, expected, benchmark, ids)
                else:
                    # Save raw native output separately; never resume an interrupted JSONL blindly.
                    raw = dest / "native/raw.jsonl"
                    raw.parent.mkdir(parents=True, exist_ok=False)
                    command = [sys.executable, "-B", str(evaluator), "--eval_file", str(question_path),
                               "--adapter", expected["checkpoint"], "--expect_train_manifest_sha256", expected["manifest_sha256"],
                               "--run_id", f"{METHOD}_1.5B_p{bundle.cfg['parent_seed']}_{method}_s{seed}_{benchmark}", "--out", str(raw),
                               "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024", "--question_text", "p1_clean"]
                    subprocess.run(command, check=True)
                    config, records = read(raw.with_suffix(".config.json")), rows(raw)
                    origin = {"source": str(raw), "reused": False}
                validate_predictions(records, config, expected, benchmark, ids)
                checked(checkpoint, expected["checkpoint_sha256"])
                write_rows(dest / "predictions.jsonl", records)
                write(dest / "predictions.config.json", config)
                receipt = {"identity": bundle.identity, "target": expected, "benchmark": benchmark,
                           "evaluation_attempt": evaluation_attempt, "complete": True, "origin": origin,
                           "predictions_sha256": sha(dest / "predictions.jsonl"),
                           "config_sha256": sha(dest / "predictions.config.json"), "reserved_final_tests_performed": False}
                write(dest / "completion.json", receipt)
                results.append(validate_result(bundle, method, seed, benchmark, attempt, evaluation_attempt)[1])
    return results


def paired_bootstrap(left, right, resamples=2000, seed=0):
    """Question-paired bootstrap, stratified by benchmark for equal-weight means."""
    if set(left) != set(right) or not left or resamples < 1:
        raise PilotError("Paired bootstrap requires matching nonempty benchmark sets")
    differences = []
    for benchmark in sorted(left):
        if set(left[benchmark]) != set(right[benchmark]) or not left[benchmark]:
            raise PilotError("Paired comparison question coverage differs")
        differences.append([int(left[benchmark][i]) - int(right[benchmark][i]) for i in sorted(left[benchmark])])
    rng = random.Random(seed)
    observed = statistics.mean(statistics.mean(v) for v in differences) * 100
    sampled = sorted(100 * statistics.mean(sum(rng.choice(v) for _ in v) / len(v) for v in differences)
                     for _ in range(resamples))
    return {"difference_pp": observed, "ci95_low_pp": sampled[int(0.025 * (resamples - 1))],
            "ci95_high_pp": sampled[int(0.975 * (resamples - 1))], "bootstrap_resamples": resamples,
            "bootstrap_seed": seed, "uncertainty_scope": "paired questions for fixed checkpoints; not training-seed uncertainty"}


def csv_bytes(records):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode()


def finalize(bundle, attempt="attempt01", evaluation_attempt="eval01"):
    _, selection = load_selection(bundle)
    table, contrasts, index, evaluation_receipts = [], [], [], []
    means = {method: [] for method in ("A", "C1", METHOD)}
    for seed in SEEDS:
        receipt = validate_run(bundle, run_directory(bundle, seed, attempt), seed)
        index.append({"method": METHOD, "model": bundle.cfg["model"], "parent_seed": bundle.cfg["parent_seed"], "continuation_seed": seed,
                      "checkpoint": receipt["checkpoint"], "checkpoint_sha256": receipt["adapter_sha256"],
                      "replay_tokens": receipt["portion_tokens"]["replay"], "fresh_tokens": receipt["portion_tokens"]["fresh"]})
        values = {method: {} for method in means}
        for method in means:
            for benchmark in QUESTIONS:
                outcomes, eval_receipt = validate_result(bundle, method, seed, benchmark, attempt, evaluation_attempt)
                evaluation_receipts.append(eval_receipt)
                values[method][benchmark] = outcomes
                table.append({"model": bundle.cfg["model"], "parent_seed": bundle.cfg["parent_seed"], "method": method, "continuation_seed": seed, "benchmark": benchmark,
                              "questions": len(outcomes), "accuracy_percent": 100 * statistics.mean(outcomes.values())})
            means[method].append(100 * statistics.mean(statistics.mean(v.values()) for v in values[method].values()))
        for control in ("A", "C1"):
            for scope in ("dev_gsm", "dev_math", "equal_weight_mean"):
                benchmarks = list(QUESTIONS) if scope == "equal_weight_mean" else [scope]
                result = paired_bootstrap({b: values[METHOD][b] for b in benchmarks}, {b: values[control][b] for b in benchmarks})
                contrasts.append({"model": bundle.cfg["model"], "parent_seed": bundle.cfg["parent_seed"], "continuation_seed": seed, "contrast": METHOD + "_minus_" + control,
                                  "benchmark": scope, **result})
    summary = {"version": VERSION, "model": bundle.cfg["model"], "parent_seed": bundle.cfg["parent_seed"], "identity": bundle.identity, "selection": selection,
               "evaluation_receipts_sha256": digest(evaluation_receipts),
               "three_seed_summary": {method: {"mean_accuracy_percent": statistics.mean(v),
                   "sample_sd_percent": statistics.stdev(v), "seeds": list(SEEDS)} for method, v in means.items()},
               "contrasts": contrasts,
               "limitations": ["One fixed ratio and model size; this report covers one parent, with selection variability not estimated.",
                    "Inherited parent/development exposure and unresolved historical matcher audit remain disclosed.",
                    "Bootstrap intervals describe fixed checkpoints; three training seeds are reported descriptively.",
                    "No reserved final tests were used."]}
    dest = bundle.root / "reports" / evaluation_attempt
    write_bytes(dest / "accuracy.csv", csv_bytes(table))
    write_bytes(dest / "paired_differences.csv", csv_bytes(contrasts))
    write_bytes(dest / "run_index.csv", csv_bytes(index))
    write(dest / "summary.json", summary)
    write(dest / "provenance.json", bundle.binding)
    return {"complete": True, "report_directory": str(dest), "three_seed_summary": summary["three_seed_summary"]}

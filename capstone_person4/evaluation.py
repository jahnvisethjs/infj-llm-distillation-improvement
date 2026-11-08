"""Call a frozen adapter around Person 3's evaluator; retain paired predictions."""

from collections import defaultdict
import importlib.util
from pathlib import Path
import random
import statistics
import sys
import time

from .audit import checked_file, gate
from .config import artifact_root, stage_fingerprint, without_paths
from .provenance import PipelineError, digest, environment, read_json, read_rows, sha, source_hashes, write_json, write_rows

STATUSES = {"correct", "incorrect", "no_answer", "truncated", "unresolved_equivalence", "execution_error"}


def method_definition(cfg):
    # Continuation seed is the intended replication axis; selection/data seeds remain fixed.
    training = {k: v for k, v in cfg["training"].items() if k != "seed"}
    protocol = {k: cfg["evaluation"].get(k) for k in ("contract_sha256", "script_sha256", "development_sha256", "final_sha256")}
    return digest(without_paths({k: cfg[k] for k in ("parent", "data", "scoring", "allocation")}) |
                  {"training": training, "evaluation": protocol, "code": source_hashes()})


def validate_predictions(predictions, questions):
    expected = {r.get("problem_id", r.get("row_id")) for r in questions}
    ids = [r.get("problem_id") for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise PipelineError("Evaluator predictions must cover the exact frozen questions once each")
    for row in predictions:
        if row.get("status") not in STATUSES or type(row.get("correct")) is not bool:
            raise PipelineError("Prediction requires explicit status and boolean correct")
        if row["correct"] != (row["status"] == "correct"):
            raise PipelineError("Correctness/status disagree; use flags for an abnormal stop after a checked answer")
        for key in ("generated_text", "extracted_answer", "generated_tokens", "stop_reason", "flags"):
            if key not in row:
                raise PipelineError(f"Prediction lacks {key}")
        if type(row["generated_tokens"]) is not int or row["generated_tokens"] < 0 or not isinstance(row["flags"], list):
            raise PipelineError("Invalid generated token count or flags")
        if row["status"] == "execution_error":
            raise PipelineError("Retry evaluator execution errors; incomplete executions cannot be reported as mathematical accuracy")
    return predictions


def evaluate(cfg, method, attempt, run=None, split="development"):
    gate(cfg, "evaluate")
    if method not in ("parent", "B", "C0", "C1", "E") or split not in ("development", "final"):
        raise PipelineError("Expected parent/B/C0/C1/E and development/final")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise PipelineError("Use a simple new evaluation attempt name")
    if split == "final":
        freeze_path = checked_file(cfg["evaluation"].get("method_freeze"), cfg["evaluation"].get("method_freeze_sha256"), "frozen method evidence")
        freeze = read_json(freeze_path)
        if freeze.get("method_definition") != method_definition(cfg) or freeze.get("frozen") is not True:
            raise PipelineError("Freeze the exact method and recipe before final evaluation")
    from .protection import canonical_question
    questions = [canonical_question(r) for r in read_rows(cfg["evaluation"][split + "_manifest"])]
    contract = read_json(cfg["evaluation"]["evaluation_contract"])
    if method == "parent":
        if run:
            raise PipelineError("Parent evaluation always uses the configured untouched parent")
        adapter_path = cfg["parent"]["adapter"]
        checkpoint_hash = cfg["parent"]["adapter_sha256"]
        training_seed = None
        reconciliation = None
    else:
        if not run:
            raise PipelineError("Supply --run pointing to the completed continuation run directory")
        run = Path(run).resolve()
        reconciliation = read_json(run / "token_reconciliation.json")
        if (reconciliation.get("valid_controlled_run") is not True or reconciliation.get("method") != method
                or reconciliation.get("stage_fingerprint") != stage_fingerprint(cfg, "train")
                or reconciliation.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]
                or reconciliation.get("run_config_sha256") != sha(run / "run_config.json")):
            raise PipelineError("Evaluation requires a completed, reconciled arm from this exact configuration and parent")
        adapter_path = str(Path(reconciliation["checkpoint"]) / "adapter")
        checkpoint_hash = reconciliation["checkpoint_sha256"]
        training_seed = read_json(run / "run_config.json")["seed"]
    checked_file(Path(adapter_path) / "adapter_model.safetensors", checkpoint_hash, "evaluation adapter")
    script = cfg["evaluation"]["evaluator_script"]
    spec = importlib.util.spec_from_file_location("person4_person3_evaluator", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Existing evaluator adapters may import their neighboring checker modules.
    sys.path.insert(0, str(Path(script).parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, "evaluate", None)):
        raise PipelineError("Evaluator adapter must export evaluate(request) -> prediction records")
    request = {"interface": "person4-evaluate-v1", "model": cfg["parent"]["model"],
               "revision": cfg["parent"]["revision"], "adapter": adapter_path,
               "adapter_sha256": checkpoint_hash, "contract": contract,
               "questions": [{"problem_id": r.get("problem_id", r.get("row_id")), "problem": r["problem"]} for r in questions],
               # Gold is deliberately separate from the question-only inference inputs.
               "references": {r.get("problem_id", r.get("row_id")): r.get("final_answer", r.get("answer")) for r in questions}}
    started = time.perf_counter()
    predictions = validate_predictions(list(module.evaluate(request)), questions)
    seconds = time.perf_counter() - started
    if sha(script) != cfg["evaluation"]["script_sha256"] or sha(Path(adapter_path) / "adapter_model.safetensors") != checkpoint_hash:
        raise PipelineError("Evaluator source or evaluated checkpoint changed during evaluation")
    by_id = {r.get("problem_id", r.get("row_id")): r for r in questions}
    predictions = [{**p, "benchmark": by_id[p["problem_id"]]["benchmark"]} for p in predictions]
    root = artifact_root(cfg, "evaluate") / "evaluation" / split / method / attempt
    write_rows(root / "predictions.jsonl", predictions)
    report = {"synthetic": False, "method": method, "split": split, "training_seed": training_seed,
              "checkpoint_sha256": checkpoint_hash, "parent_fingerprint": cfg["parent"]["adapter_sha256"],
              "config_fingerprint": cfg["config_fingerprint"], "method_definition": method_definition(cfg),
              "training_stage_fingerprint": stage_fingerprint(cfg, "train"),
              "evaluation_contract_sha256": cfg["evaluation"]["contract_sha256"],
              "questions_sha256": cfg["evaluation"][split + "_sha256"],
              "predictions": str(root / "predictions.jsonl"), "predictions_sha256": sha(root / "predictions.jsonl"),
              "training_reconciliation": reconciliation, "rows": len(predictions)}
    report["reference_runtime_reports"] = [{"path": str(p), "sha256": sha(p)} for p in
                                           sorted((artifact_root(cfg, "score-reference") / "reference/runtime").glob("*.json"))]
    report["runtime"] = {"evaluation_wall_seconds": seconds, "generated_tokens": sum(p["generated_tokens"] for p in predictions),
                         "environment": environment(), "evaluator_metadata": module.runtime_metadata() if callable(getattr(module, "runtime_metadata", None)) else None,
                         "scope": "wall time of evaluator call, including any model loading inside that call"}
    write_json(root / "evaluation_report.json", report)
    return {**report, "report": str(root / "evaluation_report.json")}


def paired_difference(left, right, resamples=2000, seed=0):
    if set(left) != set(right) or not left or type(resamples) is not int or resamples < 1:
        raise PipelineError("Paired comparison requires identical nonempty problem IDs and positive bootstrap resamples")
    differences = [int(left[k]) - int(right[k]) for k in sorted(left)]
    rng = random.Random(seed)
    draws = sorted(100 * sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(resamples))
    return {"difference_pp": 100 * sum(differences) / len(differences),
            "ci95_pp": [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]],
            "questions": len(differences), "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "uncertainty_scope": "paired evaluation questions; does not estimate training-seed uncertainty"}


def compare(cfg, report_paths, attempt):
    gate(cfg, "compare")
    if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
        raise PipelineError("Use a simple new comparison attempt name")
    reports = [read_json(path) for path in report_paths]
    by_method = defaultdict(list)
    common = ("parent_fingerprint", "evaluation_contract_sha256", "questions_sha256", "split", "method_definition")
    for report in reports:
        if (report.get("synthetic") is not False or report.get("method_definition") != method_definition(cfg)
                or report.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]
                or report.get("evaluation_contract_sha256") != cfg["evaluation"]["contract_sha256"]
                or any(report.get(k) != reports[0].get(k) for k in common)):
            raise PipelineError("Comparison mixes synthetic outputs, parents, profiles, questions, protocols or methods")
        checked_file(report["predictions"], report["predictions_sha256"], "comparison predictions")
        if report["method"] in ("B", "C0", "C1"):
            rec = report.get("training_reconciliation") or {}
            if (rec.get("valid_controlled_run") is not True or rec.get("training_smoke_only") is not False
                    or rec.get("method") != report["method"]
                    or (rec.get("stage_fingerprint") != report["training_stage_fingerprint"] if report.get("training_stage_fingerprint")
                        else rec.get("config_fingerprint") != report["config_fingerprint"])
                    or rec.get("parent_fingerprint") != report["parent_fingerprint"]
                    or rec.get("checkpoint_sha256") != report["checkpoint_sha256"]
                    or rec.get("actual_tokens") != rec.get("expected_tokens") or not rec.get("target_tokens")
                    or abs(rec["actual_tokens"] - rec["target_tokens"]) > rec["target_tokens"] * rec["budget_tolerance"]):
                raise PipelineError("Comparison requires reconciled full-arm checkpoints and verified consumed budgets")
        predictions = read_rows(report["predictions"])
        questions = read_rows(cfg["evaluation"][report["split"] + "_manifest"])
        validate_predictions(predictions, questions)
        by_method[report["method"]].append((report, predictions))
    arms = [m for m in ("B", "C0", "C1") if m in by_method]
    if "parent" not in by_method or len(by_method["parent"]) != 1 or "C0" not in by_method or len(arms) < 2:
        raise PipelineError("Supply one untouched parent report, C0 and at least one other matched arm (B and/or C1)")
    runs = {}
    for method in arms:
        seeds = [report["training_seed"] for report, _ in by_method[method]]
        if len(set(seeds)) != len(seeds):
            raise PipelineError("Duplicate training seeds; manifest-draw variability must be analyzed separately")
        runs[method] = {r["training_seed"]: p for r, p in by_method[method]}
    if any(set(runs[m]) != set(runs["C0"]) for m in arms):
        raise PipelineError("Match continuation seeds across every arm")
    parent = by_method["parent"][0][1]
    controls = [r["training_reconciliation"] for r in reports if r["method"] in arms]
    if any(any(r.get(k) != controls[0].get(k) for k in ("candidate_sha256", "target_tokens", "budget_tolerance")) for r in controls):
        raise PipelineError("Continuation arms do not share the candidate universe and token-budget policy")
    primary = ("C1", "C0") if "C1" in arms else ("C0", "B")
    contrasts = [(a, b) for i, a in enumerate(arms) for b in arms[:i]]  # later arm minus earlier: C0-B, C1-B, C1-C0
    benchmarks = sorted({p["benchmark"] for p in parent})
    results = []
    for seed in sorted(runs["C0"]):
        for benchmark in benchmarks:
            def outcomes(predictions):
                return {p["problem_id"]: p["correct"] for p in predictions if p["benchmark"] == benchmark}
            def diagnostics(predictions):
                part = [p for p in predictions if p["benchmark"] == benchmark]
                return {"extracted_answer_rate": sum(p["extracted_answer"] is not None for p in part) / len(part),
                        "truncation_rate": sum(p["status"] == "truncated" or "truncated" in p["flags"] for p in part) / len(part),
                        "unresolved_equivalence_rate": sum(p["status"] == "unresolved_equivalence" for p in part) / len(part),
                        "mean_generated_tokens": statistics.mean(p["generated_tokens"] for p in part)}
            scored = {"parent": outcomes(parent), **{m: outcomes(runs[m][seed]) for m in arms}}
            row = {"benchmark": benchmark, "training_seed": seed, "arms": arms, "primary_contrast": f"{primary[0]}_minus_{primary[1]}",
                   "parent_accuracy": sum(scored["parent"].values()) / len(scored["parent"]),
                   "diagnostics": {"parent": diagnostics(parent), **{m: diagnostics(runs[m][seed]) for m in arms}}}
            for m in arms:
                row[f"{m}_accuracy"] = sum(scored[m].values()) / len(scored[m])
                row[f"{m}_minus_parent"] = paired_difference(scored[m], scored["parent"])
            for a, b in contrasts:
                row[f"{a}_minus_{b}"] = paired_difference(scored[a], scored[b])
            results.append(row)
    key = f"{primary[0]}_minus_{primary[1]}"
    variability = {}
    for benchmark in benchmarks:
        differences = [r[key]["difference_pp"] for r in results if r["benchmark"] == benchmark]
        variability[benchmark] = {"matched_seed_count": len(differences), f"mean_{key}_pp": statistics.mean(differences),
                                  "training_seed_sd_pp": statistics.stdev(differences) if len(differences) > 1 else None,
                                  "scope": "variation across matched continuation seeds, separate from paired question intervals"}
    scoring_costs = {}
    for report in reports:
        for entry in report.get("reference_runtime_reports", []):
            checked_file(entry["path"], entry["sha256"], "scoring runtime report")
            scoring_costs[entry["path"]] = read_json(entry["path"])
    result = {"synthetic": False, "arms": arms, "primary_contrast": key, "results": results, "training_seed_variability": variability,
              "source_reports": [str(Path(p).resolve()) for p in report_paths],
              "evaluation_costs": [{"method": r["method"], "seed": r["training_seed"], "runtime": r.get("runtime")} for r in reports],
              "training_costs": [{"method": r["method"], "seed": r["training_seed"], "reconciliation": r["training_reconciliation"]}
                                 for r in reports if r["method"] in arms],
              "scoring_costs": scoring_costs,
              "training_seed_count": len(runs["C0"]), "interpretation": "A single training seed is pilot evidence; no accuracy improvement is assumed"}
    root = artifact_root(cfg, "evaluate") / "comparison" / attempt
    write_json(root / "comparison.json", result)
    header = ["Benchmark", "Seed", "Parent %"] + [f"{m} %" for m in arms] + [f"{key.replace('_', ' ')} pp (paired 95% interval)"]
    lines = [f"# {primary[0]} versus {primary[1]} comparison", "", "| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for r in results:
        d = r[key]
        cells = [r["benchmark"], str(r["training_seed"]), f"{100*r['parent_accuracy']:.2f}"] + [f"{100*r[m + '_accuracy']:.2f}" for m in arms] + [f"{d['difference_pp']:.2f} {d['ci95_pp']}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", result["interpretation"], "", "Intervals describe evaluation-item uncertainty, not training-run uncertainty."]
    from .provenance import write_bytes
    write_bytes(root / "comparison.md", ("\n".join(lines) + "\n").encode())
    return result

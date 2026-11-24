"""Explicit GPU parity smoke: native batched evaluator versus live checkpoint loading.

This standalone script does not change the routing package or previous receipts.
Only fixed training questions are used. No router is fitted and no test labels
are accessed. Actual GPU execution requires --authorize-gpu.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys


def gpu_preflight():
    from capstone_routing.common import SETTINGS, require
    for package, field in (("torch", "torch"), ("transformers", "transformers"),
                           ("peft", "peft"), ("math-verify", "math_verify")):
        require(version(package) == SETTINGS[field], f"Wrong GPU environment: {package}")
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Smoke requires exactly one allocated NVIDIA GPU")
    require(torch.version.cuda == SETTINGS["cuda"] and torch.cuda.get_device_name(0) == SETTINGS["gpu"],
            "Allocated GPU/CUDA differs from the frozen evaluation runtime")
    return {"actual_gpu_execution": True, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def smoke_questions(grading, questions, per_benchmark=8):
    from capstone_routing.common import digest, keyed, qkey, require
    require(type(per_benchmark) is int and per_benchmark > 0, "Invalid smoke size")
    expected = keyed(questions)
    require(set(keyed(grading)) == set(expected), "Grading/feature question coverage differs")
    result = []
    for bench in ("gsm", "math"):
        selected = sorted((r for r in grading if r["benchmark"] == bench), key=lambda r: digest(qkey(r)))[:per_benchmark]
        require(len(selected) == per_benchmark, "Insufficient training questions for smoke")
        for row in selected:
            require(row["split"] == "train" and expected[qkey(row)]["split"] == "train"
                    and row["question"] == expected[qkey(row)]["question"]
                    and row.get("gold_answer") not in (None, ""), "Smoke must use unchanged training questions only")
        result.extend(selected)
    return result


def compare(native, live):
    differences = []
    for native_key, live_key in (("raw_output", "answer_text"), ("generated_tokens", "generated_tokens"),
                                  ("finish_reason", "finish_reason")):
        if native.get(native_key) != live.get(live_key):
            differences.append(native_key)
    return differences


def run(args):
    if not args.authorize_gpu:
        raise ValueError("Explicit --authorize-gpu is required for the generation smoke")
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    import capstone_routing
    from capstone_routing.common import (EVALUATOR_SHA, PARSER_SHA, bundle, checked, destination,
        identity, keyed, load_config, qkey, require, rows, seal, sha, write, write_rows)
    from capstone_routing.inputs import load_registry, prediction_records
    from capstone_routing.live import VerifiedGenerator
    from capstone_routing.pool import pool_questions

    require(Path(capstone_routing.__file__).resolve().parent.parent == code_root,
            "Another routing code directory was imported")
    cfg = load_config(args.config)
    registry = load_registry(args.registry)
    require(not registry["synthetic"], "GPU smoke requires real registered checkpoints")
    bundle(args.pool, "pool")
    questions = smoke_questions(rows(Path(args.pool) / "grading/train.jsonl"), pool_questions(args.pool, ("train",)))
    candidates = [registry["candidates"][identity(args.parent_seed, m, args.continuation_seed)] for m in ("A", "B", "C1")]
    require(all(c.get("recipe") for c in candidates), "Register checkpoint loading proofs before the GPU smoke")
    evaluator = checked(cfg["evaluator"]["path"], EVALUATOR_SHA)
    parser_path = checked(cfg["evaluator"]["parser_path"], PARSER_SHA)
    require(parser_path == evaluator.parent / "answer_parser.py", "Native parser must be the verified evaluator sibling")
    runtime = gpu_preflight()
    root = destination(args.output, [args.registry, args.pool, code_root / "capstone_routing", evaluator, parser_path])
    require(not root.exists(), "Smoke output exists; preserve it and select a new output attempt")
    question_path = root / "questions.jsonl"
    write_rows(question_path, questions)
    question_map = keyed(questions)
    native_results = {}

    for candidate in candidates:
        cid = candidate["id"]
        directory = root / "native" / cid
        directory.mkdir(parents=True)
        predictions = directory / "predictions.jsonl"
        command = [sys.executable, "-B", str(evaluator), "--eval_file", str(question_path),
                   "--out", str(predictions), "--run_id", "routing_smoke_" + cid,
                   "--model", candidate["model"], "--revision", candidate["revision"],
                   "--adapter", candidate["checkpoint"], "--expect_train_manifest_sha256", candidate["train_manifest_sha256"],
                   "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024", "--question_text", "p1_clean"]
        write(directory / "command.json", command)
        print("Native evaluator:", cid, flush=True)
        with (directory / "native.log").open("x", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        _, coverage = prediction_records({"predictions": str(predictions), "config": str(directory / "predictions.config.json"),
                                          "questions_sha256": sha(question_path)}, candidate, question_map)
        require(coverage["missing_rows"] == 0, "Incomplete native smoke predictions")
        native_results[cid] = keyed(rows(predictions))

    generator = VerifiedGenerator(cfg["evaluator"])
    live_results, mismatches = [], []
    for candidate in candidates:
        cid = candidate["id"]
        print("Live loading/generation:", cid, flush=True)
        for row in questions:
            live = generator.generate(row["question"], candidate)
            result = {"candidate": cid, "benchmark": row["benchmark"], "problem_id": row["problem_id"], **live}
            live_results.append(result)
            differences = compare(native_results[cid][qkey(row)], live)
            if differences:
                mismatches.append({"candidate": cid, "benchmark": row["benchmark"], "problem_id": row["problem_id"],
                                   "different_fields": differences})
    write_rows(root / "live_answers.jsonl", live_results)
    summary = {"complete": not mismatches, "native_live_match": not mismatches,
               "selected_candidates": [c["id"] for c in candidates], "questions_per_candidate": len(questions),
               "native_responses": len(live_results), "live_responses": len(live_results),
               "mismatch_count": len(mismatches), "mismatches": mismatches, "runtime": runtime,
               "comparison": "exact completion text, token count, and stopping reason; native batch 16 versus live batch 1",
               "question_split": "train", "test_questions_used": False, "router_training_executed": False,
               "script_sha256": sha(__file__)}
    write(root / "summary.json", summary)
    seal(root, "generation_smoke", [args.registry, args.pool], synthetic=not runtime["actual_gpu_execution"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "config", "registry", "pool", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--parent-seed", type=int, default=17, choices=(17, 123))
    parser.add_argument("--continuation-seed", type=int, default=42, choices=(42, 137, 2026))
    parser.add_argument("--authorize-gpu", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(args)
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"complete": False, "error": str(exc), "output": args.output}), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

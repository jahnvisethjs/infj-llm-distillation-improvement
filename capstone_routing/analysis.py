"""Recorded-answer complementarity and explicitly exploratory question cues."""

from collections import defaultdict
from pathlib import Path
import re
import statistics

from .common import (METHODS, PARENTS, QUESTIONS, SEEDS, digest, identity, question,
                     require, seal, write, write_csv, write_rows, bundle)
from .inputs import development_questions, inspect, prediction_records


def opportunity_metrics(matrix, candidates, reference):
    require(matrix and reference in candidates, "Need questions and reference candidate")
    n = len(matrix)
    accuracy = {m: sum(r[m] for r in matrix) / n for m in candidates}
    alternatives = [m for m in candidates if m != reference]
    result = []
    for alternative in alternatives:
        counts = {"rescues": 0, "risks": 0, "both_correct": 0, "both_wrong": 0}
        for row in matrix:
            a, x = row[reference], row[alternative]
            name = "both_correct" if a and x else "both_wrong" if not a and not x else "rescues" if x else "risks"
            counts[name] += 1
        result.append({"alternative": alternative, "n": n, "reference_accuracy": accuracy[reference],
                       "alternative_accuracy": accuracy[alternative], **counts,
                       **{k + "_fraction": v / n for k, v in counts.items()},
                       "oracle_accuracy": accuracy[reference] + counts["rescues"] / n,
                       "available_improvement_pp": 100 * counts["rescues"] / n,
                       "accuracy_difference_pp": 100 * (counts["rescues"] - counts["risks"]) / n})
    rescues = sum(not r[reference] and any(r[m] for m in alternatives) for r in matrix)
    result.append({"alternative": "joint", "n": n, "reference_accuracy": accuracy[reference],
                   "rescues": rescues, "rescues_fraction": rescues / n,
                   "oracle_accuracy": sum(any(r[m] for m in candidates) for r in matrix) / n,
                   "available_improvement_pp": 100 * rescues / n})
    return result


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        if row["alternative"] == "joint":
            groups[row["experiment"], row["parent_seed"], row["benchmark"]].append(row)
    return [{"experiment": k[0], "parent_seed": k[1], "benchmark": k[2], "complete_blocks": len(v),
             "seeds": [r["continuation_seed"] for r in v],
             "mean_available_improvement_pp": statistics.mean(r["available_improvement_pp"] for r in v),
             "sample_sd_pp": statistics.stdev(r["available_improvement_pp"] for r in v) if len(v) > 1 else None,
             "min_pp": min(r["available_improvement_pp"] for r in v),
             "max_pp": max(r["available_improvement_pp"] for r in v)} for k, v in sorted(groups.items())]


def opportunity(cfg, output):
    root = Path(output)
    registry = inspect(cfg)
    questions, specs = development_questions(cfg)
    values, index, rejected = {}, [], list(registry["rejected"])
    for cid, candidate in registry["candidates"].items():
        combined = {}
        valid = True
        for bench in ("gsm", "math"):
            try:
                choices = registry["predictions"].get(cid + "/" + bench, [])
                require(len(choices) == 1, f"Expected one explicit attempt, found {len(choices)}")
                spec = {**choices[0], "questions_sha256": QUESTIONS[bench]}
                subset = {k: v for k, v in questions.items() if k[0] == bench}
                outcome, coverage = prediction_records(spec, candidate, subset)
                combined.update(outcome)
                index.append({"candidate": cid, "parent_seed": candidate["parent_seed"],
                              "continuation_seed": candidate["continuation_seed"], "method": candidate["method"],
                              "benchmark": bench, **spec, **coverage, "checkpoint_sha256": candidate["checkpoint_sha256"],
                              "parent_sha256": candidate["parent_sha256"], "actual_tokens": candidate["actual_tokens"]})
            except (OSError, KeyError, ValueError) as exc:
                valid = False
                rejected.append({"candidate": cid, "benchmark": bench, "reason": str(exc)})
        if valid:
            values[cid] = combined
    all_metrics, all_matrix, blocked = [], [], []
    for parent in PARENTS:
        for seed in SEEDS:
            ref = identity(parent, "A", seed)
            pools = {"policy": [identity(parent, m, seed) for m in METHODS],
                     "seed_diversity": [identity(parent, "A", s) for s in SEEDS]}
            for experiment, pool in pools.items():
                block = f"{experiment}_p{parent}_t{seed}"
                missing = [cid for cid in pool if cid not in values]
                if missing:
                    blocked.append({"block": block, "missing_or_invalid_candidates": missing})
                    continue
                matrix = [{"benchmark": k[0], "problem_id": k[1], "question": question(questions[k]),
                           **{cid: values[cid][k] for cid in pool}} for k in sorted(questions)]
                for row in matrix:
                    all_matrix.append({"experiment": experiment, "parent_seed": parent, "continuation_seed": seed,
                                       "benchmark": row["benchmark"], "problem_id": row["problem_id"],
                                       "question": row["question"],
                                       **({m + "_correct": row[identity(parent, m, seed)] for m in METHODS}
                                          if experiment == "policy" else {cid + "_correct": row[cid] for cid in pool})})
                for bench in ("gsm", "math", "combined"):
                    subset = [r for r in matrix if bench == "combined" or r["benchmark"] == bench]
                    for result in opportunity_metrics(subset, pool, ref):
                        all_metrics.append({"experiment": experiment, "parent_seed": parent,
                                            "continuation_seed": seed, "benchmark": bench, **result})
                write_csv(root / "blocks" / block / "correctness_matrix.csv", matrix)
    write(root / "registry.json", registry)
    write_rows(root / "questions.jsonl", list(questions.values()))
    write_csv(root / "correctness_matrix.csv", all_matrix)
    write_csv(root / "opportunity.csv", all_metrics)
    write_csv(root / "run_index.csv", index)
    summary = {"complete": not blocked, "complete_block_count": 12 - len(blocked), "blocked": blocked,
               "input_rejections": rejected, "summaries": summarize(all_metrics), "questions": specs,
               "interpretation": "Available improvement is an oracle upper bound, not achievable routing gain.",
               "seed_control_dependence": "The same three A checkpoints are reused across reference seeds per parent.",
               "official_final_tests_performed": False}
    write(root / "summary.json", summary)
    seal(root, "opportunity", synthetic=registry["synthetic"])
    return summary


TOPICS = {"arithmetic": r"\b(sum|difference|product|divide|remainder|total)\b",
          "algebra": r"\b(equation|polynomial|quadratic|variable|solve for)\b",
          "geometry": r"\b(triangle|circle|radius|angle|rectangle|perimeter|area|volume)\b",
          "probability_counting": r"\b(probability|random|arrangements|combinations|ways|permutations)\b",
          "number_theory": r"\b(prime|divisor|divisible|integer|gcd|lcm|modulo)\b",
          "rates_ratios": r"\b(ratio|rate|speed|per hour|percent|percentage)\b"}


def features(text):
    """Question-only cues; unknown/mixed tags avoid pretending to solve the problem."""
    words = re.findall(r"\b\w+\b", text)
    prose = len(re.findall(r"\b[A-Za-z]{3,}\b", text)) >= 8
    symbolic = bool(re.search(r"[=<>^]|\\(?:frac|sum|sqrt)|\b[a-z]\s*[+*/-]\s*\w", text))
    found = [k for k, pattern in TOPICS.items() if re.search(pattern, text, re.I)]
    return {"characters": len(text), "words": len(words), "topics": found or ["unknown"],
            "form": "mixed" if prose and symbolic else "word" if prose else "symbolic" if symbolic else "unknown",
            "fractions": bool(re.search(r"\\[dt]?frac|\b\d+\s*/\s*\d+", text)),
            "percentages": bool(re.search(r"%|\bpercent(?:age)?\b", text, re.I)),
            "units": bool(re.search(r"\b(?:meters?|metres?|cm|km|miles?|feet|inches|hours?|minutes?|seconds?|kg|grams?|liters?|dollars?|cents?)\b|\$\d", text, re.I)),
            "multiple_constraints": len(re.findall(r"\b(?:if|given|such that|at least|at most|and)\b|[=<>]", text, re.I)) >= 2}


def patterns(opportunity_root, output):
    receipt = bundle(opportunity_root, "opportunity")
    source = Path(opportunity_root)
    with (source / "correctness_matrix.csv").open(encoding="utf-8", newline="") as stream:
        import csv
        matrix = [r for r in csv.DictReader(stream) if r["experiment"] == "policy"]
    unique = {(r["benchmark"], r["problem_id"]): features(r["question"]) for r in matrix}
    lengths = sorted(v["words"] for v in unique.values())
    cuts = [lengths[int((len(lengths) - 1) * p)] for p in (.25, .5, .75)] if lengths else []
    groups, examples = defaultdict(list), []
    for row in matrix:
        f = unique[row["benchmark"], row["problem_id"]]
        tags = {"form": [f["form"]], "length_quartile": [str(1 + sum(f["words"] > c for c in cuts))],
                "topic": f["topics"], **{k: [str(f[k]).lower()] for k in ("fractions", "percentages", "units", "multiple_constraints")}}
        for alternative in ("B", "C1"):
            a, x = int(row["A_correct"]), int(row[alternative + "_correct"])
            item = {"A": a, alternative: x}
            for feature, values in tags.items():
                for value in values:
                    groups[row["parent_seed"], row["continuation_seed"], row["benchmark"], alternative, feature, value].append(item)
            if a != x:
                examples.append({"parent_seed": row["parent_seed"], "continuation_seed": row["continuation_seed"],
                                 "benchmark": row["benchmark"], "alternative": alternative,
                                 "kind": "rescue" if x else "risk", "problem_id": row["problem_id"], "question": row["question"]})
    table = []
    for key, records in sorted(groups.items()):
        metric = opportunity_metrics(records, ("A", key[3]), "A")[0]
        table.append({"parent_seed": key[0], "continuation_seed": key[1], "benchmark": key[2],
                      "feature": key[4], "value": key[5], "low_support": len(records) < 30, **metric})
    sampled = defaultdict(list)
    for row in sorted(examples, key=digest):
        key = tuple(row[k] for k in ("parent_seed", "continuation_seed", "benchmark", "alternative", "kind"))
        if len(sampled[key]) < 5:
            sampled[key].append(row)
    root = Path(output)
    write_csv(root / "patterns.csv", table)
    write_csv(root / "examples.csv", [r for group in sampled.values() for r in group])
    write(root / "summary.json", {"exploratory_only": True, "length_word_count_cuts": cuts,
          "feature_policy": "question-cues-v1", "low_support_below": 30,
          "limitations": "Tags are heuristic, may overlap, and do not establish adapter specialization."})
    seal(root, "patterns", [source], receipt["synthetic"])
    return {"groups": len(table), "exploratory_only": True}

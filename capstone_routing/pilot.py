"""Small, explicitly exploratory routing screen. Never labels final-test questions."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import math
from pathlib import Path
import subprocess
import sys

from .common import (EVALUATOR_SHA, PARSER_SHA, SETTINGS, bundle, checked, code_hashes,
                     destination, digest, identity, keyed, load_config, qkey, read,
                     require, rows, seal, sha, write, write_csv, write_rows)
from .inputs import load_registry, prediction_records
from .router import CS, THRESHOLDS, QuestionEncoder, choose, fit_binary, probability


def family_sample(questions, size, seed):
    """Sample whole families by question-only hashes, approximately balancing benchmarks."""
    require(type(size) is int and size >= 8 and size % 2 == 0, "Pilot sizes must be even integers >= 8")
    require(size <= len(questions), "Pilot size exceeds its original split")
    groups = defaultdict(list)
    for row in questions:
        groups[row["family_id"]].append(row)
    counts = Counter()
    selected = []
    for family in sorted(groups, key=lambda f: digest([seed, f])):
        group = groups[family]
        added = Counter(r["benchmark"] for r in group)
        before = sum(abs(counts[b] - size / 2) for b in ("gsm", "math"))
        after = sum(abs(counts[b] + added[b] - size / 2) for b in ("gsm", "math"))
        if after < before:
            selected.extend(group)
            counts.update(added)
    require(all(counts[b] > 0 for b in ("gsm", "math")), "Family structure prevents a balanced pilot")
    return sorted(selected, key=lambda q: digest([seed, qkey(q)]))


def prepare(registry_root, pool_root, output, train_size=300, validation_size=150,
            parent=17, seed=42, minimum_gain_pp=2.):
    from .pool import pool_questions
    require(0 < minimum_gain_pp <= 100, "Minimum useful pilot gain must be in (0,100]")
    registry = load_registry(registry_root)
    pool_receipt = bundle(pool_root, "pool")
    require(registry["synthetic"] == pool_receipt["synthetic"], "Mixed synthetic and real pilot inputs")
    policy = [identity(parent, m, seed) for m in ("A", "B", "C1")]
    seeds = [identity(parent, "A", s) for s in (42, 137, 2026)]
    ids = sorted(set(policy + seeds))
    candidates = {cid: registry["candidates"][cid] for cid in ids}
    require(len(ids) == 5 and len({c["parent_sha256"] for c in candidates.values()}) == 1,
            "Pilot requires five checkpoints from one actual parent")
    require(all(c.get("recipe") for c in candidates.values()) or registry["synthetic"],
            "Register loading proofs before preparing the pilot")
    root = destination(output, [registry_root, pool_root])
    require(not root.exists(), "Pilot plan exists; preserve it and select another destination")
    all_questions = pool_questions(pool_root, ("train", "validation"))
    selected = []
    for split, count in (("train", train_size), ("validation", validation_size)):
        subset = family_sample([q for q in all_questions if q["split"] == split], count, [42, split])
        grading = keyed(rows(Path(pool_root) / "grading" / (split + ".jsonl")))
        full = []
        for q in subset:
            g = grading[qkey(q)]
            require(g["question"] == q["question"] and g["split"] == split
                    and g.get("gold_answer") not in (None, ""), "Pilot grading/question mismatch")
            full.append(g)
        write_rows(root / (split + ".questions.jsonl"), subset)
        write_rows(root / "grading" / (split + ".jsonl"), full)
        selected.extend(subset)
    require(len({q["family_id"] for q in selected if q["split"] == "train"}) >= 3,
            "At least three training families are required for cross-validation")
    plan = {"registry": str(Path(registry_root).resolve()), "pool": str(Path(pool_root).resolve()),
            "registry_receipt_sha256": sha(Path(registry_root) / "receipt.json"),
            "pool_receipt_sha256": sha(Path(pool_root) / "receipt.json"),
            "candidates": candidates, "reference": identity(parent, "A", seed),
            "pools": {"policy": policy, "seed_diversity": seeds}, "parent_seed": parent,
            "continuation_seed": seed, "sampling_seed": 42, "cv_folds": 3,
            "minimum_gain_pp": minimum_gain_pp, "test_questions_used": False,
            "requested_counts": {"train": train_size, "validation": validation_size},
            "actual_counts": dict(Counter(q["split"] for q in selected)),
            "benchmark_counts": {s: dict(Counter(q["benchmark"] for q in selected if q["split"] == s))
                                 for s in ("train", "validation")},
            "maximum_responses": len(selected) * len(ids),
            "setting_selection": "3-fold family-grouped OOF on training only; validation assessed once",
            "synthetic": registry["synthetic"]}
    write(root / "plan.json", plan)
    seal(root, "pilot_plan", [registry_root, pool_root], registry["synthetic"])
    return {"plan": str(root), "candidates": ids, **{k: plan[k] for k in
            ("actual_counts", "benchmark_counts", "maximum_responses", "test_questions_used", "setting_selection")}}


def load_plan(root):
    receipt = bundle(root, "pilot_plan")
    require(receipt["code"] == code_hashes(), "Pilot code changed after preparation; create a new plan")
    plan = read(Path(root) / "plan.json")
    for kind in ("registry", "pool"):
        checked(Path(plan[kind]) / "receipt.json", plan[kind + "_receipt_sha256"])
    features, grading = [], []
    for split in ("train", "validation"):
        qs = rows(Path(root) / (split + ".questions.jsonl"))
        gs = keyed(rows(Path(root) / "grading" / (split + ".jsonl")))
        require(set(keyed(qs)) == set(gs), "Pilot grading coverage changed")
        for q in qs:
            require(q["split"] == split and gs[qkey(q)]["split"] == split
                    and q["question"] == gs[qkey(q)]["question"], "Pilot question/split changed")
        features.extend(qs)
        grading.extend(gs.values())
    keyed(features)
    families = defaultdict(set)
    for q in features:
        families[q["family_id"]].add(q["split"])
    require(all(len(s) == 1 for s in families.values()), "Pilot family crosses splits")
    require(len(plan["candidates"]) == 5 and not plan["test_questions_used"], "Invalid pilot scope")
    return plan, features, keyed(grading), receipt


def native_chunk(directory, candidate):
    """Validate the complete original evidence before selecting any subset from it."""
    receipt = bundle(directory, "label_chunk")
    original = keyed(rows(Path(directory) / "questions.jsonl"))
    require(all(q.get("split") in ("train", "validation") for q in original.values()),
            "Final-test predictions cannot be reused by the pilot")
    spec = {"predictions": str(Path(directory) / "predictions.jsonl"),
            "config": str(Path(directory) / "predictions.config.json"),
            "questions_sha256": sha(Path(directory) / "questions.jsonl")}
    values, coverage = prediction_records(spec, candidate, original)
    raw = keyed(rows(spec["predictions"]))
    return original, values, raw, coverage, receipt


def reusable(roots, candidates, grading, synthetic):
    answers, evidence = {}, []
    stable_fields = ("raw_output", "parsed_answer", "parse_status", "correct", "generated_tokens", "finish_reason")
    seen = {}
    for root in roots:
        require(Path(root).is_dir(), f"Reuse directory does not exist: {root}")
        for cid, candidate in candidates.items():
            for receipt_path in sorted((Path(root) / cid / "chunks").glob("*/receipt.json")):
                directory = receipt_path.parent
                original, values, raw, coverage, receipt = native_chunk(directory, candidate)
                require(receipt["synthetic"] == synthetic, "Mixed synthetic and real reused answers")
                matched = 0
                for key in set(original) & set(grading) & set(raw):
                    require(all(original[key][f] == grading[key][f] for f in ("question", "gold_answer", "split")),
                            "Reused answer has another question/reference/split")
                    fingerprint = digest({f: raw[key].get(f) for f in stable_fields})
                    require((cid, key) not in seen or seen[cid, key] == fingerprint,
                            "Conflicting saved predictions; cannot select an attempt by correctness")
                    seen[cid, key] = fingerprint
                    answers[cid, key] = values[key]
                    matched += 1
                if matched:
                    evidence.append({"candidate": cid, "source_chunk": str(directory.resolve()),
                                     "source_receipt_sha256": sha(receipt_path), "matched_answers": matched,
                                     **coverage})
    return answers, evidence


def gpu_preflight():
    for package, field in (("torch", "torch"), ("transformers", "transformers"),
                           ("peft", "peft"), ("math-verify", "math_verify")):
        require(version(package) == SETTINGS[field], f"Wrong native environment: {package}")
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.version.cuda == SETTINGS["cuda"] and torch.cuda.get_device_name(0) == SETTINGS["gpu"],
            "Pilot requires one allocated GPU matching the frozen A100/CUDA runtime")


def label(plan_root, cfg, output, reuse_roots=(), authorize_gpu=False):
    plan, features, grading, receipt = load_plan(plan_root)
    candidates = load_registry(plan["registry"])["candidates"]
    require(all(candidates[cid] == cp for cid, cp in plan["candidates"].items()), "Pilot checkpoint lineage changed")
    candidates = plan["candidates"]
    root = destination(output, [plan_root, *reuse_roots, cfg["evaluator"]["path"], cfg["evaluator"]["parser_path"]])
    if (root / "receipt.json").exists():
        bundle(root, "pilot_labels")
        meta = read(root / "labels.json")
        require(meta["plan_receipt_sha256"] == sha(Path(plan_root) / "receipt.json"), "Labels use another pilot plan")
        return meta
    require(not root.exists(), "Interrupted pilot preserved; choose a new output and reuse the previous pilot labels")
    evaluator = checked(cfg["evaluator"]["path"], EVALUATOR_SHA)
    parser_path = checked(cfg["evaluator"]["parser_path"], PARSER_SHA)
    require(parser_path == evaluator.parent / "answer_parser.py", "Native parser must be the verified sibling")
    answers, sources = reusable(reuse_roots, candidates, grading, receipt["synthetic"])
    reused_count = len(answers)
    missing = [(cid, qkey(q)) for cid in candidates for q in features if (cid, qkey(q)) not in answers]
    require(authorize_gpu or not missing, "Missing pilot answers require explicit --authorize-gpu")
    if missing:
        gpu_preflight()
    generated = 0
    from .live import verify_candidate
    for cid, candidate in candidates.items():
        pending = [q for q in features if (cid, qkey(q)) not in answers]
        if pending:
            verify_candidate(candidate)
        for bench in ("gsm", "math"):
            subset = [grading[qkey(q)] for q in pending if q["benchmark"] == bench]
            for offset in range(0, len(subset), 128):
                chunk = subset[offset:offset + 128]
                directory = root / cid / "chunks" / f"{bench}_{offset:05d}"
                write_rows(directory / "questions.jsonl", chunk)
                predictions = directory / "predictions.jsonl"
                command = [sys.executable, "-B", str(evaluator), "--eval_file", str(directory / "questions.jsonl"),
                           "--out", str(predictions), "--run_id", f"routing_pilot_{cid}_{bench}_{offset}",
                           "--model", candidate["model"], "--revision", candidate["revision"],
                           "--adapter", candidate["checkpoint"], "--expect_train_manifest_sha256", candidate["train_manifest_sha256"],
                           "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024", "--question_text", "p1_clean"]
                print(f"Pilot generation: {cid}/{bench}, {len(chunk)} questions", flush=True)
                subprocess.run(command, check=True)
                spec = {"predictions": str(predictions), "config": str(directory / "predictions.config.json"),
                        "questions_sha256": sha(directory / "questions.jsonl")}
                values, coverage = prediction_records(spec, candidate, keyed(chunk))
                require(coverage["missing_rows"] == 0, "Incomplete native pilot output")
                seal(directory, "label_chunk", synthetic=receipt["synthetic"])
                answers.update({(cid, key): value for key, value in values.items()})
                generated += len(chunk)
    require(len(answers) == len(features) * 5, "Incomplete pilot answer matrix")
    write_rows(root / "outcomes.jsonl", [{"candidate": cid, "benchmark": q["benchmark"], "problem_id": q["problem_id"],
               "split": q["split"], "question_sha256": digest(q["question"]), "correct": answers[cid, qkey(q)]}
               for cid in candidates for q in features])
    meta = {"complete": True, "plan_receipt_sha256": sha(Path(plan_root) / "receipt.json"),
            "checkpoint_hashes": {cid: c["checkpoint_sha256"] for cid, c in candidates.items()},
            "total_answers": len(answers), "reused_answers": reused_count, "generated_answers": generated,
            "reuse_sources": sources, "test_questions_used": False}
    write(root / "labels.json", meta)
    seal(root, "pilot_labels", [plan_root], receipt["synthetic"])
    return meta


def family_folds(questions, folds=3):
    groups = defaultdict(list)
    for i, q in enumerate(questions):
        groups[q["family_id"]].append(i)
    require(len(groups) >= folds, "Insufficient independent training families")
    assigned, sizes = {}, [0] * folds
    for f in sorted(groups, key=lambda f: (-len(groups[f]), digest([42, f]))):
        fold = min(range(folds), key=lambda k: (sizes[k], k))
        assigned[f] = fold
        sizes[fold] += len(groups[f])
    return [assigned[q["family_id"]] for q in questions]


def select_settings(questions, vectors, labels, plan):
    folds = family_folds(questions, plan["cv_folds"])
    ids = list(plan["candidates"])
    scores = {}
    for c in CS:
        scores[c] = [{} for _ in questions]
        for fold in range(plan["cv_folds"]):
            train_ids = [i for i, f in enumerate(folds) if f != fold]
            for cid in ids:
                model = fit_binary([vectors[i] for i in train_ids], [labels[cid][i] for i in train_ids], c)
                for i, f in enumerate(folds):
                    if f == fold:
                        scores[c][i][cid] = probability(model, vectors[i])
    selections, trials = {}, []
    for name, pool in plan["pools"].items():
        options = []
        for c in CS:
            for threshold in THRESHOLDS:
                choices = [choose(s, plan["reference"], pool, threshold) for s in scores[c]]
                result = {"pool": name, "c": c, "threshold": threshold,
                          "oof_accuracy": sum(labels[cid][i] for i, cid in enumerate(choices)) / len(questions),
                          "switches": sum(cid != plan["reference"] for cid in choices)}
                options.append(result)
                trials.append(result)
        best = min(options, key=lambda r: (-r["oof_accuracy"], r["switches"], r["c"], r["threshold"]))
        fixed = min(pool, key=lambda cid: (-sum(labels[cid]), cid != plan["reference"], cid))
        selections[name] = {**best, "candidates": pool, "best_fixed": fixed,
                            "models": {cid: fit_binary(vectors, labels[cid], best["c"]) for cid in pool}}
    return selections, trials, folds


def fit(plan_root, labels_root, output, encoder=None, resamples=2000, permutations=1000):
    from .evaluation import paired_bootstrap, random_control, switch_counts
    plan, questions, _, receipt = load_plan(plan_root)
    root = destination(output, [plan_root, labels_root])
    if (root / "receipt.json").exists():
        bundle(root, "pilot_report")
        meta = read(root / "summary.json")
        require(meta["plan_receipt_sha256"] == sha(Path(plan_root) / "receipt.json")
                and meta["labels_receipt_sha256"] == sha(Path(labels_root) / "receipt.json"), "Report uses another pilot")
        return meta
    require(not root.exists(), "Interrupted pilot report preserved; choose a new output")
    label_receipt = bundle(labels_root, "pilot_labels")
    meta = read(Path(labels_root) / "labels.json")
    require(label_receipt["synthetic"] == receipt["synthetic"]
            and meta["plan_receipt_sha256"] == sha(Path(plan_root) / "receipt.json")
            and meta["checkpoint_hashes"] == {cid: c["checkpoint_sha256"] for cid, c in plan["candidates"].items()},
            "Pilot labels use another plan/checkpoint set")
    outcomes = rows(Path(labels_root) / "outcomes.jsonl")
    lookup = {}
    expected = {(cid, qkey(q)) for cid in plan["candidates"] for q in questions}
    feature_map = keyed(questions)
    for row in outcomes:
        key = row["candidate"], qkey(row)
        require(key in expected and key not in lookup and type(row["correct"]) is int and row["correct"] in (0, 1),
                "Duplicate/invalid pilot outcome")
        q = feature_map[qkey(row)]
        require(row["question_sha256"] == digest(q["question"]) and row["split"] == q["split"], "Stale pilot label")
        lookup[key] = row["correct"]
    require(set(lookup) == expected, "Incomplete pilot labels")
    require(encoder is None or receipt["synthetic"], "Real pilot requires the pinned question encoder")
    encoder = encoder or QuestionEncoder()
    vectors = {qkey(q): encoder.encode(q["question"]) for q in questions}
    require(all(len(v) == 384 and all(math.isfinite(x) for x in v) for v in vectors.values()), "Invalid pilot embeddings")
    train_q = [q for q in questions if q["split"] == "train"]
    validation = [q for q in questions if q["split"] == "validation"]
    y_train = {cid: [lookup[cid, qkey(q)] for q in train_q] for cid in plan["candidates"]}
    selections, trials, folds = select_settings(train_q, [vectors[qkey(q)] for q in train_q], y_train, plan)
    # Persist all setting choices before consulting validation correctness.
    write(root / "selection.json", selections)
    write_csv(root / "training_cv_trials.csv", trials)
    write_rows(root / "training_folds.jsonl", [{**q, "fold": f} for q, f in zip(train_q, folds)])
    write_rows(root / "embeddings.jsonl", [{"benchmark": q["benchmark"], "problem_id": q["problem_id"],
               "question_sha256": digest(q["question"]), "vector": vectors[qkey(q)]} for q in questions])
    from .common import ENCODER, ENCODER_REVISION
    write(root / "encoder.json", {"model": ENCODER, "revision": ENCODER_REVISION,
          "policy": "token-chunks-weighted-l2-v1", "synthetic_encoder": receipt["synthetic"],
          "packages": {k: version(k) for k in ("numpy", "scipy", "scikit-learn", "torch", "sentence-transformers", "transformers")}
                      if not receipt["synthetic"] else {"evidence": "SYNTHETIC TEST"}})
    y_val = {cid: [lookup[cid, qkey(q)] for q in validation] for cid in plan["candidates"]}
    reference = y_val[plan["reference"]]
    metrics, switches, contrasts, routed, opportunity, random_reports = [], [], {}, {}, [], {}
    for name, selection in selections.items():
        pool = selection["candidates"]
        scores = [{cid: probability(model, vectors[qkey(q)]) for cid, model in selection["models"].items()} for q in validation]
        choices = [choose(s, plan["reference"], pool, selection["threshold"]) for s in scores]
        chosen = [y_val[cid][i] for i, cid in enumerate(choices)]
        fixed = y_val[selection["best_fixed"]]
        oracle = [max(y_val[cid][i] for cid in pool) for i in range(len(validation))]
        random_expected, random_report = random_control({cid: y_val[cid] for cid in pool}, choices, permutations)
        random_reports[name] = random_report
        routed[name] = chosen
        contrasts[name + "/A"] = [x - a for x, a in zip(chosen, reference)]
        contrasts[name + "/fixed"] = [x - a for x, a in zip(chosen, fixed)]
        contrasts[name + "/random_expected"] = [x - a for x, a in zip(chosen, random_expected)]
        for bench in ("gsm", "math", "combined"):
            indexes = [i for i, q in enumerate(validation) if bench == "combined" or q["benchmark"] == bench]
            avg = lambda values: sum(values[i] for i in indexes) / len(indexes)
            counts = switch_counts([reference[i] for i in indexes], [chosen[i] for i in indexes],
                                   [choices[i] for i in indexes], plan["reference"])
            metrics.append({"pool": name, "benchmark": bench, **counts,
                "best_fixed_checkpoint": selection["best_fixed"], "best_fixed_accuracy": avg(fixed),
                "oracle_accuracy": avg(oracle), "available_gain_pp": 100 * (avg(oracle) - avg(reference)),
                "random_expected_accuracy": avg(random_expected), "gain_over_fixed_pp": 100 * (avg(chosen) - avg(fixed)),
                "gain_over_random_pp": 100 * (avg(chosen) - avg(random_expected))})
        for i, q in enumerate(validation):
            switches.append({"pool": name, **q, "routing_scores": scores[i], "selected_checkpoint": choices[i],
                "selected_correct": chosen[i], "A_correct": reference[i],
                "switch_kind": "stay" if choices[i] == plan["reference"] else "successful" if chosen[i] and not reference[i]
                               else "harmful" if reference[i] and not chosen[i] else "neutral"})
        for cid in pool:
            alternative = y_val[cid]
            opportunity.append({"pool": name, "candidate": cid, "n": len(validation),
                "accuracy": sum(alternative) / len(validation),
                "rescues": sum(not a and x for a, x in zip(reference, alternative)),
                "risks": sum(a and not x for a, x in zip(reference, alternative)),
                "both_correct": sum(a and x for a, x in zip(reference, alternative)),
                "both_wrong": sum(not a and not x for a, x in zip(reference, alternative))})
    contrasts["policy_minus_seed"] = [x - y for x, y in zip(routed["policy"], routed["seed_diversity"])]
    intervals = paired_bootstrap(validation, contrasts, resamples)
    for interval in intervals.values():
        interval["scope"] = "pilot validation families; settings chosen on training OOF; fixed checkpoints; fitting uncertainty excluded"
    combined = [m for m in metrics if m["benchmark"] == "combined"]
    promising = [m["pool"] for m in combined if m["difference_pp"] >= plan["minimum_gain_pp"]
                 and m["gain_over_fixed_pp"] > 0 and m["gain_over_random_pp"] > 0]
    assessment = "promising_pilot_signal" if promising else "no_clear_pilot_signal"
    limits = ["Exploratory screening of one matched block; no final-test result or variation across blocks.",
              "Router C and switching threshold chosen by training-only family-grouped OOF predictions.",
              "Best fixed adapter chosen on training questions; validation labels were reserved for assessment.",
              "A negative small pilot cannot rule out modest gains or a stronger question-only router.",
              "Bootstrap intervals condition on these fitted routers and checkpoints; they exclude fitting uncertainty.",
              "Policy and A-seed pools each contain three checkpoints and share questions, features, and inference.",
              "Inspect checkpoint construction differences before attributing any gain to selection policies.",
              "This pilot never starts a full experiment automatically. Keep the original final test untouched."]
    summary = {"complete": True, "exploratory_only": True, "synthetic": receipt["synthetic"],
        "plan_receipt_sha256": sha(Path(plan_root) / "receipt.json"),
        "labels_receipt_sha256": sha(Path(labels_root) / "receipt.json"),
        "counts": plan["actual_counts"], "candidates": list(plan["candidates"]),
        "generated_answers": meta["generated_answers"], "reused_answers": meta["reused_answers"],
        "assessment": assessment, "promising_pools": promising, "minimum_gain_pp": plan["minimum_gain_pp"],
        "metrics": combined, "paired_intervals": intervals, "test_questions_used": False,
        "full_experiment_started": False, "limitations": limits,
        "construction": {cid: {k: cp.get(k) for k in ("parent_sha256", "continuation_seed", "actual_tokens", "construction")}
                         for cid, cp in plan["candidates"].items()}}
    write_csv(root / "validation_metrics.csv", metrics)
    write_csv(root / "rescue_risk.csv", opportunity)
    write_csv(root / "paired_intervals.csv", [{"contrast": k, **v} for k, v in intervals.items()])
    write_rows(root / "validation_switches.jsonl", switches)
    write_rows(root / "correctness_matrix.jsonl", [{**q, **{cid: y_val[cid][i] for cid in plan["candidates"]}}
                                                  for i, q in enumerate(validation)])
    write(root / "random_controls.json", random_reports)
    write(root / "summary.json", summary)
    lines = ["# Small routing pilot", "", "Evidence: " + ("SYNTHETIC TEST" if receipt["synthetic"] else "recorded SOL predictions"),
        "", f"Assessment: **{assessment}**. This is a screening result, not a final-test claim.", "",
        "Settings selected with 3-fold family-grouped training predictions; validation assessed afterward.",
        "Best fixed checkpoint selected on training. Final-test questions were not evaluated.", "",
        "| Pool | A % | Router % | Fixed % | Random % | Oracle % | Gain over A (pp) | Success / harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in combined:
        lines.append(f"| {m['pool']} | {100*m['reference_accuracy']:.2f} | {100*m['router_accuracy']:.2f} | "
            f"{100*m['best_fixed_accuracy']:.2f} | {100*m['random_expected_accuracy']:.2f} | {100*m['oracle_accuracy']:.2f} | "
            f"{m['difference_pp']:.2f} | {m['successful_switches']} / {m['harmful_switches']} |")
    lines += ["", f"Promising means at least {plan['minimum_gain_pp']:.1f} pp over A and positive gains over the training-selected fixed adapter and matched-frequency random routing.",
              "This is a predeclared screening flag, not a significance test or an instruction to run the full experiment.", ""]
    for name in plan["pools"]:
        interval = intervals[name + "/A"]
        lines.append(f"{name}: router-minus-A paired family-bootstrap 95% interval "
                     f"[{interval['ci95_low_pp']:.2f}, {interval['ci95_high_pp']:.2f}] pp.")
    lines += ["", "Policy-minus-seed router difference: " + f"{intervals['policy_minus_seed']['difference_pp']:.2f} pp.",
              "", *["- " + text for text in limits], ""]
    from .common import write_bytes
    write_bytes(root / "report.md", "\n".join(lines).encode())
    seal(root, "pilot_report", [plan_root, labels_root], receipt["synthetic"])
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for flag in ("registry", "pool", "output"):
        prep.add_argument("--" + flag, required=True)
    prep.add_argument("--train-size", type=int, default=300)
    prep.add_argument("--validation-size", type=int, default=150)
    prep.add_argument("--parent", type=int, default=17, choices=(17, 123))
    prep.add_argument("--seed", type=int, default=42, choices=(42, 137, 2026))
    prep.add_argument("--minimum-gain-pp", type=float, default=2.)
    lab = commands.add_parser("label")
    for flag in ("plan", "config", "output"):
        lab.add_argument("--" + flag, required=True)
    lab.add_argument("--reuse-labels", action="append", default=[])
    lab.add_argument("--authorize-gpu", action="store_true")
    fitting = commands.add_parser("fit")
    for flag in ("plan", "labels", "output"):
        fitting.add_argument("--" + flag, required=True)
    return ap


def main():
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.registry, args.pool, args.output, args.train_size, args.validation_size,
                             args.parent, args.seed, args.minimum_gain_pp)
        elif args.command == "label":
            result = label(args.plan, load_config(args.config), args.output, args.reuse_labels, args.authorize_gpu)
        else:
            result = fit(args.plan, args.labels, args.output)
        import json
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        import json
        print(json.dumps({"complete": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

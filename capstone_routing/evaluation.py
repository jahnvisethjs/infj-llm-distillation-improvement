"""Paired held-out switch accounting and diversity controls; never fits a model."""

from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

from .common import (PARENTS, SEEDS, bundle, qkey, read, require, seal, sha,
                     write, write_bytes, write_csv, write_rows, code_hashes)
from .labeling import depends_on
from .pool import pool_questions
from .router import choose, load_embeddings, load_labels, probability, scenarios


def switch_counts(reference, selected, choices, reference_id):
    require(len(reference) == len(selected) == len(choices) and reference, "Switch accounting requires paired questions")
    success = sum(not a and x and c != reference_id for a, x, c in zip(reference, selected, choices))
    harm = sum(a and not x and c != reference_id for a, x, c in zip(reference, selected, choices))
    delta = sum(selected) - sum(reference)
    require(delta == success - harm, "Switch identity failed")
    return {"n": len(reference), "reference_accuracy": statistics.mean(reference),
            "router_accuracy": statistics.mean(selected), "successful_switches": success,
            "harmful_switches": harm, "switches": sum(c != reference_id for c in choices),
            "difference_pp": 100 * delta / len(reference)}


def random_control(labels, choices, permutations=1000, seed=0):
    require(choices and permutations > 0 and all(len(v) == len(choices) for v in labels.values()), "Invalid random control inputs")
    frequencies = Counter(choices)
    expected = [sum(n / len(choices) * labels[c][i] for c, n in frequencies.items()) for i in range(len(choices))]
    rng = random.Random(seed)
    sampled = []
    for _ in range(permutations):
        assignment = list(choices)
        rng.shuffle(assignment)
        require(Counter(assignment) == frequencies, "Random routing changed selection counts")
        sampled.append(sum(labels[c][i] for i, c in enumerate(assignment)) / len(choices))
    return expected, {"expected_accuracy": statistics.mean(expected),
                      "permutation_mean_accuracy": statistics.mean(sampled),
                      "permutation_sd": statistics.stdev(sampled) if permutations > 1 else 0.,
                      "permutations": permutations, "seed": seed, "selection_counts": dict(frequencies)}


def paired_bootstrap(questions, contrasts, resamples=2000, seed=0):
    """One shared family draw across all contrast columns; checkpoint reuse is retained."""
    import numpy as np
    require(questions and resamples > 0 and all(len(v) == len(questions) for v in contrasts.values()), "Invalid paired bootstrap")
    groups = defaultdict(list)
    for i, row in enumerate(questions):
        require(row.get("family_id"), "Bootstrap requires question families")
        groups[row["family_id"]].append(i)
    names = sorted(contrasts)
    values = np.asarray([contrasts[name] for name in names], dtype=float).T
    indexes = list(groups.values())
    sums = np.asarray([values[ids].sum(axis=0) for ids in indexes])
    counts = np.asarray([len(ids) for ids in indexes])
    rng = random.Random(seed)
    sampled = []
    for _ in range(resamples):
        draw = [rng.randrange(len(indexes)) for _ in indexes]
        sampled.append(100 * sums[draw].sum(axis=0) / counts[draw].sum())
    sampled = np.asarray(sampled)
    low, high = np.quantile(sampled, [.025, .975], axis=0)
    return {name: {"difference_pp": float(100 * values[:, j].mean()), "ci95_low_pp": float(low[j]),
                   "ci95_high_pp": float(high[j]), "family_count": len(groups), "resamples": resamples,
                   "seed": seed, "scope": "paired question families, fixed checkpoints and frozen routers"}
            for j, name in enumerate(names)}


def evaluate_test(frozen_root, pool_root, embeddings_root, test_labels, output, resamples=2000, permutations=1000):
    receipt = bundle(frozen_root, "frozen")
    require(receipt["code"] == code_hashes(), "Routing code changed after freezing; create a new experiment")
    require(depends_on(frozen_root, pool_root) and depends_on(frozen_root, embeddings_root), "Frozen experiment uses other inputs")
    root = Path(output)
    require(not root.exists(), "Final evaluation output exists; preserve it and use report to reread results")
    questions = pool_questions(pool_root, ("test",))
    vectors = load_embeddings(embeddings_root, questions)
    routers = read(Path(frozen_root) / "routers.json")
    require({r["id"] for r in routers} == {r["id"] for r in scenarios()}, "Incomplete frozen router set")
    candidates = read(Path(frozen_root) / "registry.json")["candidates"]
    labels, metadata, label_receipt = load_labels(test_labels, "test", questions, sorted(candidates))
    require(label_receipt["synthetic"] == receipt["synthetic"], "Mixed real and synthetic evidence")
    require(metadata["frozen_receipt_sha256"] == sha(Path(frozen_root) / "receipt.json")
            and metadata["pool_receipt_sha256"] == sha(Path(pool_root) / "receipt.json")
            and metadata["checkpoint_hashes"] == {cid: c["checkpoint_sha256"] for cid, c in candidates.items()},
            "Test labels belong to another freeze/pool/checkpoint set")
    metrics, details, random_reports, results, all_contrasts = [], [], [], {}, {}
    for router in routers:
        rid, ref = router["id"], router["reference"]
        scores = [{cid: probability(model, vectors[qkey(q)]) for cid, model in router["models"].items()} for q in questions]
        choices = [choose(s, ref, router["pool"], router["threshold"]) for s in scores]
        reference = [labels[ref][qkey(q)] for q in questions]
        chosen = [labels[c][qkey(q)] for q, c in zip(questions, choices)]
        fixed = [labels[router["best_fixed"]][qkey(q)] for q in questions]
        oracle = [max(labels[c][qkey(q)] for c in router["candidates"]) for q in questions]
        random_expected, random_report = random_control(
            {cid: [labels[cid][qkey(q)] for q in questions] for cid in router["pool"]}, choices, permutations)
        random_reports.append({"scenario": rid, **random_report})
        results[rid] = chosen
        all_contrasts[rid + "/A"] = [x - a for x, a in zip(chosen, reference)]
        all_contrasts[rid + "/fixed"] = [x - a for x, a in zip(chosen, fixed)]
        all_contrasts[rid + "/random_expected"] = [x - a for x, a in zip(chosen, random_expected)]
        for bench in ("gsm", "math", "combined"):
            ids = [i for i, q in enumerate(questions) if bench == "combined" or q["benchmark"] == bench]
            counts = switch_counts([reference[i] for i in ids], [chosen[i] for i in ids], [choices[i] for i in ids], ref)
            metrics.append({"scenario": rid, "experiment": router["experiment"], "parent_seed": router["parent_seed"],
                            "continuation_seed": router["continuation_seed"], "benchmark": bench, **counts,
                            "best_fixed_checkpoint": router["best_fixed"], "best_fixed_accuracy": statistics.mean(fixed[i] for i in ids),
                            "oracle_accuracy": statistics.mean(oracle[i] for i in ids),
                            "random_expected_accuracy": statistics.mean(random_expected[i] for i in ids),
                            "available_improvement_pp": 100 * statistics.mean(oracle[i] - reference[i] for i in ids)})
        for i, q in enumerate(questions):
            details.append({"scenario": rid, **q, "routing_scores": scores[i], "selected_checkpoint": choices[i],
                            "selected_correct": chosen[i], "reference_correct": reference[i],
                            "switch_kind": "stay" if choices[i] == ref else "successful" if chosen[i] and not reference[i]
                            else "harmful" if reference[i] and not chosen[i] else "neutral"})
    primary = [k for k in results if k.startswith("primary_")]
    all_contrasts["primary_mean/A"] = [statistics.mean(all_contrasts[k + "/A"][i] for k in primary) for i in range(len(questions))]
    control_names = []
    for parent in PARENTS:
        for seed in SEEDS:
            name = f"policy_minus_seed_p{parent}_t{seed}"
            left, right = results[f"policy_three_p{parent}_t{seed}"], results[f"seed_three_p{parent}_t{seed}"]
            all_contrasts[name] = [a - b for a, b in zip(left, right)]
            control_names.append(name)
        all_contrasts[f"policy_minus_seed_p{parent}_mean"] = [statistics.mean(
            all_contrasts[f"policy_minus_seed_p{parent}_t{s}"][i] for s in SEEDS) for i in range(len(questions))]
    all_contrasts["policy_minus_seed_mean"] = [statistics.mean(all_contrasts[k][i] for k in control_names) for i in range(len(questions))]
    intervals = []
    for bench in ("gsm", "math", "combined"):
        ids = [i for i, q in enumerate(questions) if bench == "combined" or q["benchmark"] == bench]
        estimates = paired_bootstrap([questions[i] for i in ids], {k: [v[i] for i in ids] for k, v in all_contrasts.items()}, resamples)
        intervals.extend({"contrast": k, "benchmark": bench, **v} for k, v in estimates.items())
    variation = []
    for experiment in ("primary", "policy_three", "seed_three"):
        for parent in PARENTS:
            for bench in ("gsm", "math", "combined"):
                values = [m["difference_pp"] for m in metrics if m["experiment"] == experiment
                          and m["parent_seed"] == parent and m["benchmark"] == bench]
                variation.append({"experiment": experiment, "parent_seed": parent, "benchmark": bench,
                                  "mean_difference_pp": statistics.mean(values), "sample_sd_pp": statistics.stdev(values),
                                  "scope": "descriptive reference-seed variation; seed-three reuses checkpoints"})
    write_csv(root / "metrics.csv", metrics)
    write_rows(root / "switches.jsonl", details)
    write_csv(root / "paired_intervals.csv", intervals)
    write_csv(root / "seed_variation.csv", variation)
    write(root / "random_controls.json", random_reports)
    summary = {"complete": True, "synthetic": receipt["synthetic"], "test_questions": len(questions),
               "primary": next(r for r in intervals if r["contrast"] == "primary_mean/A" and r["benchmark"] == "combined"),
               "secondary": next(r for r in intervals if r["contrast"] == "policy_minus_seed_mean" and r["benchmark"] == "combined"),
               "construction": {cid: {k: cp[k] for k in ("parent_sha256", "continuation_seed", "actual_tokens", "construction")}
                                for cid, cp in candidates.items()},
               "limitations": ["Oracle opportunity is an upper bound, not a learned gain.",
                   "The same questions and A checkpoints recur across blocks; they are not independent observations.",
                   "A-only controls vary continuation seed; policy controls vary data policy at a matched seed.",
                   "Unmatched checkpoint construction differences limit policy attribution.",
                   "A failed simple router does not establish that question-only routing is impossible.",
                   "A final test is valid only if experiment selection remains frozen after this report."],
               "official_final_tests_performed": False}
    write(root / "summary.json", summary)
    seal(root, "routing_test", [frozen_root, pool_root, embeddings_root, test_labels], receipt["synthetic"])
    return summary


def report(opportunity_root, output, test_root=None):
    opportunity_receipt = bundle(opportunity_root, "opportunity")
    summary = read(Path(opportunity_root) / "summary.json")
    lines = ["# Adapter routing evidence", "", "Evidence: " + ("SYNTHETIC" if opportunity_receipt["synthetic"] else "recorded SOL predictions"),
             "", f"Complete development comparisons: {summary['complete_block_count']}/12 (six policy, six shared A-only controls).",
             "", "Development patterns are exploratory. Oracle gains describe available, not achievable, improvement.", "",
             "| Pool | Parent | Benchmark | Mean oracle gain over reference A (pp) | Complete blocks |",
             "|---|---:|---|---:|---:|"]
    for row in summary["summaries"]:
        lines.append(f"| {row['experiment']} | {row['parent_seed']} | {row['benchmark']} | {row['mean_available_improvement_pp']:.3f} | {row['complete_blocks']} |")
    upstream = [opportunity_root]
    if test_root:
        test_receipt = bundle(test_root, "routing_test")
        require(test_receipt["synthetic"] == opportunity_receipt["synthetic"], "Do not combine synthetic and real evidence")
        test = read(Path(test_root) / "summary.json")
        lines += ["", "## Frozen routing test", ""]
        for label, key in (("Router minus A", "primary"), ("Three-policy router minus A-only seed router", "secondary")):
            value = test[key]
            lines.append(f"{label}: {value['difference_pp']:.3f} pp; paired family-bootstrap 95% CI [{value['ci95_low_pp']:.3f}, {value['ci95_high_pp']:.3f}].")
        lines += ["", *test["limitations"]]
        upstream.append(test_root)
    lines += ["", "## Interpretation", "", "- Small oracle gains imply little opportunity in the recorded pool.",
              "- Oracle gains without held-out router gains leave predictability unproven.",
              "- Validation-only gains indicate failed generalization.",
              "- Consistent untouched-test gains support useful routing; compare seed diversity before attributing them to policy diversity.", ""]
    root = Path(output)
    write_bytes(root / "report.md", "\n".join(lines).encode())
    seal(root, "report", upstream, opportunity_receipt["synthetic"])
    return {"report": str(root / "report.md")}

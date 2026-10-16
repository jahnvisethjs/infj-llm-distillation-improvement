"""
Export finished evaluation results for Person 4 (analysis). Dev sets only; test results are never exported here.

  python export_for_analysis.py --runs $R/runs.json --results $R            # writes $R/analysis/

Writes
  analysis/per_problem.csv   one row per (run, benchmark, problem): for paired tests and changed-answer analysis
  analysis/run_ledger.csv    one row per run, column names from the working doc's ledger template
                             (Person 3's columns; Person 2 / Person 4 fill training and selection columns)
  analysis/README.txt        what every column means

Run labels come from runs.json: budget_examples, budget_tokens, seed, arm, parent_id, method, baseline.
Missing labels are left empty, never guessed. Only complete runs (all 500 problems) are exported.
"""
from __future__ import annotations

import argparse, csv, json, os
from collections import Counter

from metrics import load, summarize

DEV = {"gsm": "data/dev_gsm.jsonl", "math": "data/dev_math.jsonl"}
LABELS = ["budget_examples", "budget_tokens", "seed", "arm", "parent_id", "method", "phase", "baseline", "note"]


def n_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for l in f if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="default: <results>/analysis")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "analysis")
    os.makedirs(out, exist_ok=True)

    runs = json.load(open(a.runs))
    problem_rows, ledger, skipped = [], [], []
    for run in runs:
        files = {b: f"{a.results}/{run['run_id']}_{b}.jsonl" for b in DEV}
        if any(n_lines(f) < n_lines(DEV[b]) for b, f in files.items()):
            skipped.append(run["run_id"]); continue
        labels = {k: run.get(k, "") if run.get(k) is not None else "" for k in LABELS}
        size = run["model"].split("-")[-1]
        L = {"run_id": run["run_id"], "model": size, "is_baseline": not run.get("adapter"), **labels}
        provenance = set()
        for bench, f in files.items():
            rows = load(f)
            s = summarize(rows)
            cfg_path = f.replace(".jsonl", ".config.json")
            cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
            seconds = (cfg.get("load_seconds", 0) + cfg["steady_eval_seconds"]
                       if "steady_eval_seconds" in cfg else s["eval_seconds"])
            for key in ("parser_version", "eval_config_hash", "checkpoint_sha256", "model_revision"):
                vals = {r.get(key) for r in rows}
                if len(vals) != 1:
                    raise SystemExit(f"STOP: {f} mixes several {key} values; re-grade or re-run it first")
            r0 = rows[0]
            provenance.add((r0["parser_version"], r0["model_revision"], str(r0["checkpoint_sha256"])))
            L.update({
                f"{bench}_dev_correct": sum(r["correct"] for r in rows),
                f"{bench}_dev_n": s["n"],
                f"{bench}_dev_accuracy": round(s["accuracy"], 4),
                f"{bench}_valid_answer_rate": round(s["valid_answer_rate"], 4),
                f"{bench}_accuracy_among_valid": round(s["accuracy_among_valid"], 4),
                f"{bench}_length_limit_rate": round(s["length_limit_rate"], 4),
                f"{bench}_mean_generated_tokens": round(s["gen_tokens_mean"], 1),
                f"{bench}_repetition_rate": round(s["repetition_rate(full_output, tokens)"], 4),
                f"{bench}_evaluation_gpu_hours": round(seconds / 3600, 4),
                f"{bench}_predictions_path": os.path.abspath(f),
                f"{bench}_eval_config_hash": r0["eval_config_hash"],
            })
            for r in rows:
                problem_rows.append({
                    "run_id": run["run_id"], "model": size, "is_baseline": not run.get("adapter"),
                    **{k: labels[k] for k in ("budget_examples", "budget_tokens", "seed", "arm", "parent_id")},
                    "benchmark": bench, "problem_id": r["problem_id"],
                    "correct": int(r["correct"]), "valid": int(r["parse_status"] == "ok"),
                    "parse_status": r["parse_status"], "parsed_answer": r["parsed_answer"],
                    "gold_answer": r["gold_answer"], "generated_tokens": r["generated_tokens"],
                    "hit_length_limit": int(r["finish_reason"] == "length"),
                })
        (parser, revision, ckpt), = provenance if len(provenance) == 1 else [(None, None, None)]
        if parser is None:
            raise SystemExit(f"STOP: {run['run_id']}: GSM and MATH files differ in grader/model/checkpoint")
        L.update(parser_version=parser, model_revision=revision, checkpoint_sha256=ckpt,
                 evaluation_gpu_hours=round(L["gsm_evaluation_gpu_hours"] + L["math_evaluation_gpu_hours"], 4))
        ledger.append(L)

    if not ledger:
        raise SystemExit("no complete runs to export")
    for name, rows in (("per_problem.csv", problem_rows), ("run_ledger.csv", ledger)):
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # budget_tokens is needed for every fine-tuned run; arm/parent_id only for continuation runs (phase != curve)
    missing = Counter(k for r in ledger if not r["is_baseline"]
                      for k in (("budget_tokens",) + (() if r["phase"] in ("", "curve") else ("arm", "parent_id")))
                      if r[k] == "")
    with open(os.path.join(out, "README.txt"), "w") as fh:
        fh.write(README)
    print(f"exported {len(ledger)} runs, {len(problem_rows)} problem rows -> {out}/")
    if skipped:
        print("not exported (unfinished):", ", ".join(skipped))
    for k, n in missing.items():
        print(f"NOTE: {n} fine-tuned run(s) have no '{k}' label in runs.json (left empty)")


README = """Person 3 evaluation export (dev sets only; official test sets are never included)

per_problem.csv -- one row per run x benchmark x problem
  run_id, model (0.5B/1.5B), is_baseline, budget_examples, budget_tokens (supervised training tokens),
  seed, arm (STOP/REPEAT/RANDOM/FIT/MATCHED_RANDOM for continuation runs), parent_id,
  benchmark (gsm/math), problem_id (same IDs in every run -> join on it for paired comparisons),
  correct (0/1; invalid answers count as 0), valid (1 = a final answer could be read),
  parse_status (ok / no_answer / unparseable_number / unparseable_math / ambiguous / ...),
  parsed_answer, gold_answer, generated_tokens, hit_length_limit (1 = stopped at the 1,024-token cap)

run_ledger.csv -- one row per run, ledger-template names, per benchmark (gsm_ / math_):
  dev_correct, dev_n, dev_accuracy, valid_answer_rate, accuracy_among_valid, length_limit_rate,
  mean_generated_tokens, repetition_rate (share of outputs with 4-gram repetition >= 0.20, in model tokens),
  evaluation_gpu_hours (model load + generation, one A100), predictions_path, eval_config_hash;
  plus parser_version, model_revision, checkpoint_sha256, total evaluation_gpu_hours.

Comparisons: compare a fine-tuned run only with its own model's baseline (column 'baseline'),
or arms with each other; never across GPU types or grader versions. Paired bootstrap used by
Person 3: 2,000 resamples of problem_id, seed 0, both runs resampled together (metrics.paired_bootstrap).
Fair "format vs reasoning" comparison: restrict to problems where valid = 1 in BOTH runs.
"""

if __name__ == "__main__":
    main()

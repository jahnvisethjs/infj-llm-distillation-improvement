"""
Run every evaluation listed in a run registry, then build one summary table.

  python run_eval.py --runs runs.json --results /content/drive/MyDrive/eval_results            # all pending runs
  python run_eval.py --runs runs.json --results ... --only R1000_05B_s42                          # one run
  python run_eval.py --runs runs.json --results ... --smoke                                       # 10-problem check of pending runs
  python run_eval.py --runs runs.json --results ... --summary-only                                # just rebuild summary.csv
  python run_eval.py --runs runs.json --results ... --dry-run                                     # print commands only

Registry (runs.json): a list of runs, e.g.
  {"run_id": "base",           "model": "Qwen/Qwen2.5-0.5B", "adapter": null}
  {"run_id": "R1000_05B_s42",  "model": "Qwen/Qwen2.5-0.5B", "adapter": "/content/drive/.../final/adapter",
   "phase": "curve", "budget_examples": 1000, "seed": 42, "method": "nested_prefix", "note": "v3 data"}
Optional fields (phase, budget_examples, budget_tokens, seed, method, note) are copied into the summary.
Field "train_manifest_sha256" (Person 1's hash of the training manifest, e.g. train_n500) makes
evaluate.py refuse an adapter that was trained on different data.
Field "baseline" (run_id of the untrained run to compare against) is REQUIRED when the registry
holds more than one baseline for the same model (e.g. old L4 "base" and "base_a100").
Field "where" ("sol" or "colab") assigns a run to one platform; run with --where sol / --where colab.

Rules built in:
- every run uses the frozen evaluate.py settings; only --model/--revision/--adapter change
- the model version comes from pinned_revisions.json (never typed by hand)
- an adapter must declare the same base model, and Person 2's run_config.json (if found) the same revision
- finished files are skipped; unfinished ones resume (evaluate.py refuses to mix settings)
- test sets run ONLY with --final (plan: final evaluation once, after all settings are frozen)
"""
from __future__ import annotations

import argparse, csv, json, os, subprocess, sys
from pathlib import Path

DEV = {"gsm": "data/dev_gsm.jsonl", "math": "data/dev_math.jsonl"}
TEST = {"gsm": "data/test_gsm.jsonl", "math": "data/test_math500.jsonl"}


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for l in f if l.strip())


def check_adapter(run, revision):
    """Refuse adapters that were trained on a different base model or version."""
    ad = Path(run["adapter"])
    cfg = ad / "adapter_config.json"
    if not cfg.exists():
        raise SystemExit(f"[{run['run_id']}] adapter_config.json not found in {ad}")
    base = json.load(open(cfg)).get("base_model_name_or_path", "")
    if base and base.rstrip("/").split("/")[-1] != run["model"].split("/")[-1]:
        raise SystemExit(f"[{run['run_id']}] adapter was trained on {base}, registry says {run['model']}")
    for rc in [ad.parent / "run_config.json", ad.parent.parent / "run_config.json"]:
        if rc.exists():
            trained_rev = json.load(open(rc)).get("revision")
            if trained_rev and trained_rev != revision:
                raise SystemExit(f"[{run['run_id']}] adapter trained from revision {trained_rev}, "
                                 f"pinned evaluation revision is {revision}")
            return str(rc)
    return None


def evaluate(run, bench, eval_file, out, revision, extra, dry):
    cmd = [sys.executable, "evaluate.py", "--eval_file", eval_file, "--run_id", f"{run['run_id']}_{bench}",
           "--model", run["model"], "--revision", revision, "--out", out] + extra
    if run.get("adapter"):
        cmd += ["--adapter", run["adapter"]]
        if run.get("train_manifest_sha256"):      # evaluate.py stops if the adapter was trained on other data
            cmd += ["--expect_train_manifest_sha256", run["train_manifest_sha256"]]
    print("  $", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def build_summary(runs, results, final):
    from metrics import load, summarize, paired_bootstrap, comparison_problems
    suffix = "_test" if final else ""
    sets = TEST if final else DEV
    run_ids = {r["run_id"] for r in runs}
    bases = {}
    for r in runs:
        if not r.get("adapter"):
            bases.setdefault(r["model"], []).append(r["run_id"])

    def baseline_for(run):
        if run.get("baseline"):
            if run["baseline"] not in run_ids:
                raise SystemExit(f"[{run['run_id']}] baseline {run['baseline']!r} is not in runs.json")
            return run["baseline"]
        found = bases.get(run["model"], [])
        if len(found) > 1:
            raise SystemExit(f"[{run['run_id']}] {len(found)} baselines for {run['model']} ({', '.join(found)}). "
                             f"Add \"baseline\": \"<run_id>\" to this run in runs.json.")
        return found[0] if found else None

    rows = []
    for run in runs:
        for bench in sets:
            f = f"{results}/{run['run_id']}{suffix}_{bench}.jsonl"
            if count_lines(f) < count_lines(sets[bench]):
                continue                                      # not finished yet
            res = load(f)
            s = summarize(res)
            row = {"run_id": run["run_id"], "model": run["model"].split("/")[-1], "benchmark": bench,
                   "adapter": "yes" if run.get("adapter") else "no (baseline)",
                   **{k: run.get(k, "") for k in ["phase", "method", "budget_examples", "budget_tokens", "seed", "note"]},
                   "n": s["n"], "accuracy": round(100 * s["accuracy"], 1),
                   "valid_answer_rate": round(100 * s["valid_answer_rate"], 1),
                   "accuracy_among_valid": round(100 * s["accuracy_among_valid"], 1),
                   "length_limit_rate": round(100 * s["length_limit_rate"], 1),
                   "eval_minutes": round(s["eval_seconds"] / 60, 1),
                   "gpu": json.load(open(f.replace(".jsonl", ".config.json"))).get("gpu", "not recorded")
                          if os.path.exists(f.replace(".jsonl", ".config.json")) else "not recorded",
                   "where": run.get("where", "")}
            base = baseline_for(run) if run.get("adapter") else None
            bf = f"{results}/{base}{suffix}_{bench}.jsonl" if base else None
            if bf and count_lines(bf) == s["n"]:
                base_rows = load(bf)
                problems = comparison_problems(bf, base_rows, f, res)
                row["baseline"] = base
                if problems:                       # never put an unfair comparison in the table
                    row["comparison_blocked"] = "; ".join(problems)
                    print(f"[{run['run_id']}] {bench}: NOT compared with {base}: {row['comparison_blocked']}")
                else:
                    pb = paired_bootstrap(base_rows, res)
                    r1 = lambda x: None if x is None else round(x, 1)
                    row.update(vs_baseline_pp=r1(pb["diff_pp"]),
                               ci95_low=r1(pb["ci95_pp"][0]), ci95_high=r1(pb["ci95_pp"][1]),
                               wrong_to_correct=pb["wrong_to_correct"], correct_to_wrong=pb["correct_to_wrong"],
                               mcnemar_p=f"{pb['mcnemar_p']:.2g}",
                               both_valid_n=pb["both_valid_n"],
                               both_valid_vs_baseline_pp=r1(pb["both_valid_diff_pp"]),
                               both_valid_ci95_low=r1(pb["both_valid_ci95_pp"] and pb["both_valid_ci95_pp"][0]),
                               both_valid_ci95_high=r1(pb["both_valid_ci95_pp"] and pb["both_valid_ci95_pp"][1]),
                               both_valid_mcnemar_p=f"{pb['both_valid_mcnemar_p']:.2g}")
            rows.append(row)
    if not rows:
        print("no finished runs yet"); return
    out = f"{results}/summary{suffix}.csv"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nsummary written: {out}")
    show = ["run_id", "benchmark", "accuracy", "valid_answer_rate", "vs_baseline_pp", "ci95_low", "ci95_high",
            "both_valid_vs_baseline_pp"]
    print("  ".join(f"{c:>17}" for c in show))
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):>17}" for c in show))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--only", default=None, help="run just this run_id")
    ap.add_argument("--where", default=None,
                    help="only runs whose registry field 'where' matches (e.g. sol / colab); splits work between platforms")
    ap.add_argument("--smoke", action="store_true", help="10 problems of dev_fast per pending run, no summary")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--final", action="store_true", help="FINAL TEST SETS (use once, after everything is frozen)")
    args = ap.parse_args()

    runs = json.load(open(args.runs))
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids)), "run_id values must be unique"
    pins = json.load(open(f"{args.results}/pinned_revisions.json"))

    if not args.summary_only:
        todo = [r for r in runs if not args.only or r["run_id"] == args.only]
        if args.where:
            skipped = [r["run_id"] for r in todo if r.get("where") != args.where]
            todo = [r for r in todo if r.get("where") == args.where]
            if skipped:
                print(f"not on {args.where} (skipped here):", ", ".join(skipped))
        if args.only and not todo:
            raise SystemExit(f"run_id {args.only} not in registry")
        sets = TEST if args.final else DEV
        suffix = "_test" if args.final else ""
        for run in todo:
            if run["model"] not in pins:
                raise SystemExit(f"[{run['run_id']}] no pinned revision for {run['model']} (pin it first)")
            rev = pins[run["model"]]
            if run.get("adapter"):
                rc = check_adapter(run, rev)
                print(f"[{run['run_id']}] adapter ok" + (f" (checked {rc})" if rc else ""))
            if args.smoke:
                out = f"{args.results}/{run['run_id']}_smoke.jsonl"
                if count_lines(out) >= 10:
                    print(f"[{run['run_id']}] smoke already done"); continue
                print(f"[{run['run_id']}] smoke test")
                evaluate(run, "smoke", "data/dev_fast.jsonl", out, rev, ["--limit", "10"], args.dry_run)
                continue
            for bench, f in sets.items():
                out = f"{args.results}/{run['run_id']}{suffix}_{bench}.jsonl"
                done, total = count_lines(out), count_lines(f)
                if done >= total:
                    print(f"[{run['run_id']}] {bench}: complete ({done}/{total}), skipped"); continue
                print(f"[{run['run_id']}] {bench}: {done}/{total} done, running")
                evaluate(run, bench, f, out, rev, ["--final_test"] if args.final else [], args.dry_run)

    if not args.smoke and not args.dry_run:
        build_summary(runs, args.results, args.final)


if __name__ == "__main__":
    main()

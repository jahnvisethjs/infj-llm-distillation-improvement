"""
Task 1 (working doc 4): complete the saturation study.  Reads only; never changes results.

  python saturation_study.py --results /scratch/$USER/eval_results            # all seeds found
  python saturation_study.py --results ... --boot 300                         # more bootstrap refits

Writes <results>/analysis/saturation/
  experiment_index.csv     1.1  one row per run (model, revision, manifest, rows, supervised tokens, updates, seed, ...)
  missing_inputs.txt       1.1  what is missing and whether it blocks analysis
  results_table.csv        1.2  accuracy per model x benchmark x budget x seed, plus mean / min / max over seeds
  curves_<model>.png       1.2  accuracy vs supervised tokens (log x), seeds as dots, mean line, fitted curve
  fits.csv, fits.txt       1.3  curve fits, AICc, 95%-of-improvement budget, bootstrap CIs, diagnostic checks
  transitions.csv          1.4  retained / improvement / regression / persistent failure, per seed
  review_sample.csv        1.4  ~40 regressions + improvements to annotate by hand (auto hints pre-filled)
  confirmation_plan.txt    1.5  what the evidence says to run next, with an eval-cost estimate
  budget_rule.txt          1.6  which budget the selection study should use, and why
Dev sets only. Fits use only fine-tuned runs; the baseline is fixed at its observed accuracy.
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, re, statistics as st
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

BENCH = ("gsm", "math")
TARGET = 0.95


# ---------------------------------------------------------------- loading
def load_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def run_info(R, run):
    rid = run["run_id"]
    m = re.match(r"R(\d+)_(\w+?)_s(\d+)$", rid)
    size = run["model"].split("-")[-1]
    info = {"run_id": rid, "model": size, "model_id": run["model"], "is_baseline": not run.get("adapter"),
            "budget_examples": int(m.group(1)) if m else 0, "seed": int(m.group(3)) if m else "",
            "adapter": run.get("adapter") or "", "baseline": run.get("baseline", ""), "note": run.get("note", "")}
    rc = {}
    if run.get("adapter"):
        d = os.path.dirname(run["adapter"])
        for f in ("run_config.json", "final/run_config.json", "training_summary.json", "../training_summary.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                rc.update({k: v for k, v in json.load(open(p)).items() if not isinstance(v, (dict, list))})
    info.update({
        "revision": rc.get("revision", ""), "manifest_sha256": rc.get("manifest_sha256", ""),
        "rows": rc.get("rows", info["budget_examples"] or ""),
        "supervised_tokens": rc.get("processed_tokens", run.get("budget_tokens") or ""),
        "unique_supervised_tokens": rc.get("unique_supervised_tokens", ""),
        "input_tokens": rc.get("input_tokens", ""), "optimizer_updates": rc.get("optimizer_updates", ""),
        "epochs": rc.get("epochs", ""), "learning_rate": rc.get("learning_rate", ""),
        "warmup_updates": rc.get("warmup_updates", ""), "accumulation": rc.get("accumulation", ""),
        "parent_optimizer_updates": rc.get("parent_optimizer_updates", ""),
        "training_seconds": rc.get("train_seconds", rc.get("training_seconds", rc.get("wall_seconds", ""))),
        "data_release": next((x.split("_qwen")[0] for x in str(run.get("adapter", "")).split("/") if x.startswith("sol_")), ""),
    })
    if info["is_baseline"]:
        info["supervised_tokens"] = 0
    problems, ok = [], True
    for b in BENCH:
        p, c = f"{R}/{rid}_{b}.jsonl", f"{R}/{rid}_{b}.config.json"
        if not os.path.exists(p):
            ok = False; problems.append(f"{b} predictions missing"); continue
        n = sum(1 for _ in open(p))
        if n < 500:
            ok = False; problems.append(f"{b} incomplete ({n}/500)")
        info[f"{b}_predictions"] = p
        if os.path.exists(c):
            cfg = json.load(open(c))
            info[f"{b}_eval_config_hash"] = cfg.get("eval_config_hash", "")
            info[f"{b}_gpu"] = cfg.get("gpu", "")
            secs = cfg.get("load_seconds", 0) + cfg.get("steady_eval_seconds", 0)
            info[f"{b}_eval_gpu_hours"] = round(secs / 3600, 3) if secs else ""
    info["complete"] = ok
    info["known_problems"] = "; ".join(problems)
    return info


# ---------------------------------------------------------------- curve models (T = supervised tokens, b = baseline)
def m_exp(T, D, s, b):            return b + D * (1 - np.exp(-T / s))
def m_weib(T, D, s, k, b):        return b + D * (1 - np.exp(-(T / s) ** k))
def m_power(T, c, a, al):         return c - a * T ** (-al)
def m_loglin(T, m0, beta):        return m0 + beta * np.log(T)


def aicc(rss, n, k):
    k = k + 1                                     # + noise variance
    if n - k - 1 <= 0 or rss <= 0:
        return float("nan")
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_all(T, y, b):
    """Returns {name: dict(params, rss, k, ceiling, improvement, t95)}; failures are skipped."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    Tmin, Tmax = T.min(), T.max()
    out = {}

    def add(name, f, p0, lo, hi, k, post):
        try:
            p, _ = curve_fit(f, T, y, p0=p0, bounds=(lo, hi), maxfev=20000)
            rss = float(((f(T, *p) - y) ** 2).sum())
            out[name] = dict(params=p, rss=rss, k=k, **post(p))
        except Exception:
            pass

    span = max(1e-3, y.max() - b)
    add("exponential", lambda T, D, s: m_exp(T, D, s, b), [span, Tmin], [-1, Tmin / 1e3], [1, Tmax * 1e2], 2,
        lambda p: dict(ceiling=b + p[0], improvement=p[0], t95=p[1] * math.log(1 / (1 - TARGET))))
    add("weibull", lambda T, D, s, k: m_weib(T, D, s, k, b), [span, Tmin, 1.0], [-1, Tmin / 1e3, 0.2], [1, Tmax * 1e2, 5], 3,
        lambda p: dict(ceiling=b + p[0], improvement=p[0], t95=p[1] * math.log(1 / (1 - TARGET)) ** (1 / p[2])))

    def pw_post(p):
        D = p[0] - b
        t = (p[1] / ((1 - TARGET) * D)) ** (1 / p[2]) if D > 0 and p[1] > 0 else float("nan")
        return dict(ceiling=p[0], improvement=D, t95=t)
    add("power_law", m_power, [y.max(), span * Tmin ** 0.5, 0.5], [0, -10, 0.01], [1, 1e6, 3], 3, pw_post)
    # comparison models (not saturation curves): flat within the tested range, and log-linear trend
    m = float(y.mean()); out["flat"] = dict(params=np.array([m]), rss=float(((y - m) ** 2).sum()), k=1,
                                             ceiling=m, improvement=m - b, t95=float("nan"))
    add("log_linear", m_loglin, [m, 0.0], [-1, -1], [2, 1], 2,
        lambda p: dict(ceiling=float("nan"), improvement=float("nan"), t95=float("nan"), slope_per_doubling=p[1] * math.log(2)))
    for v in out.values():
        v["aicc"] = aicc(v["rss"], len(y), v["k"])
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--boot", type=int, default=200, help="problem-bootstrap refits for CIs")
    ap.add_argument("--sample", type=int, default=40, help="cases for manual review")
    a = ap.parse_args()
    R = a.results
    out = os.path.join(R, "analysis", "saturation"); os.makedirs(out, exist_ok=True)
    runs = json.load(open(f"{R}/runs.json"))

    # ---- 1.1 experiment index
    index = [run_info(R, r) for r in runs]
    cols = list(dict.fromkeys(k for r in index for k in r))
    with open(f"{out}/experiment_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(index)
    missing = []
    for r in index:
        if not r["complete"]:
            missing.append(f"BLOCKS ANALYSIS  {r['run_id']}: {r['known_problems']} (excluded until finished)")
        if not r["is_baseline"]:
            for k in ("supervised_tokens", "optimizer_updates", "manifest_sha256", "revision"):
                if r.get(k) in ("", None):
                    missing.append(f"blocks {'curves' if k == 'supervised_tokens' else 'traceability'}  {r['run_id']}: no {k}")
            if r.get("training_seconds") in ("", None):
                missing.append(f"needed for compute estimate only  {r['run_id']}: no training runtime in run records (ask Person 2)")
            if r.get("unique_supervised_tokens") not in ("", None) and r["supervised_tokens"] != r["unique_supervised_tokens"]:
                missing.append(f"check  {r['run_id']}: processed tokens != unique supervised tokens (repeated epochs?)")
    for size in sorted({r["model"] for r in index}):
        hashes = {r.get(f"{b}_eval_config_hash") for r in index if r["model"] == size and r["complete"] for b in BENCH} - {None, ""}
        gpus = {r.get(f"{b}_gpu") for r in index if r["model"] == size and r["complete"] for b in BENCH} - {None, ""}
        if len(gpus) > 1:
            missing.append(f"BLOCKS COMPARISON  {size}: runs evaluated on different GPUs {sorted(gpus)}")
        epochs = {str(r["epochs"]) for r in index if r["model"] == size and not r["is_baseline"]}
        lrs = {str(r["learning_rate"]) for r in index if r["model"] == size and not r["is_baseline"]}
        if len(epochs) > 1 or len(lrs) > 1:
            missing.append(f"comparability  {size}: epochs {sorted(epochs)} / learning rates {sorted(lrs)} differ across runs")
    releases = sorted({r["data_release"] for r in index if r["data_release"]})
    note_release = (f"Data releases used: {releases}. Curves mix releases; describe them as performance under the tested "
                    f"protocol and data releases.") if len(releases) > 1 else ""
    open(f"{out}/missing_inputs.txt", "w").write("\n".join(missing + ([note_release] if note_release else [])
                                                           or ["nothing missing that blocks analysis"]) + "\n")

    good = [r for r in index if r["complete"]]
    preds = {(r["run_id"], b): {x["problem_id"]: x for x in load_rows(r[f"{b}_predictions"])} for r in good for b in BENCH}

    # ---- 1.2 results table
    table, curves = [], {}
    for size in sorted({r["model"] for r in good}):
        base = [r for r in good if r["model"] == size and r["is_baseline"]]
        if len(base) != 1:
            print(f"!! {size}: expected one baseline, found {len(base)}; skipped"); continue
        base = base[0]
        ft = sorted([r for r in good if r["model"] == size and not r["is_baseline"]], key=lambda r: (r["budget_examples"], r["seed"]))
        for b in BENCH:
            ids = sorted(preds[(base["run_id"], b)])
            M = {r["run_id"]: np.array([int(preds[(r["run_id"], b)][i]["correct"]) for i in ids]) for r in [base] + ft}
            curves[(size, b)] = dict(base=base, ft=ft, ids=ids, M=M)
            table.append({"model": size, "benchmark": b, "budget_examples": 0, "supervised_tokens": 0, "seed": "baseline",
                          "accuracy": round(100 * M[base["run_id"]].mean(), 1)})
            by_budget = defaultdict(list)
            for r in ft:
                acc = 100 * M[r["run_id"]].mean(); by_budget[r["budget_examples"]].append((r, acc))
                table.append({"model": size, "benchmark": b, "budget_examples": r["budget_examples"],
                              "supervised_tokens": r["supervised_tokens"], "seed": r["seed"], "accuracy": round(acc, 1)})
            for n, lst in sorted(by_budget.items()):
                accs = [x[1] for x in lst]
                table.append({"model": size, "benchmark": b, "budget_examples": n, "supervised_tokens": lst[0][0]["supervised_tokens"],
                              "seed": f"mean of {len(accs)} seed(s)", "accuracy": round(st.mean(accs), 1),
                              "seed_min": round(min(accs), 1), "seed_max": round(max(accs), 1)})
    with open(f"{out}/results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "benchmark", "budget_examples", "supervised_tokens", "seed", "accuracy", "seed_min", "seed_max"])
        w.writeheader(); w.writerows(table)

    # ---- 1.3 fits + problem bootstrap
    fit_rows, lines, decisions = [], [], {}
    rng = np.random.default_rng(0)
    for (size, b), c in curves.items():
        T = np.array([float(r["supervised_tokens"]) for r in c["ft"]])
        Y = lambda idx: (c["M"][c["base"]["run_id"]][idx].mean(), np.array([c["M"][r["run_id"]][idx].mean() for r in c["ft"]]))
        allidx = np.arange(len(c["ids"]))
        b0, y0 = Y(allidx)
        if len(T) < 3:
            lines.append(f"{size} {b}: only {len(T)} fine-tuned runs; no fit"); continue
        fits = fit_all(T, y0, b0)
        boots = defaultdict(lambda: defaultdict(list))
        for _ in range(a.boot):
            idx = rng.integers(0, len(allidx), len(allidx))
            bb, yy = Y(idx)
            for name, v in fit_all(T, yy, bb).items():
                for key in ("ceiling", "improvement", "t95", "slope_per_doubling"):
                    if key in v and v[key] == v[key]:
                        boots[name][key].append(v[key])
        ci = lambda name, key: (np.percentile(boots[name][key], [2.5, 97.5]) if len(boots[name][key]) >= max(20, a.boot // 2) else (np.nan, np.nan))
        best_sat = min((n for n in ("exponential", "weibull", "power_law") if n in fits and fits[n]["aicc"] == fits[n]["aicc"]),
                       key=lambda n: fits[n]["aicc"], default=None)
        lines.append(f"\n===== {size} {b.upper()}  baseline {100*b0:.1f}%   fine-tuned runs n={len(T)}   "
                     f"tested tokens {T.min():,.0f}-{T.max():,.0f}")
        for name, v in sorted(fits.items(), key=lambda kv: (kv[1]["aicc"] if kv[1]["aicc"] == kv[1]["aicc"] else 1e9)):
            lo_c, hi_c = ci(name, "ceiling"); lo_t, hi_t = ci(name, "t95"); lo_d, hi_d = ci(name, "improvement")
            row = {"model": size, "benchmark": b, "fit": name, "aicc": round(v["aicc"], 2), "rss": v["rss"], "n_points": len(T),
                   "baseline": round(100 * b0, 1), "ceiling": round(100 * v["ceiling"], 1) if v["ceiling"] == v["ceiling"] else "",
                   "ceiling_ci": f"[{100*lo_c:.1f}, {100*hi_c:.1f}]" if lo_c == lo_c else "",
                   "improvement_pp": round(100 * v["improvement"], 1) if v["improvement"] == v["improvement"] else "",
                   "improvement_ci": f"[{100*lo_d:.1f}, {100*hi_d:.1f}]" if lo_d == lo_d else "",
                   "t95_tokens": round(v["t95"]) if v["t95"] == v["t95"] else "",
                   "t95_ci": f"[{lo_t:,.0f}, {hi_t:,.0f}]" if lo_t == lo_t else "",
                   "params": " ".join(f"{p:.4g}" for p in v["params"])}
            if name == "log_linear":
                lo_s, hi_s = ci(name, "slope_per_doubling")
                row["slope_pp_per_doubling"] = round(100 * v["slope_per_doubling"], 2)
                row["slope_ci"] = f"[{100*lo_s:.2f}, {100*hi_s:.2f}]" if lo_s == lo_s else ""
            fit_rows.append(row)
            lines.append(f"  {name:11s} AICc {row['aicc']:8.2f}  ceiling {row['ceiling']!s:>5} {row['ceiling_ci']:16s} "
                         f"95%-budget {row['t95_tokens']!s:>11} tokens {row['t95_ci']}"
                         + (f"  trend {row['slope_pp_per_doubling']:+.2f} pp per doubling {row['slope_ci']}" if name == "log_linear" else ""))
        # diagnostics (working doc 1.3 checks)
        chk, verdict = [], "no fit"
        if best_sat:
            v = fits[best_sat]; lo_d, _ = ci(best_sat, "improvement"); lo_c, hi_c = ci(best_sat, "ceiling"); lo_t, hi_t = ci(best_sat, "t95")
            pos = v["improvement"] > 0 and lo_d == lo_d and lo_d > 0
            constrained = lo_c == lo_c and (hi_c - lo_c) <= 0.10 and v["ceiling"] <= y0.max() + 0.05
            t95 = v["t95"]
            in_range = t95 == t95 and T.min() <= t95 <= T.max()
            below = t95 == t95 and t95 < T.min()
            sl = fits.get("log_linear", {}); lo_s, hi_s = ci("log_linear", "slope_per_doubling")
            decline = bool(sl) and hi_s == hi_s and hi_s < 0
            rising = bool(sl) and lo_s == lo_s and lo_s > 0
            flat_better = fits["flat"]["aicc"] <= v["aicc"]
            chk = [f"best saturation fit by AICc: {best_sat}" + ("  (but the FLAT model fits as well or better: no budget effect inside the tested range)" if flat_better else ""),
                   f"fitted improvement positive (bootstrap CI above 0): {'YES' if pos else 'NO'}",
                   f"ceiling constrained (CI width <= 10 pp, not far above observed max): {'YES' if constrained else 'NO'}",
                   f"95% budget inside the tested range: {'YES' if in_range else ('NO - BELOW the smallest tested budget (extrapolated)' if below else 'NO')}",
                   f"fit hides a decline (log-linear trend CI below 0 across fine-tuned budgets): {'YES - DECLINE' if decline else ('no; trend still RISING' if rising else 'no')}"]
            if pos and constrained and in_range and not decline:
                verdict = f"SUPPORTED: saturation at ~{t95:,.0f} supervised tokens (95% CI {lo_t:,.0f}-{hi_t:,.0f})"
            elif pos and below:
                verdict = (f"NOT IDENTIFIED INSIDE THE TESTED RANGE: the gain is reached before the smallest budget "
                           f"({T.min():,.0f} tokens); saturation is at or below that point" + ("; accuracy then declines" if decline else ""))
            elif decline:
                verdict = "NOT A PLATEAU: accuracy declines after an early peak; report the peak and the decline, not a saturation point"
            elif rising:
                verdict = "STILL IMPROVING at the largest budget: no saturation inside the tested range"
            else:
                verdict = "NOT SUFFICIENTLY SUPPORTED: fit unstable or ceiling unconstrained"
        decisions[(size, b)] = dict(verdict=verdict, T=T, fits=fits, best=best_sat)
        lines += ["  checks:"] + [f"    - {x}" for x in chk] + [f"  VERDICT: {verdict}"]
    with open(f"{out}/fits.csv", "w", newline="") as f:
        cols = list(dict.fromkeys(k for r in fit_rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(fit_rows)
    open(f"{out}/fits.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for size in sorted({k[0] for k in curves}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            for ax, b in zip(axes, BENCH):
                c = curves.get((size, b))
                if not c: continue
                b0 = 100 * c["M"][c["base"]["run_id"]].mean()
                seeds = sorted({r["seed"] for r in c["ft"]})
                for s, mk in zip(seeds, "osD^v"):
                    rs = [r for r in c["ft"] if r["seed"] == s]
                    ax.plot([float(r["supervised_tokens"]) for r in rs], [100 * c["M"][r["run_id"]].mean() for r in rs],
                            mk + "-", ms=5, lw=0.8, alpha=.6, color="#8a8983", label=f"seed {s}")
                agg = defaultdict(list)
                for r in c["ft"]:
                    agg[float(r["supervised_tokens"])].append(100 * c["M"][r["run_id"]].mean())
                xs = sorted(agg)
                multi = [x for x in xs if len(agg[x]) > 1]
                if multi:
                    mu = [st.mean(agg[x]) for x in multi]
                    ax.errorbar(multi, mu, yerr=[[m - min(agg[x]) for m, x in zip(mu, multi)], [max(agg[x]) - m for m, x in zip(mu, multi)]],
                                fmt="o", color="#2a78d6", ms=7, capsize=4, lw=1.5, label="mean of seeds (range)", zorder=5)
                d = decisions.get((size, b))
                if d and d["best"]:
                    xx = np.logspace(math.log10(min(xs)), math.log10(max(xs)), 200); v = d["fits"][d["best"]]
                    f = {"exponential": lambda T: m_exp(T, *v["params"], b0 / 100), "weibull": lambda T: m_weib(T, *v["params"], b0 / 100),
                         "power_law": lambda T: m_power(T, *v["params"])}[d["best"]]
                    ax.plot(xx, 100 * f(xx), "--", color="#eb6834", lw=1.5, label=f"fit: {d['best']}")
                ax.axhline(b0, color="#8a8983", ls=":", lw=1.2, label=f"base model {b0:.1f}%")
                ax.set_xscale("log"); ax.set_xlabel("Supervised training tokens (log scale)"); ax.set_ylabel("Dev accuracy (%)")
                ax.set_title(f"Qwen2.5-{size}  {'GSM8K' if b == 'gsm' else 'MATH'} dev", loc="left", fontweight="bold")
                ax.grid(True, color="#e6e5e0"); ax.legend(frameon=False, fontsize=8)
                for x in xs:
                    n = next(r["budget_examples"] for r in c["ft"] if float(r["supervised_tokens"]) == x)
                    ax.annotate(f"{n:,}", (x, ax.get_ylim()[0]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7, color="#8a8983")
            fig.tight_layout(); fig.savefig(f"{out}/curves_{size}.png", dpi=160); plt.close(fig)
    except ImportError:
        print("(matplotlib not installed: plots skipped)")

    # ---- 1.4 transitions + review sample
    trans, pool = [], []
    for (size, b), c in curves.items():
        seeds = sorted({r["seed"] for r in c["ft"]})
        mean_by = defaultdict(list)
        for r in c["ft"]:
            mean_by[r["budget_examples"]].append(c["M"][r["run_id"]].mean())
        peak = max(mean_by, key=lambda n: st.mean(mean_by[n])); largest = max(mean_by); smallest = min(mean_by)
        seed_budgets = defaultdict(set)
        for r in c["ft"]:
            seed_budgets[r["seed"]].add(r["budget_examples"])
        common = set.intersection(*seed_budgets.values()) if seed_budgets else set()
        largest_common = max(common) if common else None
        pairs = [("baseline", None, smallest, "baseline->smallest", largest)] + \
                [("budget", e, largest, lab, largest) for e, lab in ((smallest, "smallest->largest"), (peak, "peak->largest"))
                 if e != largest and not (lab == "peak->largest" and peak == smallest)]
        if largest_common and largest_common not in (smallest, largest) and len(seed_budgets) > 1:
            pairs.append(("budget", smallest, largest_common, "smallest->largest_all_seeds", largest_common))
        for kind, early, late, lab, _ in pairs:
            for s in seeds:
                er = c["base"] if kind == "baseline" else next((r for r in c["ft"] if r["budget_examples"] == early and r["seed"] == s), None)
                lr = next((r for r in c["ft"] if r["budget_examples"] == late and r["seed"] == s), None)
                if not er or not lr: continue
                A, B = c["M"][er["run_id"]], c["M"][lr["run_id"]]
                d = B - A; boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                trans.append({"model": size, "benchmark": b, "pair": lab, "seed": s, "earlier": er["run_id"], "later": lr["run_id"],
                              "retained": int(((A == 1) & (B == 1)).sum()), "improvement": int(((A == 0) & (B == 1)).sum()),
                              "regression": int(((A == 1) & (B == 0)).sum()), "persistent_failure": int(((A == 0) & (B == 0)).sum()),
                              "change_pp": round(100 * d.mean(), 1), "ci95": f"[{100*lo:+.1f}, {100*hi:+.1f}]",
                              "significant": "yes" if lo > 0 or hi < 0 else "no"})
                if lab == ("peak->largest" if peak != smallest else "smallest->largest"):
                    for j, pid in enumerate(c["ids"]):
                        if A[j] != B[j]:
                            x = preds[(lr["run_id"], b)][pid]
                            hint = ("truncated" if x["finish_reason"] == "length" else "malformed/unreadable" if x["parse_status"] != "ok"
                                    else "repetition" if x.get("repetition_score", 0) >= 0.5 else "")
                            pool.append({"model": size, "benchmark": b, "seed": s, "problem_id": pid,
                                         "type": "regression" if A[j] == 1 else "improvement", "earlier": er["run_id"], "later": lr["run_id"],
                                         "gold": x["gold_answer"], "earlier_answer": preds[(er["run_id"], b)][pid]["parsed_answer"],
                                         "later_answer": x["parsed_answer"], "auto_hint": hint,
                                         "later_output_tail": x["raw_output"][-600:].replace("\n", " "),
                                         "cause (fill: arithmetic / setup / incomplete / malformed / repetition / truncation / unclear)": "",
                                         "notes": ""})
    with open(f"{out}/transitions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trans[0])); w.writeheader(); w.writerows(trans)
    random.Random(0).shuffle(pool)
    regs = [p for p in pool if p["type"] == "regression"]; imps = [p for p in pool if p["type"] == "improvement"]
    sample = regs[: int(a.sample * 0.7)] + imps[: a.sample - min(len(regs), int(a.sample * 0.7))]
    if sample:
        with open(f"{out}/review_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sample[0])); w.writeheader(); w.writerows(sample)

    # ---- 1.5 confirmation plan + 1.6 budget rule
    eval_h = [r[f"{b}_eval_gpu_hours"] for r in good for b in BENCH if isinstance(r.get(f"{b}_eval_gpu_hours"), float)]
    per_run = 2 * st.mean(eval_h) if eval_h else float("nan")
    plan, rule = [], []
    for (size, b), d in decisions.items():
        c = curves[(size, b)]; seeds = sorted({r["seed"] for r in c["ft"]})
        smallest = min(r["budget_examples"] for r in c["ft"])
        plan.append(f"{size} {b.upper()}: {d['verdict']}")
        for lab in ("smallest->largest", "peak->largest", "smallest->largest_all_seeds"):
            tr = [t for t in trans if t["model"] == size and t["benchmark"] == b and t["pair"] == lab]
            if not tr: continue
            neg = sum(t["change_pp"] < 0 for t in tr); sig = sum(t["significant"] == "yes" and t["change_pp"] < 0 for t in tr)
            plan.append(f"   {lab} ({tr[0]['earlier'].split('_')[0]} -> {tr[0]['later'].split('_')[0]}): lower in {neg}/{len(tr)} seeds, "
                        f"significantly lower in {sig}/{len(tr)}; changes {', '.join(f's{t['seed']} {t['change_pp']:+.1f}' for t in tr)}"
                        + ("  -> DECLINE CONSISTENT ACROSS SEEDS" if len(tr) >= 2 and sig == len(tr)
                           else "  -> only one seed: cannot claim consistency" if len(tr) == 1 else "  -> not consistent across seeds"))
        if "BELOW" in d["verdict"] or "NOT IDENTIFIED" in d["verdict"]:
            plan.append(f"   -> run budgets below {smallest} (e.g. 50, 100, 250) for this model; seeds {seeds}. "
                        "Warn Person 2: with 8 examples/update and 10 warmup updates, budgets < 100 never finish warmup.")
        if "decline" in d["verdict"].lower() and len(seeds) < 3:
            plan.append("   -> repeat the peak and the largest budget with another seed before claiming a consistent decline")
    plan.append(f"\nEvaluation cost: about {per_run:.2f} GPU-hours per run (both dev sets, one A100), from {len(eval_h)} saved configs. "
                "Training cost: ask Person 2 (not in run records)." if per_run == per_run else "\nEvaluation cost: no timing in configs.")
    open(f"{out}/confirmation_plan.txt", "w").write("\n".join(plan) + "\n")
    for (size, b), d in decisions.items():
        c = curves[(size, b)]; largest = max(c["ft"], key=lambda r: r["budget_examples"])
        if d["verdict"].startswith("SUPPORTED"):
            rule.append(f"{size} {b.upper()}: use the estimated saturation budget ({d['verdict'].split('at ~')[1]}).")
        else:
            rule.append(f"{size} {b.upper()}: saturation not sufficiently supported -> proposal FALLBACK: largest completed budget "
                        f"{largest['budget_examples']:,} examples = {float(largest['supervised_tokens']):,.0f} supervised tokens. "
                        "Report it as a fallback comparison budget, NOT as an observed saturation point.")
    rule.append("\nPrimary metric is GSM8K dev (working doc); if GSM8K and MATH disagree, the team's pre-agreed rule decides. "
                "Record the decision and its date before looking at selection-study results.")
    open(f"{out}/budget_rule.txt", "w").write("\n".join(rule) + "\n")
    print("\n" + "\n".join(plan) + "\n\n" + "\n".join(rule))
    print(f"\nall outputs -> {out}/")


if __name__ == "__main__":
    main()

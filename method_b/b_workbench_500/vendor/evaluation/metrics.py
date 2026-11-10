"""
Metrics (Person 3) and paired bootstrap comparison.

  python metrics.py results/base_devfast.jsonl              # metrics per benchmark
  python metrics.py results/R500_gsm.jsonl results/R1000_gsm.jsonl --compare   # B vs A
"""
from __future__ import annotations

import argparse, json, math, os, random
import statistics as st

VALID = {"ok"}
_TOK = None


def _tokens(text):
    """Qwen tokenizer (same revision as training) if available, else whitespace words."""
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B", revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
        except Exception:
            _TOK = False
            print("(tokenizer unavailable: answer repetition uses words instead of tokens)")
    return _TOK(text, add_special_tokens=False).input_ids if _TOK else text.split()


def _token_unit():
    _tokens("")                       # loads the tokenizer once if possible
    return "tokens" if _TOK else "WORDS"


def answer_repetition(row):
    """Repetition score on the answer part only (ignores text after the answer)."""
    from answer_parser import answer_region
    ids = _tokens(answer_region(row["raw_output"]))
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if len(grams) < 20 else 1 - len(set(grams)) / len(grams)


def continues_after_answer(row):
    """True if the model kept writing (new question, system prompt...) after its answer."""
    from answer_parser import answer_region
    return len(row["raw_output"].strip()) - len(answer_region(row["raw_output"]).strip()) > 20


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(rows):
    n = len(rows)
    valid = [r for r in rows if r["parse_status"] in VALID]
    toks = sorted(r["generated_tokens"] for r in rows)
    p95 = toks[min(len(toks) - 1, int(0.95 * len(toks)))]
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "valid_answer_rate": len(valid) / n,
        "accuracy_among_valid": (sum(r["correct"] for r in valid) / len(valid)) if valid else 0.0,
        "length_limit_rate": sum(r["finish_reason"] == "length" for r in rows) / n,
        "gen_tokens_mean": st.mean(toks), "gen_tokens_median": st.median(toks), "gen_tokens_p95": p95,
        # full_output is always counted in model tokens (saved by evaluate.py); answer_part falls
        # back to words when the tokenizer can't load -- the two are then NOT comparable.
        f"repetition_rate(answer_part, {_token_unit()})":
            sum(answer_repetition(r) >= 0.20 for r in rows) / n,
        "repetition_rate(full_output, tokens)": sum(r["repetition_score"] >= 0.20 for r in rows) / n,
        "keeps_writing_after_answer": sum(continues_after_answer(r) for r in rows) / n,
        "eval_seconds": sum(r["latency"] for r in rows),
        "parse_status_counts": {s: sum(r["parse_status"] == s for r in rows)
                                for s in sorted({r["parse_status"] for r in rows})},
    }


def _bootstrap_ci(diffs, n_boot, seed):
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return (100 * boots[int(0.025 * n_boot)], 100 * boots[int(0.975 * n_boot) - 1])


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test (binomial on the discordant pairs b vs c)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_bootstrap(a_rows, b_rows, n_boot=2000, seed=0):
    """Mean (B - A) accuracy difference in percentage points + 95% percentile CI.
    Rows are aligned by problem_id; both models are resampled together.
    Also reports an exact McNemar test, and the same comparison restricted to problems
    where BOTH models gave a valid answer (separates format gains from reasoning gains)."""
    a = {r["problem_id"]: r for r in a_rows}
    b = {r["problem_id"]: r for r in b_rows}
    ids = sorted(a.keys() & b.keys())
    if len(ids) != len(a) or len(ids) != len(b):
        print(f"WARNING: only {len(ids)} shared problem_ids (A={len(a)}, B={len(b)})")
    diffs = [int(b[i]["correct"]) - int(a[i]["correct"]) for i in ids]
    w2c, c2w = sum(d == 1 for d in diffs), sum(d == -1 for d in diffs)
    both = [i for i in ids if a[i]["parse_status"] in VALID and b[i]["parse_status"] in VALID]
    bd = [int(b[i]["correct"]) - int(a[i]["correct"]) for i in both]
    bw2c, bc2w = sum(d == 1 for d in bd), sum(d == -1 for d in bd)
    return {"n_shared": len(ids), "diff_pp": 100 * sum(diffs) / len(diffs),
            "ci95_pp": _bootstrap_ci(diffs, n_boot, seed),
            "wrong_to_correct": w2c, "correct_to_wrong": c2w, "mcnemar_p": mcnemar_exact(w2c, c2w),
            "both_valid_n": len(both),
            "both_valid_acc_A": 100 * sum(a[i]["correct"] for i in both) / len(both) if both else None,
            "both_valid_acc_B": 100 * sum(b[i]["correct"] for i in both) / len(both) if both else None,
            "both_valid_diff_pp": 100 * sum(bd) / len(bd) if bd else None,
            "both_valid_ci95_pp": _bootstrap_ci(bd, n_boot, seed) if bd else None,
            "both_valid_mcnemar_p": mcnemar_exact(bw2c, bc2w)}


# Settings that must match between two runs before they are compared. adapter_mode is left
# out on purpose: it is None for a baseline and "merged" for every fine-tuned model, so a
# baseline and a fine-tuned run always have different eval_config_hash values.
COMPARE_KEYS = ("model", "revision", "precision", "max_new_tokens", "batch_size", "stop_at_question",
                "question_text", "prompt_sha256", "add_special_tokens", "decoding", "eos_token_id",
                "transformers", "peft", "torch", "gpu", "cuda", "math_verify", "eval_file_sha256")


def comparison_problems(a_path, a_rows, b_path, b_rows):
    """Reasons why two result files must NOT be compared (empty list = OK)."""
    problems = []
    for key in ("parser_version", "eval_config_hash"):
        for path, rows in ((a_path, a_rows), (b_path, b_rows)):
            if len({r.get(key) for r in rows}) != 1:
                problems.append(f"{os.path.basename(path)} mixes several {key} values")
    if {r.get("parser_version") for r in a_rows} != {r.get("parser_version") for r in b_rows}:
        problems.append("different grader versions (re-grade both with rescore.py)")
    if {r.get("eval_config_hash") for r in a_rows} == {r.get("eval_config_hash") for r in b_rows}:
        return problems                        # identical settings: nothing more to check
    cfgs = []
    for path in (a_path, b_path):
        cp = path.replace(".jsonl", "") + ".config.json"
        if not os.path.exists(cp):
            return problems + [f"settings hashes differ and {os.path.basename(cp)} is missing, "
                               "so the settings can't be checked"]
        cfgs.append(json.load(open(cp)))
    for key in COMPARE_KEYS:
        if cfgs[0].get(key) != cfgs[1].get(key):
            problems.append(f"setting '{key}' differs: {cfgs[0].get(key)!r} vs {cfgs[1].get(key)!r}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", action="store_true", help="paired bootstrap: 2nd file minus 1st")
    args = ap.parse_args()

    if args.compare:
        a, b = load(args.files[0]), load(args.files[1])
        # Only compare like with like: same grader and same generation settings
        # (adapter on/off is the one difference that is allowed).
        problems = comparison_problems(args.files[0], a, args.files[1], b)
        if problems:
            raise SystemExit("STOP: these runs can't be compared:\n  - " + "\n  - ".join(problems))
        for bench in sorted({r["benchmark"] for r in a}):
            res = paired_bootstrap([r for r in a if r["benchmark"] == bench],
                                   [r for r in b if r["benchmark"] == bench])
            print(bench, json.dumps(res, indent=2))
        return

    rows = [r for f in args.files for r in load(f)]
    for bench in sorted({r["benchmark"] for r in rows}):
        print(f"== {bench}")
        for k, v in summarize([r for r in rows if r["benchmark"] == bench]).items():
            print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")


if __name__ == "__main__":
    main()

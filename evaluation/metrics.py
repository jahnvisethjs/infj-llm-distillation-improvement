"""
Metrics (Person 3) and paired bootstrap comparison.

  python metrics.py results/base_devfast.jsonl              # metrics per benchmark
  python metrics.py results/R500_gsm.jsonl results/R1000_gsm.jsonl --compare   # B vs A
"""
from __future__ import annotations

import argparse, json, random
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
        "repetition_rate(answer_part)": sum(answer_repetition(r) >= 0.20 for r in rows) / n,
        "repetition_rate(full_output)": sum(r["repetition_score"] >= 0.20 for r in rows) / n,
        "keeps_writing_after_answer": sum(continues_after_answer(r) for r in rows) / n,
        "eval_seconds": sum(r["latency"] for r in rows),
        "parse_status_counts": {s: sum(r["parse_status"] == s for r in rows)
                                for s in sorted({r["parse_status"] for r in rows})},
    }


def paired_bootstrap(a_rows, b_rows, n_boot=2000, seed=0):
    """Mean (B - A) accuracy difference in percentage points + 95% percentile CI.
    Rows are aligned by problem_id; both models are resampled together."""
    a = {r["problem_id"]: int(r["correct"]) for r in a_rows}
    b = {r["problem_id"]: int(r["correct"]) for r in b_rows}
    ids = sorted(a.keys() & b.keys())
    if len(ids) != len(a) or len(ids) != len(b):
        print(f"WARNING: only {len(ids)} shared problem_ids (A={len(a)}, B={len(b)})")
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return {"n_shared": len(ids), "diff_pp": 100 * sum(diffs) / len(diffs),
            "ci95_pp": (100 * boots[int(0.025 * n_boot)], 100 * boots[int(0.975 * n_boot) - 1]),
            "wrong_to_correct": sum(d == 1 for d in diffs),
            "correct_to_wrong": sum(d == -1 for d in diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", action="store_true", help="paired bootstrap: 2nd file minus 1st")
    args = ap.parse_args()

    if args.compare:
        a, b = load(args.files[0]), load(args.files[1])
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

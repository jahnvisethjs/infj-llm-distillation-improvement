"""
Step 1 (E1): download the benchmarks and build the evaluation files.

  python prepare_eval_data.py            # writes everything into data/

Every row has the SAME fields (same idea as Person 1's training files):
  problem_id, benchmark, source, split, question, gold_answer, gold_solution, subject, level
  (subject/level are null for GSM8K; gold_answer is the final answer only)

Creates:
  data/dev_gsm.jsonl      500 problems from GSM8K TRAIN      (used during the project)
  data/dev_math.jsonl     500 problems from MATH TRAIN       (used during the project,
                                                              balanced by subject and level)
  data/test_gsm.jsonl     all 1,319 GSM8K TEST problems      (final evaluation only)
  data/test_math500.jsonl all 500 MATH-500 problems          (final evaluation only)
  data/*_ids.txt          the problem IDs, small enough to commit to git
  data/eval_data_metadata.json   dataset names + exact revisions + seed

Same seed + same dataset revisions => everyone gets exactly the same problems.
"""
import hashlib, json, os
from collections import defaultdict

from datasets import load_dataset
from answer_parser import gsm_gold, math_gold, _boxed_contents
from huggingface_hub import HfApi

SEED = "1729"           # evaluation split seed from the plan
N_DEV = 500
OUT = "data"

GSM = "openai/gsm8k"
MATH_TRAIN = "EleutherAI/hendrycks_math"
MATH500 = "HuggingFaceH4/MATH-500"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def row(pid, bench, source, split, question, answer, solution, subject=None, level=None):
    return {"problem_id": pid, "benchmark": bench, "source": source, "split": split,
            "question": question, "gold_answer": answer, "gold_solution": solution,
            "subject": subject, "level": level}


def order_key(pid):
    """Reproducible shuffle: sort by SHA256(seed|id)."""
    return hashlib.sha256(f"{SEED}|{pid}".encode()).hexdigest()


def write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/{name}_ids.txt", "w") as f:
        f.write("\n".join(r["problem_id"] for r in rows) + "\n")
    print(f"  {name:15s} {len(rows):5d} problems")


def single_answer(r):
    """Rule: skip MATH problems whose solution boxes 2+ DIFFERENT answers
    (multi-part or 'either answer' problems can't be graded with one gold answer)."""
    return len(set(_boxed_contents(r["gold_solution"]))) <= 1


def stratified_take(rows, n, strata_key, keep=lambda r: True):
    """Proportional sample across strata (largest-remainder rounding), deterministic.
    Quotas use ALL rows; rows failing `keep` are skipped and the next row in the
    same stratum (same subject + level) takes their place."""
    groups = defaultdict(list)
    for r in rows:
        groups[strata_key(r)].append(r)
    for g in groups.values():
        g.sort(key=lambda r: order_key(r["problem_id"]))
    total = len(rows)
    quota = {k: n * len(g) / total for k, g in groups.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    leftover = n - sum(alloc.values())
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[:leftover]:
        alloc[k] += 1
    picked = [r for k in sorted(groups, key=str) for r in [x for x in groups[k] if keep(x)][: alloc[k]]]
    return sorted(picked, key=lambda r: order_key(r["problem_id"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    api = HfApi()
    revs = {d: api.dataset_info(d).sha for d in [GSM, MATH_TRAIN, MATH500]}
    print("dataset revisions:", json.dumps(revs, indent=2))

    # ---- GSM8K
    gsm = load_dataset(GSM, "main", revision=revs[GSM])
    gsm_train = [row(f"gsm8k_train_{i:05d}", "gsm", GSM, "train", r["question"],
                     gsm_gold(r["answer"]), r["answer"]) for i, r in enumerate(gsm["train"])]
    test_gsm = [row(f"gsm8k_test_{i:05d}", "gsm", GSM, "test", r["question"],
                    gsm_gold(r["answer"]), r["answer"]) for i, r in enumerate(gsm["test"])]
    dev_gsm = sorted(gsm_train, key=lambda r: order_key(r["problem_id"]))[:N_DEV]

    # ---- MATH train (all 7 subjects)
    math_train = []
    for subj in MATH_SUBJECTS:
        ds = load_dataset(MATH_TRAIN, subj, split="train", revision=revs[MATH_TRAIN])
        for i, r in enumerate(ds):
            math_train.append(row(f"math_train_{subj}_{i:05d}", "math", MATH_TRAIN, "train",
                                  r["problem"], math_gold({"solution": r["solution"]}),
                                  r["solution"], subj, r["level"]))
    dev_math = stratified_take(math_train, N_DEV, lambda r: (r["subject"], r["level"]),
                               keep=single_answer)
    skipped = sorted(r["problem_id"] for r in math_train if not single_answer(r))
    print(f"MATH train problems skipped (2+ different boxed answers): {len(skipped)}")

    # ---- MATH-500 (final test)
    m500 = load_dataset(MATH500, split="test", revision=revs[MATH500])
    test_math500 = [row(f"math500_{r['unique_id']}", "math", MATH500, "test", r["problem"],
                        r["answer"], r["solution"], r["subject"],
                        r["level"] if str(r["level"]).startswith("Level") else f"Level {r['level']}")
                    for r in m500]

    # ---- safety check: dev questions must not appear in the test sets
    norm = lambda s: " ".join(s.lower().split())
    test_q = {norm(r["question"]) for r in test_gsm + test_math500}
    clash = [r["problem_id"] for r in dev_gsm + dev_math if norm(r["question"]) in test_q]
    print(f"dev/test exact overlaps: {len(clash)}", clash[:5])

    print("writing files:")
    write("dev_gsm", dev_gsm)
    write("dev_math", dev_math)
    write("test_gsm", test_gsm)
    write("test_math500", test_math500)
    json.dump({"seed": SEED, "n_dev": N_DEV, "revisions": revs,
               "gsm8k_train_size": len(gsm_train), "math_train_size": len(math_train),
               "math_rule": "skip solutions with 2+ different boxed answers; next problem in same subject+level replaces it",
               "math_train_skipped_count": len(skipped),
               "dev_test_exact_overlaps": clash},
              open(f"{OUT}/eval_data_metadata.json", "w"), indent=2)
    print("done -> next: python make_dev_fast.py --gsm data/dev_gsm.jsonl "
          "--math data/dev_math.jsonl --out data/dev_fast.jsonl")


if __name__ == "__main__":
    main()
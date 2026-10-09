"""
Sanity-check all evaluation files.   python check_eval_data.py
Prints PASS/FAIL for each check.
"""
import json
from answer_parser import gsm_gold, math_gold, _to_number

def load(name):
    return [json.loads(l) for l in open(f"data/{name}.jsonl") if l.strip()]

ok = True
def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)

sets = {n: load(n) for n in ["dev_gsm", "dev_math", "dev_fast", "test_gsm", "test_math500"]}
ids = {n: [r["problem_id"] for r in rows] for n, rows in sets.items()}

# sizes
for n, size in [("dev_gsm", 500), ("dev_math", 500), ("dev_fast", 200), ("test_gsm", 1319), ("test_math500", 500)]:
    check(len(sets[n]) == size, f"{n}: {len(sets[n])} problems (expected {size})")

# no duplicate IDs or questions inside a file
for n, rows in sets.items():
    check(len(set(ids[n])) == len(rows), f"{n}: no duplicate IDs")
    check(len({r['question'] for r in rows}) == len(rows), f"{n}: no duplicate questions")

# dev comes from TRAIN, test from TEST
check(all(i.startswith("gsm8k_train_") for i in ids["dev_gsm"]), "dev_gsm comes from GSM8K train")
check(all(i.startswith("math_train_") for i in ids["dev_math"]), "dev_math comes from MATH train")
check(all(i.startswith("gsm8k_test_") for i in ids["test_gsm"]), "test_gsm comes from GSM8K test")
check(all(i.startswith("math500_") for i in ids["test_math500"]), "test_math500 comes from MATH-500")

# dev_fast = 100 + 100, nested inside dev
fast = set(ids["dev_fast"])
check(sum(r["benchmark"] == "gsm" for r in sets["dev_fast"]) == 100, "dev_fast has 100 GSM")
check(sum(r["benchmark"] == "math" for r in sets["dev_fast"]) == 100, "dev_fast has 100 MATH")
check(fast <= set(ids["dev_gsm"]) | set(ids["dev_math"]), "dev_fast is inside the dev sets")

# dev and test never share a question
norm = lambda s: " ".join(s.lower().split())
dev_q = {norm(r["question"]) for n in ["dev_gsm", "dev_math"] for r in sets[n]}
test_q = {norm(r["question"]) for n in ["test_gsm", "test_math500"] for r in sets[n]}
check(not (dev_q & test_q), f"dev/test question overlap: {len(dev_q & test_q)}")

# every gold answer can be read by the answer checker
FIELDS = ["problem_id", "benchmark", "source", "split", "question",
          "gold_answer", "gold_solution", "subject", "level"]
check(all(list(r) == FIELDS for rows in sets.values() for r in rows), "every row has the same fields")
bad_gsm = [r["problem_id"] for n in ["dev_gsm", "test_gsm"] for r in sets[n]
           if _to_number(r["gold_answer"]) is None]
check(not bad_gsm, f"all GSM gold answers readable ({len(bad_gsm)} bad) {bad_gsm[:3]}")
bad_math = [r["problem_id"] for n in ["dev_math", "test_math500"] for r in sets[n]
            if not str(r["gold_answer"] or "").strip()]
check(not bad_math, f"all MATH gold answers readable ({len(bad_math)} bad) {bad_math[:3]}")
from answer_parser import _boxed_contents
multi = [r["problem_id"] for r in sets["dev_math"] if len(set(_boxed_contents(r["gold_solution"]))) > 1]
check(not multi, f"no dev_math solution has 2+ different boxed answers ({len(multi)} found) {multi[:5]}")

# subject balance of dev_math (info only)
from collections import Counter
print("\ndev_math subjects:", dict(Counter(r["subject"] for r in sets["dev_math"])))
print("dev_math levels:  ", dict(sorted(Counter(r["level"] for r in sets["dev_math"]).items())))
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - paste this output")
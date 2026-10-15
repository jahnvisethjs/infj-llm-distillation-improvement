"""
Build Person 3's frozen inputs for the data team (Person 1 contract, schema person3-question-manifest-v1).

  python make_frozen_inputs.py --who "Megha Suresh (Person 3)"

Reads   data/dev_gsm.jsonl, data/dev_math.jsonl, data/dev_fast.jsonl, data/eval_data_metadata.json
Writes  handoff/frozen_development.csv            question text only (no answers)
        handoff/frozen_development_metadata.json  sources, revisions, seed, selection rules, CSV hash
        handoff/reserved_evaluation_metadata.json confirms GSM8K test + MATH-500 as final tests

Gold answers are deliberately NOT included: the data team only needs question text
to remove overlapping training examples.
"""
import argparse, csv, datetime, hashlib, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--who", required=True, help="your name, recorded as the confirmation")
ap.add_argument("--out", default="handoff")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

L = lambda n: [json.loads(l) for l in open(f"data/{n}.jsonl", encoding="utf-8") if l.strip()]
gsm, math, fast = L("dev_gsm"), L("dev_math"), L("dev_fast")
revs = json.load(open("data/eval_data_metadata.json"))["revisions"]
GSM_REV, MATH_REV, M500_REV = revs["openai/gsm8k"], revs["EleutherAI/hendrycks_math"], revs["HuggingFaceH4/MATH-500"]

gids, mids = {r["problem_id"] for r in gsm}, {r["problem_id"] for r in math}
rows = ([("gsm_dev", r) for r in gsm] + [("math_dev", r) for r in math]
        + [("gsm_fast", r) for r in fast if r["problem_id"] in gids]
        + [("math_fast", r) for r in fast if r["problem_id"] in mids])
assert len(rows) == 1200, f"expected 500+500+100+100 rows, got {len(rows)}"

csv_path = f"{args.out}/frozen_development.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["set_id", "problem_id", "problem"])
    for s, r in rows:
        w.writerow([s, r["problem_id"], r["question"]])
sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
counts = {}
for s, _ in rows:
    counts[s] = counts.get(s, 0) + 1
now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

MCONF = ("all seven subject configurations: algebra, counting_and_probability, geometry, "
         "intermediate_algebra, number_theory, prealgebra, precalculus")
def src(family, role, dataset, conf, rev, policy, parent=None):
    d = {"benchmark_family": family, "role": role, "dataset_id": dataset, "configuration": conf,
         "split": "train", "revision": rev, "origin": "person3_frozen", "selection_policy": policy}
    if parent:
        d["parent_set_id"] = parent
    return d

meta = {
    "schema_version": "person3-question-manifest-v1", "manifest_sha256": sha,
    "confirmed_by": args.who, "confirmed_utc": now, "set_counts": counts,
    "sets": {
        "gsm_dev": src("gsm8k", "development", "openai/gsm8k", "main", GSM_REV,
            {"method": "fixed_seed_sample", "seed": 1729,
             "rule": "first 500 GSM8K train problems ordered by SHA256('1729|'+problem_id)"}),
        "math_dev": src("math", "development", "EleutherAI/hendrycks_math", MCONF, MATH_REV,
            {"method": "stratified_fixed_seed_sample", "seed": 1729,
             "rule": "500 MATH train problems, proportional by subject x level (largest-remainder rounding), "
                     "each stratum ordered by SHA256('1729|'+problem_id); solutions boxing 2+ different "
                     "answers are skipped and replaced by the next problem in the same stratum"}),
        "gsm_fast": src("gsm8k", "fast_development", "openai/gsm8k", "main", GSM_REV,
            {"method": "nested_fixed_seed_sample", "seed": 1729,
             "rule": "first 100 of gsm_dev ordered by SHA256('1729|'+problem_id)"}, parent="gsm_dev"),
        "math_fast": src("math", "fast_development", "EleutherAI/hendrycks_math", MCONF, MATH_REV,
            {"method": "nested_fixed_seed_sample", "seed": 1729,
             "rule": "first 100 of math_dev ordered by SHA256('1729|'+problem_id)"}, parent="math_dev"),
    }}
json.dump(meta, open(f"{args.out}/frozen_development_metadata.json", "w"), indent=2)

# confirm the official final tests (question file itself is the data team's reserved_evaluation.csv)
reserved = {
    "schema_version": "person3-question-manifest-v1",
    "manifest_sha256": "c78f02fd1a9c87058318ee6b1c92f5ce5aeff8751fc696fa098b3fb0e477fbe4",
    "confirmed_by": args.who, "confirmed_utc": now,
    "set_counts": {"gsm8k_test": 1319, "math500_test": 500},
    "sets": {
        "gsm8k_test": {"benchmark_family": "gsm8k", "role": "final_test", "dataset_id": "openai/gsm8k",
                       "configuration": "main", "split": "test", "revision": GSM_REV,
                       "origin": "person3_confirmed_final_test",
                       "selection_policy": {"method": "all_official_test_questions"}},
        "math500_test": {"benchmark_family": "math", "role": "final_test", "dataset_id": "HuggingFaceH4/MATH-500",
                         "split": "test", "revision": M500_REV, "origin": "person3_confirmed_final_test",
                         "selection_policy": {"method": "all_official_test_questions"}},
    }}
json.dump(reserved, open(f"{args.out}/reserved_evaluation_metadata.json", "w"), indent=2)

print("set counts:", counts)
print("frozen_development.csv sha256:", sha)
print("written to", args.out + "/")

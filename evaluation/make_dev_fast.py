"""
Build dev_fast (E2): fixed 100 GSM + 100 MATH problems, nested inside the full dev sets.
Uses the same reproducible SHA256 ordering idea as Person 1 (seed 1729 for evaluation).

  python make_dev_fast.py --gsm data/dev_gsm.jsonl --math data/dev_math.jsonl --out data/dev_fast.jsonl
"""
import argparse, hashlib, json

SEED = "1729"


def key(pid):
    return hashlib.sha256(f"{SEED}|{pid}".encode()).hexdigest()


def take(path, n, id_field):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return sorted(rows, key=lambda r: key(str(r[id_field])))[:n]


ap = argparse.ArgumentParser()
ap.add_argument("--gsm", required=True)
ap.add_argument("--math", required=True)
ap.add_argument("--out", default="data/dev_fast.jsonl")
ap.add_argument("--n_each", type=int, default=100)
ap.add_argument("--id_field", default="problem_id")
args = ap.parse_args()

rows = take(args.gsm, args.n_each, args.id_field) + take(args.math, args.n_each, args.id_field)
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(args.out.replace(".jsonl", "_ids.txt"), "w") as f:
    f.write("\n".join(str(r[args.id_field]) for r in rows) + "\n")
print(f"wrote {len(rows)} problems to {args.out}")

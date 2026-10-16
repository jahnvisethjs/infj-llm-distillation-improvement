"""
Add one of Person 2's adapters to the run registry (instead of editing runs.json by hand).
  python sol/add_run.py --run_id R500_05B_s42 --model 0.5B --adapter /scratch/$USER/adapters/R500_05B_s42/final/adapter \
                        --phase curve --budget 500 --seed 42 --method nested_prefix --note "v3 data" \
                        --train_manifest_sha256 <train_n500 sha256 from Person 1 release>
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--run_id", required=True)
ap.add_argument("--model", required=True, choices=["0.5B", "1.5B"])
ap.add_argument("--adapter", required=True)
ap.add_argument("--phase", default="curve")
ap.add_argument("--budget", type=int)
ap.add_argument("--seed", type=int)
ap.add_argument("--method", default="")
ap.add_argument("--note", default="")
ap.add_argument("--train_manifest_sha256", default=None,
                help="Person 1 release hash of the manifest this adapter was trained on (strongly recommended)")
ap.add_argument("--baseline", default=None,
                help="run_id of the untrained run to compare with (e.g. base_a100 for 0.5B, base15 for 1.5B)")
ap.add_argument("--runs", default=os.path.expandvars("/scratch/$USER/eval_results/runs.json"))
a = ap.parse_args()

if not os.path.exists(os.path.join(a.adapter, "adapter_config.json")):
    sys.exit(f"no adapter_config.json in {a.adapter} -- point --adapter at the folder that contains it")
runs = json.load(open(a.runs))
if any(r["run_id"] == a.run_id for r in runs):
    sys.exit(f"{a.run_id} is already in {a.runs}")
runs.append({"run_id": a.run_id, "model": f"Qwen/Qwen2.5-{a.model}", "adapter": os.path.abspath(a.adapter),
             "phase": a.phase, "budget_examples": a.budget, "seed": a.seed, "method": a.method,
             "note": a.note, "where": "sol", "train_manifest_sha256": a.train_manifest_sha256,
             **({"baseline": a.baseline} if a.baseline else {})})
if not a.train_manifest_sha256:
    print("WARNING: no --train_manifest_sha256; the training-data check will be skipped for this run")
json.dump(runs, open(a.runs, "w"), indent=2)
print(f"added {a.run_id}; {len(runs)} runs in registry")

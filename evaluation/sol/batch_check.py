"""
fp32 batch-proof check for one model on this GPU: same 48 problems, batch 16 vs one at a time.
Exits with an error unless all 48 outputs are identical (so a job script stops before the long runs).

  python sol/batch_check.py --model Qwen/Qwen2.5-0.5B --revision <rev> --results <dir> --tag a100_05B
"""
import argparse, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

if not os.path.exists("data/batchtest.jsonl"):
    lines = open("data/dev_fast.jsonl").read().splitlines()
    open("data/batchtest.jsonl", "w").write("\n".join(lines[:24] + lines[100:124]) + "\n")

outs = {}
for bs in [16, 1]:
    out = f"{a.results}/bt_{a.tag}_b{bs}.jsonl"
    subprocess.run([sys.executable, "evaluate.py", "--eval_file", "data/batchtest.jsonl",
                    "--run_id", f"bt_{a.tag}_b{bs}", "--model", a.model, "--revision", a.revision,
                    "--batch_size", str(bs), "--out", out], check=True)
    outs[bs] = {r["problem_id"]: r for r in map(json.loads, open(out))}

same_text = sum(outs[16][i]["raw_output"] == outs[1][i]["raw_output"] for i in outs[16])
same_grade = sum(outs[16][i]["correct"] == outs[1][i]["correct"] for i in outs[16])
print(f"BATCH CHECK {a.model}: identical outputs {same_text}/48, same grading {same_grade}/48")
if same_text != 48:
    sys.exit("BATCH CHECK FAILED: outputs depend on batching on this GPU. Stop and investigate before evaluating.")

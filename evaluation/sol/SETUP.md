# Running the evaluation on ASU Sol

The Python code is unchanged; only the launcher differs. Names marked "check" must be confirmed against ASU Research Computing docs.

## One-time setup (login node)

```bash
# 1. code (your branch) — the .jsonl question files come from Drive/your Mac (scp/rsync/Globus)
git clone https://github.com/jahnvisethjs/infj-llm-distillation-improvement.git
cd infj-llm-distillation-improvement && git switch person_3-evaluation
# copy data/*.jsonl into evaluation/data/, then: python evaluation/check_eval_data.py  -> ALL CHECKS PASSED

# 2. environment (check module name)
module load mamba/latest
mamba create -n p3eval python=3.11 -y
source activate p3eval
pip install -r evaluation/requirements.txt          # torch must be a CUDA build that matches Sol

# 3. results folder + registry + pins (copy from Drive eval_results/)
mkdir -p /scratch/$USER/eval_results
#   copy runs.json and pinned_revisions.json there; edit adapter paths in runs.json to Sol paths
#   copy adapters to e.g. /scratch/$USER/adapters/<run_id>/

# 4. pre-download the pinned models (compute nodes may have no internet)
export HF_HOME=/scratch/$USER/hf_cache
python - <<'EOF'
import json, os
from huggingface_hub import snapshot_download
pins = json.load(open(os.path.expandvars("/scratch/$USER/eval_results/pinned_revisions.json")))
for model, rev in pins.items():
    snapshot_download(model, revision=rev)
    print("cached", model, rev)
EOF
```

## Every run

```bash
cd ~/infj-llm-distillation-improvement/evaluation
sbatch sol/run_eval.sbatch        # evaluates every unfinished run in runs.json
squeue -u $USER                   # check the job
tail -f logs/p3-eval-<jobid>.out  # watch progress
python run_eval.py --runs /scratch/$USER/eval_results/runs.json --results /scratch/$USER/eval_results --summary-only
```

## Consistency rule when moving from Colab (L4) to Sol

Different GPU types can change a few greedy answers. Use ONE Sol GPU type for every evaluation,
re-run the baselines (and any already-evaluated models) on it, and record the GPU in EVAL_NOTES.md.
Do not compare L4 results with Sol results. Run the fp32 batch check once on the new GPU
(48 problems, batch 16 vs 1: must be 48/48 identical).

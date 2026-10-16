#!/bin/bash
# First Sol job: checks + both baselines (0.5B and 1.5B) on one A100.
# Stops automatically if any check fails, before the long runs.
#   sbatch sol/first_run.sbatch
#SBATCH --job-name=p3-first-run
#SBATCH --account=class_cse579spring2026      # TEMPORARY (other course); change when the FSE 570 account is ready
#SBATCH --partition=general
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1                     # ONE GPU type for every evaluation
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00                       # unfinished work resumes when resubmitted
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load mamba/latest
source activate p3eval

RESULTS=/scratch/$USER/eval_results
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
cd "$HOME/infj-llm-distillation-improvement/evaluation"
mkdir -p logs

echo "== GPU";            nvidia-smi --query-gpu=name --format=csv,noheader
echo "== code version";   git log --oneline -1
echo "== tests";          python -m pytest -q test_answer_parser.py | tail -1
echo "== data";           python check_eval_data.py | tail -1

REV05=$(python -c "import json;print(json.load(open('$RESULTS/pinned_revisions.json'))['Qwen/Qwen2.5-0.5B'])")
REV15=$(python -c "import json;print(json.load(open('$RESULTS/pinned_revisions.json'))['Qwen/Qwen2.5-1.5B'])")
echo "== batch checks (must be 48/48)"
python sol/batch_check.py --model Qwen/Qwen2.5-0.5B --revision "$REV05" --results "$RESULTS" --tag a100_05B
python sol/batch_check.py --model Qwen/Qwen2.5-1.5B --revision "$REV15" --results "$RESULTS" --tag a100_15B

echo "== smoke tests"
python run_eval.py --runs "$RESULTS/runs.json" --results "$RESULTS" --where sol --smoke
echo "== baselines"
python run_eval.py --runs "$RESULTS/runs.json" --results "$RESULTS" --where sol
echo "== done"

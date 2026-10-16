#!/bin/bash
# Person 3 evaluation job for ASU Sol (SLURM).   Submit with:  sbatch sol/run_eval.sbatch
# Check partition/QOS/GPU names against ASU Research Computing docs before first use.
#SBATCH --job-name=p3-eval
#SBATCH --partition=general          # <- check Sol docs
#SBATCH --qos=public                 # <- check Sol docs
#SBATCH --gres=gpu:a100:1            # pick ONE GPU type and keep it for every evaluation
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00              # a run that hits the limit resumes when resubmitted
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load mamba/latest               # <- check Sol docs for the module name
source activate p3eval                 # conda env created once (see sol/SETUP.md)

EVAL_DIR=$HOME/infj-llm-distillation-improvement/evaluation
RESULTS=/scratch/$USER/eval_results     # runs.json, pinned_revisions.json and all result files
export HF_HOME=/scratch/$USER/hf_cache  # models pre-downloaded here on a login node
export HF_HUB_OFFLINE=1                 # compute nodes: never download, use the cache only

cd "$EVAL_DIR"
mkdir -p logs
nvidia-smi --query-gpu=name --format=csv,noheader
python -m pytest -q test_answer_parser.py | tail -1
python check_eval_data.py | tail -1

python run_eval.py --runs "$RESULTS/runs.json" --results "$RESULTS" --where sol --smoke
python run_eval.py --runs "$RESULTS/runs.json" --results "$RESULTS" --where sol

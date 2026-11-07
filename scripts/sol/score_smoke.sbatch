#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --job-name=c0-score-smoke
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Eight-row real scoring check on this GPU type; inspect its runtime report before sizing.
# Args: CONFIG [PROFILE]. The full scoring job also runs this smoke itself when no
# matching proof exists for its GPU/runtime, so this job is optional but recommended.
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" scoring-smoke "$@"

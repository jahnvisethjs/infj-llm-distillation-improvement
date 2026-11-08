#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=c0-train
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# One B or C0 arm per job and continuation seed, always initialized from the original parent.
# Args: CONFIG METHOD NEW_ATTEMPT [PROFILE] [OWN_ARM_RESUME_CHECKPOINT] [TRAINING_SEED].
# One hour: the 5,000-pool runs took 6 to 8 minutes each (123 to 124 updates) on an A100.
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" train "$@"

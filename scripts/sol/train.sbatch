#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=c0-train
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# One B or C0 arm per job, always initialized from the original parent.
# Args: CONFIG METHOD NEW_ATTEMPT [PROFILE] [OWN_ARM_RESUME_CHECKPOINT].
# Four hours is an initial cap; tune from the actual training smoke.
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" train "$@"

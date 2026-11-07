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
#SBATCH --job-name=c0-train-smoke
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Disposable, at most four selected C0 rows. Cannot complete an experimental arm.
# Args: CONFIG NEW_SMOKE_ATTEMPT [PROFILE].
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" training-smoke "$@"

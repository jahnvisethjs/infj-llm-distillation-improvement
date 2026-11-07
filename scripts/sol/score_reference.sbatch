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
#SBATCH --job-name=c0-score
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Initial cap, not a measured completion estimate. Runs the in-allocation smoke first
# when no matching proof exists; cached rows are reused after an interrupted pass.
# Args: CONFIG [PROFILE].
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" score-reference "$@"

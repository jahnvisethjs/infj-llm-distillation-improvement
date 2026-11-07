#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=c0-finalize
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# CPU-only reconciliation/handoff. Accuracy evaluation remains pending.
# Args: CONFIG FULL_ARM_ATTEMPT [PROFILE]. Both arms must use that attempt name.
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" finalize "$@"

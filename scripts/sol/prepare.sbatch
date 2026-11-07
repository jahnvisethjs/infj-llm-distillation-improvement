#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=c0-prepare
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# CPU-only audit, matching and token verification; no GPU is requested.
# Args: CONFIG [PROFILE].
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/batch_stage.sh" prepare "$@"

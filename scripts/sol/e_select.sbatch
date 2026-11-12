#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=e-select
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/e_batch_stage.sh" select "$@"

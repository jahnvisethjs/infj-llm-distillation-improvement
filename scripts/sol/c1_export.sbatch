#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=c1-export
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# C1 export stage. First argument is CONFIG; remaining arguments pass to the dispatcher.
set -euo pipefail
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/sol/c1_batch_stage.sh" export "$@"


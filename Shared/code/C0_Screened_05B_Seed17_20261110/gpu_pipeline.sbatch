#!/bin/bash
#SBATCH --account=class_fse57081354fall2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=c0-screen-05b-p17
set -euo pipefail
package="${1:?Supply the absolute uploaded package directory}"
shift
training_python="/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
printf 'Job=%s Host=%s\n' "${SLURM_JOB_ID:-interactive}" "${HOSTNAME:-unknown}"
exec "$training_python" -B "$package/screened_c0.py" pipeline "$@"

#!/bin/bash
set -euo pipefail
package="${1:?package directory}"
work="${2:?work directory}"
broot="${3:?B root}"
submission="${4:?submission ledger}"
slot="${5:?allocation index}"
worker_seconds="${6:?bounded worker seconds}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
printf 'C0/C1 TRAINING ONLY Job=%s Host=%s Slot=%s\n' "${SLURM_JOB_ID:-unknown}" "${HOSTNAME:-unknown}" "$slot"
exec "$broot/venv_b_training_py312/bin/python" -B "$package/nightly.py" worker --work "$work" --b-root "$broot" --submission "$submission" --slot "$slot" --worker-seconds "$worker_seconds"

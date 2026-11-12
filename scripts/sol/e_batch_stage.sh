#!/usr/bin/env bash
# E stages reuse upstream artifacts and never launch grading or benchmark evaluation.
set -euo pipefail
stage="${1:?Supply an E stage}"
config="${2:?Supply the original C1 SOL config}"
shift 2
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd -- "$repo"
source scripts/sol/common.sh
c0_activate_env
: "${C0_PYTHON:?Export C0_PYTHON or C0_ENV}"
[[ -x "$C0_PYTHON" && -f "$config" ]] || { printf '%s\n' 'Missing Python or config' >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
case "$stage" in
  select|train|finalize)
    exec "$C0_PYTHON" -B -m capstone_person4_approachE "$stage" --config "$config" "$@" ;;
  *) printf '%s\n' "Unknown E stage: $stage" >&2; exit 2 ;;
esac

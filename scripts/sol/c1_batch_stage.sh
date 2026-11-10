#!/usr/bin/env bash
# Spool-safe C1 dispatch; generation is performed externally by Person 3.
set -euo pipefail
stage="${1:?Supply a C1 stage}"
config="${2:?Supply the SOL configuration}"
shift 2
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd -- "$repo"
source scripts/sol/common.sh
c0_activate_env
: "${C0_PYTHON:?Export C0_PYTHON or C0_ENV}"
[[ -x "$C0_PYTHON" && -f "$config" ]] || { printf '%s\n' 'Missing Python or configuration' >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
printf 'C1 stage=%s Job=%s Config=%s\n' "$stage" "${SLURM_JOB_ID:-local}" "$config"
case "$stage" in
  export)
    exec "$C0_PYTHON" -B -m capstone_person4 export-generation-request --config "$config" "$@" ;;
  ingest)
    predictions="${1:?Supply Person 3 predictions}"; report="${2:?Supply Person 3 run config}"; shift 2
    exec "$C0_PYTHON" -B -m capstone_person4 ingest-generation --config "$config" --predictions "$predictions" --report "$report" "$@" ;;
  select|train|finalize)
    exec "$C0_PYTHON" -B scripts/sol/c1_stage.py "$stage" --config "$config" "$@" ;;
  *) printf '%s\n' "Unknown C1 stage: $stage" >&2; exit 2 ;;
esac

#!/bin/bash
# Dispatch one batch stage. Slurm copies .sbatch scripts into its spool directory,
# so repository discovery uses C0_REPO/SLURM_SUBMIT_DIR, never the copied script.
set -euo pipefail
stage="${1:?Supply a batch stage}"
config="${2:?Supply the SOL configuration}"
shift 2
repo="${C0_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd -- "$repo"
source scripts/sol/common.sh
c0_activate_env
: "${C0_PYTHON:?Export C0_PYTHON (absolute python of the compatible environment) or C0_ENV (mamba environment name)}"
[[ -x "$C0_PYTHON" && -f "$config" ]] || { printf '%s\n' 'Missing C0_PYTHON executable or SOL configuration' >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
printf 'Stage=%s Job=%s Host=%s Config=%s Python=%s\n' "$stage" "${SLURM_JOB_ID:-local}" "${HOSTNAME:-unknown}" "$config" "$C0_PYTHON"
case "$stage" in
  prepare)
    exec bash scripts/sol/prepare.sh "$config" "${1:-}" ;;
  scoring-smoke|score-reference)
    exec bash scripts/sol/gpu_stage.sh "$config" "$stage" "" "${1:-}" ;;
  select)
    c0_set_profile "${1:-}"
    "$C0_PYTHON" -B -m capstone_person4 diagnose --config "$config" "${c0_profile_args[@]}"
    "$C0_PYTHON" -B -m capstone_person4 select --config "$config" --method B "${c0_profile_args[@]}"
    exec "$C0_PYTHON" -B -m capstone_person4 select --config "$config" --method C0 "${c0_profile_args[@]}" ;;
  training-smoke)
    attempt="${1:?Supply a new smoke attempt name}"
    exec bash scripts/sol/gpu_stage.sh "$config" training-smoke "$attempt" "${2:-}" ;;
  train)
    method="${1:?Supply B or C0}"
    attempt="${2:?Supply a new training attempt name}"
    case "$method" in B|C0) ;; *) printf '%s\n' 'Training method must be B or C0' >&2; exit 2 ;; esac
    c0_set_profile "${3:-}"
    resume_args=()
    if [[ -n "${4:-}" ]]; then resume_args=(--resume "$4"); fi
    seed_args=()
    if [[ -n "${5:-}" ]]; then seed_args=(--training-seed "$5"); fi
    exec "$C0_PYTHON" -B -m capstone_person4 train --config "$config" --method "$method" --attempt "$attempt" \
      "${c0_profile_args[@]}" "${resume_args[@]}" "${seed_args[@]}" ;;
  finalize)
    attempt="${1:?Supply the full-arm attempt name used for both B and C0}"
    # Resolve fingerprinted run locations using the package, without guessed paths.
    # One finalize job covers every continuation seed in C0_SEEDS (default: the config's seed).
    c0_set_profile "${2:-}"
    if [[ -z "${C0_SEEDS:-}" ]]; then
        exec "$C0_PYTHON" -B scripts/sol/finalize_attempt.py --config "$config" --attempt "$attempt" "${c0_profile_args[@]}"
    fi
    for seed in $C0_SEEDS; do
        "$C0_PYTHON" -B scripts/sol/finalize_attempt.py --config "$config" --attempt "$attempt" --training-seed "$seed" "${c0_profile_args[@]}"
    done ;;
  *) printf '%s\n' "Unknown batch stage: $stage" >&2; exit 2 ;;
esac

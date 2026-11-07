#!/usr/bin/env bash
set -euo pipefail
: "${C0_PYTHON:?Set the actual SOL executable}"
: "${C0_ACCOUNT:?Set the allocated SOL account}"
: "${C0_PARTITION:?Set the actual partition}"
: "${C0_WALLTIME:?Set walltime based on measured throughput}"
: "${C0_MEMORY:?Set measured memory request, e.g. a Slurm memory value}"
: "${C0_GRES:?Set one compatible NVIDIA GPU request}"
config="${1:?Supply the SOL configuration}"
stage="${2:?Supply the GPU stage}"
attempt="${3:-}"
dependency="${4:-}"
profile="${5:-}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
c0_set_profile "$profile"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo"
[[ -x "$C0_PYTHON" && -f "$config" ]] || { printf '%s\n' 'Missing Python or config' >&2; exit 2; }
case "$stage" in
  scoring-smoke) ;;
  score-reference|training-smoke|train-B|train-C0)
    [[ "$dependency" =~ ^[0-9]+$ ]] || { printf '%s\n' 'Supply the successful prerequisite job ID as argument 4' >&2; exit 2; } ;;
  *) printf '%s\n' "Unknown stage: $stage" >&2; exit 2 ;;
esac
case "$stage" in
  training-smoke|train-B|train-C0) : "${attempt:?Supply a new training attempt}" ;;
esac
run_name="$(basename -- "$config" .json)${profile:+-$profile}"
args=(--parsable --account "$C0_ACCOUNT" --partition "$C0_PARTITION" --time "$C0_WALLTIME"
      --mem "$C0_MEMORY" --gres "$C0_GRES" --chdir "$repo" --job-name "c0-$run_name-$stage")
if [[ -n "${C0_QOS:-}" ]]; then args+=(--qos "$C0_QOS"); fi
if [[ -n "$dependency" ]]; then args+=(--dependency "afterok:$dependency"); fi
mkdir -p -- outputs/person4/sol-logs
args+=(--output "$repo/outputs/person4/sol-logs/%x-%j.out" --error "$repo/outputs/person4/sol-logs/%x-%j.err")
# Calling this script is the explicit submission step; nothing auto-submits downstream jobs.
sbatch "${args[@]}" scripts/sol/gpu_stage.sh "$config" "$stage" "$attempt" "$profile"

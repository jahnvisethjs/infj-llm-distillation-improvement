#!/bin/bash
# Submit explicitly with sbatch. Stops after the small pilot; no full run or installs.
set -euo pipefail
: "${ROUTING_DIR:?Export the original experiment data directory}"
: "${ROUTING_PY:?Export the unchanged native evaluator Python}"
: "${ROUTING_PILOT_PY:?Export the separate pilot fitting Python}"
: "${ROUTING_ENCODER_CACHE:?Export the prefetched MiniLM cache directory}"
# Slurm executes a spool copy: BASH_SOURCE identifies that copy, not deployment.
# The submitter exports the deployment; --chdir supplies the fallback working dir.
pilot_code="${ROUTING_PILOT_CODE:-$PWD}"
[[ -f "$pilot_code/capstone_routing/pilot.py" ]] || {
    printf 'Missing deployed routing package in %s; export ROUTING_PILOT_CODE or submit with --chdir.\n' "$pilot_code" >&2
    exit 2
}
cd -- "$pilot_code"
pilot_plan="${PILOT_PLAN:-$ROUTING_DIR/results/pilot_plan01}"
pilot_labels="${PILOT_LABELS:-$ROUTING_DIR/results/pilot_labels01}"
pilot_report="${PILOT_REPORT:-$ROUTING_DIR/results/pilot_report01}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_CACHE=/scratch/sshind38/capstone_approach_b/hf_cache
pilot_reuse=()
for prior in "$ROUTING_DIR/results/labels_train01" "$ROUTING_DIR/results/labels_train02" "${PILOT_REUSE:-}"; do
    if [[ -n "$prior" && -d "$prior" ]]; then
        pilot_reuse+=(--reuse-labels "$prior")
    fi
done
"$ROUTING_PY" -B -m capstone_routing.pilot label \
    --plan "$pilot_plan" --config "$ROUTING_DIR/configs/routing.pool.local-sources.json" \
    --output "$pilot_labels" "${pilot_reuse[@]}" --authorize-gpu

export HF_HUB_CACHE="$ROUTING_ENCODER_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$ROUTING_ENCODER_CACHE"
"$ROUTING_PILOT_PY" -B -m capstone_routing.pilot fit \
    --plan "$pilot_plan" --labels "$pilot_labels" --output "$pilot_report"
printf '\nPilot finished. Read: %s/report.md\n' "$pilot_report"

#!/bin/bash
# Explicit submission of one bounded pilot job. No automatic full-run continuation.
set -euo pipefail
: "${ROUTING_DIR:?Export the original experiment data directory}"
: "${ROUTING_PY:?Export the unchanged native evaluator Python}"
pilot_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROUTING_PILOT_CODE="$pilot_code"
cd -- "$pilot_code"
export ROUTING_PILOT_PY="${ROUTING_PILOT_PY:-${ROUTING_PILOT_ENV:-$ROUTING_DIR/venv_pilot_v3}/bin/python}"
export ROUTING_ENCODER_CACHE="${ROUTING_ENCODER_CACHE:-$ROUTING_DIR/inputs/pilot_encoder_cache}"
export PILOT_PLAN="${PILOT_PLAN:-$ROUTING_DIR/results/pilot_plan01}"
export PILOT_LABELS="${PILOT_LABELS:-$ROUTING_DIR/results/pilot_labels01}"
export PILOT_REPORT="${PILOT_REPORT:-$ROUTING_DIR/results/pilot_report01}"
"$ROUTING_PY" -B -c 'import sys; from capstone_routing.pilot import load_plan; p, _, _, _ = load_plan(sys.argv[1]); print("Pilot counts:", p["actual_counts"], "maximum answers:", p["maximum_responses"])' "$PILOT_PLAN"
[[ -x "$ROUTING_PILOT_PY" ]] || { printf 'Run routing_pilot_setup.sh first\n' >&2; exit 2; }
if [[ -d "$PILOT_REPORT" ]]; then
    printf 'Report directory already exists; inspect it before submitting another job: %s\n' "$PILOT_REPORT" >&2
    exit 2
fi
if [[ -d "$PILOT_LABELS" && ! -f "$PILOT_LABELS/receipt.json" ]]; then
    printf 'Interrupted labels preserved. Set PILOT_REUSE to %s and choose new PILOT_LABELS/PILOT_REPORT paths.\n' "$PILOT_LABELS" >&2
    exit 2
fi
mkdir -p -- "$ROUTING_DIR/logs"
pilot_job=$(sbatch --parsable --job-name=routing-pilot \
    --account=class_fse57081354fall2026 --partition=public --qos=class \
    --gpus=a100:1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 \
    --chdir="$pilot_code" --output="$ROUTING_DIR/logs/routing-pilot-%j.log" \
    --export=ALL "$pilot_code/scripts/sol/routing_pilot_worker.sh")
printf 'Submitted batch job %s\nYou may disconnect now.\nReport: %s/report.md\nLog: %s/logs/routing-pilot-%s.log\n' \
    "$pilot_job" "$PILOT_REPORT" "$ROUTING_DIR" "${pilot_job%%;*}"

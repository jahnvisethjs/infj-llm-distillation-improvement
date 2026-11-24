#!/bin/bash
# CPU setup on a login node: separate environment/cache and a sealed small plan.
set -euo pipefail
: "${ROUTING_DIR:?Export the original experiment data directory}"
: "${ROUTING_PY:?Export the unchanged native evaluator Python}"
pilot_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$pilot_code"
pilot_env="${ROUTING_PILOT_ENV:-$ROUTING_DIR/venv_pilot_v3}"
pilot_plan="${PILOT_PLAN:-$ROUTING_DIR/results/pilot_plan01}"
export ROUTING_PILOT_PY="${ROUTING_PILOT_PY:-$pilot_env/bin/python}"
export ROUTING_ENCODER_CACHE="${ROUTING_ENCODER_CACHE:-$ROUTING_DIR/inputs/pilot_encoder_cache}"
if [[ ! -e "$pilot_env" ]]; then
    "$ROUTING_PY" -m venv "$pilot_env"
fi
[[ -x "$ROUTING_PILOT_PY" ]] || { printf 'Missing pilot Python: %s\n' "$ROUTING_PILOT_PY" >&2; exit 2; }
[[ "$ROUTING_PILOT_PY" != "$ROUTING_PY" ]] || { printf 'Pilot and native Python must be separate\n' >&2; exit 2; }
# A separate CPU build avoids installing CUDA dependencies or changing native Python.
"$ROUTING_PILOT_PY" -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"$ROUTING_PILOT_PY" -m pip install -r "$pilot_code/requirements-routing.txt"
HF_HUB_CACHE="$ROUTING_ENCODER_CACHE" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
SENTENCE_TRANSFORMERS_HOME="$ROUTING_ENCODER_CACHE" \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$ROUTING_PILOT_PY" -B -c 'from capstone_routing.router import QuestionEncoder; v = QuestionEncoder().encode("Pilot encoder readiness check."); assert len(v) == 384; print("Pinned MiniLM cached; CPU encoder ready.")'
if [[ -e "$pilot_plan" ]]; then
    "$ROUTING_PY" -B -c 'import sys; from capstone_routing.pilot import load_plan; p, _, _, _ = load_plan(sys.argv[1]); print("Existing verified pilot plan:", p["actual_counts"])' "$pilot_plan"
else
    "$ROUTING_PY" -B -m capstone_routing.pilot prepare \
        --registry "$ROUTING_DIR/results/registry_sources01" \
        --pool "$ROUTING_DIR/results/pool01" --output "$pilot_plan"
fi
printf '\nCPU setup finished. No GPU job submitted. Next: bash scripts/sol/routing_pilot_submit.sh\n'

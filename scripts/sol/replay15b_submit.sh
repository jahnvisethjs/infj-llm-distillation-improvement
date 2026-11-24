#!/bin/bash
# Explicit opt-in: validates inputs, then submits the bounded pilot chain.
set -euo pipefail
: "${REPLAY15B_REPO:?Export the actual repository directory on Sol}"
: "${REPLAY15B_PYTHON:=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python}"
pilot_config="${1:?Pass the generated pilot configuration}"
shift
cd -- "$REPLAY15B_REPO"
"$REPLAY15B_PYTHON" -B -m capstone_person4_replay_c1_15b preflight --config "$pilot_config"
exec "$REPLAY15B_PYTHON" -B -m capstone_person4_replay_c1_15b submit --config "$pilot_config" "$@"

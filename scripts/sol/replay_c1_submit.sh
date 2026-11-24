#!/bin/bash
# Explicit opt-in: validates inputs, then submits the bounded pilot chain.
set -euo pipefail
: "${REPLAY_C1_REPO:?Export the actual repository directory on Sol}"
: "${REPLAY_C1_PYTHON:=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python}"
pilot_config="${1:?Pass the generated pilot configuration}"
shift
cd -- "$REPLAY_C1_REPO"
"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 preflight --config "$pilot_config"
exec "$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 submit --config "$pilot_config" "$@"

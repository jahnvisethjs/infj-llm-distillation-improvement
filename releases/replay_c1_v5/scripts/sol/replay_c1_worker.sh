#!/bin/bash
# Invoked by the explicitly authorized submission command. No installs or cleanup.
set -euo pipefail
pilot_python="${1:?training Python required}"
pilot_repo="${2:?repository required}"
pilot_config="${3:?configuration required}"
shift 3
pilot_stage="${1:?stage required}"
shift
cd -- "$pilot_repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$pilot_python" -B -m capstone_person4_approachReplayC1 "$pilot_stage" --config "$pilot_config" "$@"

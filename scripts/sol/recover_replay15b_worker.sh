#!/bin/bash
# Separate evaluation-only job. Never invoke the training CLI or edit its source.
set -euo pipefail
recovery_python="${1:?training/evaluation Python required}"
recovery_root="${2:?recovery tool directory required}"
original_repo="${3:?original v1 directory required}"
original_config="${4:?original parent17 config required}"
training_attempt="${5:?training attempt required}"
source_evaluation="${6:?source evaluation attempt required}"
target_evaluation="${7:?new evaluation attempt required}"
expected_identity="${8:?original training identity required}"
cd -- "$recovery_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$recovery_python" -B -m capstone_replay15b_recovery run \
  --repo "$original_repo" --config "$original_config" \
  --training-attempt "$training_attempt" \
  --source-evaluation-attempt "$source_evaluation" \
  --evaluation-attempt "$target_evaluation" \
  --expected-identity "$expected_identity" --authorize-gpu

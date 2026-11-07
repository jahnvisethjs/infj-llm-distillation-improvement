#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
c0_activate_env
: "${C0_PYTHON:?Set C0_PYTHON to the compatible SOL Python executable, or C0_ENV to a mamba environment}"
config="${1:?Supply the SOL configuration}"
stage="${2:?Supply scoring-smoke, score-reference, training-smoke, train-B, or train-C0}"
attempt="${3:-}"
c0_set_profile "${4:-}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo"
case "$stage" in
  scoring-smoke) args=(score-reference --smoke) ;;
  score-reference) args=(score-reference) ;;
  training-smoke) args=(train --method C0 --smoke --attempt "${attempt:?Supply a new smoke attempt}") ;;
  train-B) args=(train --method B --attempt "${attempt:?Supply a new B attempt}") ;;
  train-C0) args=(train --method C0 --attempt "${attempt:?Supply a new C0 attempt}") ;;
  *) printf '%s\n' "Unknown GPU stage: $stage" >&2; exit 2 ;;
esac
# Runtime, parent, recipe and matching artifact gates run inside each command.
"$C0_PYTHON" -B -m capstone_person4 "${args[@]}" --config "$config" "${c0_profile_args[@]}"

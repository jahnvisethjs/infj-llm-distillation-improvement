#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
c0_activate_env
: "${C0_PYTHON:?Set C0_PYTHON to the compatible SOL Python executable, or C0_ENV to a mamba environment}"
config="${1:?Supply the configuration generated for actual SOL locations}"
c0_set_profile "${2:-}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo"
[[ -x "$C0_PYTHON" && -f "$config" ]] || { printf '%s\n' 'Missing executable or configuration' >&2; exit 2; }
"$C0_PYTHON" -B -m capstone_person4 audit --config "$config" --target inputs "${c0_profile_args[@]}"

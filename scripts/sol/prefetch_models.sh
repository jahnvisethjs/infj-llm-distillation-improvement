#!/usr/bin/env bash
# Run ONCE on a SOL login node (internet) before GPU jobs: fill the Hugging Face cache
# with the pinned base models named in the configuration. Compute jobs then run with
# HF_HUB_OFFLINE=1. Nothing is trained, scored or submitted here.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ -n "${C0_ENV:-}" ]]; then module load mamba/latest; source activate "$C0_ENV"; : "${C0_PYTHON:=$(command -v python)}"; fi
: "${C0_PYTHON:?Set C0_PYTHON or C0_ENV}"
config="${1:?Supply the SOL configuration}"
export HF_HUB_OFFLINE=0
"$C0_PYTHON" -B - "$config" <<'PY'
import json, sys
from huggingface_hub import snapshot_download
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
for name, profile in cfg["profiles"].items():
    path = snapshot_download(profile["model"], revision=profile["revision"],
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    print(name, profile["model"], profile["revision"], "->", path)
PY

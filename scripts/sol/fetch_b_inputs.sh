#!/usr/bin/env bash
# Bundle Method B's provenance files (no weights, logs or optimizer state) with SHA256s for the C0/C1 rerun.
# Run on a Sol login node:  bash fetch_b_inputs.sh B_ROOT [OUT_DIR]
set -euo pipefail
ROOT=${1:-/scratch/sshind38/capstone_approach_b/outputs/B_method_B_document_v3}
OUT=${2:-$HOME/b_inputs_$(date -u +%Y%m%dT%H%M%SZ)}
KEEP='bank|audit|strat|selected|budget|receipt|config|sha|adapter|evaluation|development|length_bins|seen_groups|priorit|completion|target_results|progress|train_B|rebinned|original_manifest|runtime/|tokenization|prompt_template|protocol|registry|verification|label_checks|summary'
DROP='\.(safetensors|bin|pt|pth|pyc)$|/logs/|worker\.lock'
[ -d "$ROOT" ] || { echo "missing B root: $ROOT" >&2; exit 2; }
mkdir -p "$OUT/files"
find "$ROOT" -type f | grep -Ei "$KEEP" | grep -Ev "$DROP" | sort > "$OUT/file_list.txt"
[ -s "$OUT/file_list.txt" ] || { echo "no matching files under $ROOT" >&2; exit 2; }
xargs -d '\n' cp --parents -t "$OUT/files" < "$OUT/file_list.txt"
(cd "$OUT/files" && find . -type f -exec sha256sum {} + | sort -k2 > "$OUT/SHA256SUMS.txt")
find "$ROOT" -type f -name 'adapter_model.safetensors' -exec sha256sum {} + | sort -k2 > "$OUT/ADAPTER_SHA256SUMS.txt"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "bundle: $OUT.tar.gz ($(wc -l < "$OUT/file_list.txt") files)"

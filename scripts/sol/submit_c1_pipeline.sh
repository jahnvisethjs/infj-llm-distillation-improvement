#!/usr/bin/env bash
# One parent per invocation: ingest -> gated selection -> smoke -> three C1 runs -> handoff.
# Requires completed matching C0 runs, an exported request and Person 3 predictions.
# CONFIG TAG PREDICTIONS REPORT [C0_ATTEMPT] [C1_ATTEMPT] [SMOKE_ATTEMPT]
set -euo pipefail
config="${1:?Supply the SOL configuration}"
tag="${2:?Supply a short job tag}"
predictions="${3:?Supply Person 3 predictions}"
report="${4:?Supply Person 3 run config}"
c0_attempt="${5:-pilot01}"
attempt="${6:-c1pilot01}"
smoke="${7:-c1smoke01}"
: "${C0_REPO:?Export the repository directory on SOL}"
cd -- "$C0_REPO"
source scripts/sol/common.sh
c0_activate_env
: "${C0_PYTHON:?Export C0_PYTHON or C0_ENV}"
[[ -x "$C0_PYTHON" && -f "$config" && -f "$predictions" && -f "$report" ]] || {
    printf '%s\n' 'Missing Python, config, predictions or report' >&2; exit 2;
}
# Resolve before submission so batch jobs are independent of the caller's directory.
config="$(realpath -- "$config")"
predictions="$(realpath -- "$predictions")"
report="$(realpath -- "$report")"
read -r -a seeds <<< "${C1_SEEDS:-42 137 2026}"
(( ${#seeds[@]} > 0 && ${#seeds[@]} + 4 <= 10 )) || {
    printf '%s\n' 'C1 chain needs 1 to 6 seeds (at most ten jobs)' >&2; exit 2;
}
declare -A seen=()
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || { printf '%s\n' "Invalid continuation seed: $seed" >&2; exit 2; }
    [[ -z "${seen[$seed]:-}" ]] || { printf '%s\n' "Duplicate continuation seed: $seed" >&2; exit 2; }
    seen[$seed]=1
done
"$C0_PYTHON" -B scripts/sol/c1_stage.py preflight --config "$config" --predictions "$predictions" --report "$report" \
    --c0-attempt "$c0_attempt" --attempt "$attempt" --smoke-attempt "$smoke" --seeds "${seeds[@]}"
read -r -a extra <<< "${C1_SBATCH_ARGS:-${C0_SBATCH_ARGS:-}}"
submit() {
    local name="$1" dependency="$2" script="$3"; shift 3
    local args=(--parsable --job-name "$tag-$name" "${extra[@]}")
    if [[ -n "$dependency" ]]; then args+=(--dependency "afterok:$dependency"); fi
    local job
    job=$(sbatch "${args[@]}" "scripts/sol/$script" "$config" "$@")
    job="${job%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]] || { printf '%s\n' "Invalid sbatch job ID: $job" >&2; return 2; }
    printf 'Submitted %s=%s\n' "$name" "$job" >&2
    printf '%s\n' "$job"
}
ingest=$(submit ingest "" c1_ingest.sbatch "$predictions" "$report")
selection=$(submit select "$ingest" c1_select.sbatch)
smoke_job=$(submit smoke "$selection" c1_train_smoke.sbatch --c0-attempt "$c0_attempt" --attempt "$smoke" --training-seed "${seeds[0]}")
jobs=()
for seed in "${seeds[@]}"; do
    job=$(submit "train-s$seed" "$smoke_job" c1_train.sbatch --c0-attempt "$c0_attempt" --attempt "$attempt" --training-seed "$seed")
    jobs+=("$job")
    printf 'C1-s%s=%s\n' "$seed" "$job"
done
dependencies=$(IFS=:; printf '%s' "${jobs[*]}")
final=$(submit finalize "$dependencies" c1_finalize.sbatch --c0-attempt "$c0_attempt" --attempt "$attempt" --seeds "${seeds[@]}")
printf 'ingest=%s\nselect=%s\nsmoke=%s\nfinalize=%s\n' "$ingest" "$selection" "$smoke_job" "$final"

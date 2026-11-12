#!/usr/bin/env bash
# CONFIG TAG [C0_ATTEMPT] [C1_ATTEMPT] [E_ATTEMPT] [SMOKE_ATTEMPT]
# Run only after explicit GPU-experiment authorization. One parent per invocation.
set -euo pipefail
config="${1:?Supply the completed C1 SOL config}"
tag="${2:?Supply a short job tag}"
c0_attempt="${3:-pilot01}"
c1_attempt="${4:-c1pilot01}"
attempt="${5:-epilot01}"
smoke="${6:-esmoke01}"
: "${C0_REPO:?Export the repository directory on SOL}"
cd -- "$C0_REPO"
source scripts/sol/common.sh
c0_activate_env
: "${C0_PYTHON:?Export C0_PYTHON or C0_ENV}"
[[ -x "$C0_PYTHON" && -f "$config" ]] || { printf '%s\n' 'Missing Python or config' >&2; exit 2; }
config="$(realpath -- "$config")"
read -r -a seeds <<< "${E_SEEDS:-42 137 2026}"
(( ${#seeds[@]} > 0 && ${#seeds[@]} + 3 <= 10 )) || { printf '%s\n' 'E chain requires 1 to 7 seeds' >&2; exit 2; }
declare -A seen=()
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^(0|[1-9][0-9]*)$ && -z "${seen[$seed]:-}" ]] || { printf '%s\n' 'Invalid/duplicate E seed' >&2; exit 2; }
    seen[$seed]=1
done
for attempt_name in "$c0_attempt" "$c1_attempt" "$attempt" "$smoke"; do
    [[ "$attempt_name" != *[/\\]* && "$attempt_name" != . && "$attempt_name" != .. && -n "$attempt_name" ]] || { printf '%s\n' 'Use simple attempt names' >&2; exit 2; }
done
"$C0_PYTHON" -B -m capstone_person4_approachE preflight --config "$config" \
    --c0-attempt "$c0_attempt" --c1-attempt "$c1_attempt" --seeds "${seeds[@]}"
read -r -a extra <<< "${E_SBATCH_ARGS:-${C0_SBATCH_ARGS:-}}"
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
selection=$(submit select "" e_select.sbatch)
smoke_job=$(submit smoke "$selection" e_train_smoke.sbatch --attempt "$smoke" --smoke --training-seed "${seeds[0]}")
jobs=()
for seed in "${seeds[@]}"; do
    job=$(submit "train-s$seed" "$smoke_job" e_train.sbatch --attempt "$attempt" --training-seed "$seed")
    jobs+=("$job")
done
dependencies=$(IFS=:; printf '%s' "${jobs[*]}")
final=$(submit finalize "$dependencies" e_finalize.sbatch --attempt "$attempt" --c0-attempt "$c0_attempt" --c1-attempt "$c1_attempt" --seeds "${seeds[@]}")
printf 'select=%s\nsmoke=%s\nfinalize=%s\n' "$selection" "$smoke_job" "$final"

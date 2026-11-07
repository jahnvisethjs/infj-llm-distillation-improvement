#!/usr/bin/env bash
# Submit the complete chain for ONE configuration with afterok dependencies:
#   prepare -> score-smoke -> score -> select -> train-smoke -> C0 [and B] -> finalize
# C0 is Person 4's arm; B (random baseline) is trained here only when C0_ARMS="B C0",
# otherwise the selection job still writes the matched-budget B manifest for whoever
# trains B. Seven or eight jobs stay within the class QoS limit of ten active
# submissions; at most two run at once. Calling this script is the explicit submission
# step; a failed job stops its dependents (afterok).
#   bash scripts/sol/submit_pipeline.sh CONFIG TAG [FULL_ATTEMPT] [SMOKE_ATTEMPT]
# Extra sbatch options for every job (for example --mail-user) go in C0_SBATCH_ARGS.
set -euo pipefail
config="${1:?Supply the SOL configuration}"
tag="${2:?Supply a short job tag, e.g. q05-s17}"
attempt="${3:-pilot01}"
smoke="${4:-smoke01}"
: "${C0_REPO:?Export the repository directory on SOL}"
cd -- "$C0_REPO"
[[ -f "$config" ]] || { printf '%s\n' "Missing configuration: $config" >&2; exit 2; }
if [[ -z "${C0_PYTHON:-}" && -z "${C0_ENV:-}" ]]; then
    printf '%s\n' 'Export C0_PYTHON or C0_ENV so batch jobs inherit the environment' >&2; exit 2
fi
read -r -a extra <<< "${C0_SBATCH_ARGS:-}"
submit() {  # submit NAME DEPENDENCY SCRIPT ARGS...
    local name="$1" dependency="$2" script="$3"; shift 3
    local args=(--parsable --job-name "$tag-$name" "${extra[@]}")
    if [[ -n "$dependency" ]]; then args+=(--dependency "afterok:$dependency"); fi
    sbatch "${args[@]}" "scripts/sol/$script" "$config" "$@"
}
prep=$(submit prep "" prepare.sbatch)
score_smoke=$(submit score-smoke "$prep" score_smoke.sbatch)
score=$(submit score "$score_smoke" score_reference.sbatch)
select_job=$(submit select "$score" select.sbatch)
train_smoke=$(submit train-smoke "$select_job" train_smoke.sbatch "$smoke")
arm_jobs=()
for arm in ${C0_ARMS:-C0}; do
    case "$arm" in B|C0) ;; *) printf '%s\n' "C0_ARMS entries must be B or C0: $arm" >&2; exit 2 ;; esac
    job=$(submit "$arm" "$train_smoke" train.sbatch "$arm" "$attempt")
    arm_jobs+=("$job"); printf '%s\n' "$arm=$job"
done
final=$(submit finalize "$(IFS=:; printf '%s' "${arm_jobs[*]}")" finalize.sbatch "$attempt")
printf '%s\n' "prepare=$prep" "score_smoke=$score_smoke" "score=$score" "select=$select_job" \
    "train_smoke=$train_smoke" "finalize=$final"
printf 'Monitor: squeue -u "%s" -o "%%.18i %%.30j %%.2t %%.10M %%.10l %%R"\n' "${USER:-$(id -un)}"

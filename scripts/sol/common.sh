#!/usr/bin/env bash
# Source this helper; it never launches a stage or submits a job.
c0_set_profile() {
    c0_profile_args=()
    case "${1:-}" in
      '') ;;
      0.5B|1.5B) c0_profile_args=(--profile "$1") ;;
      *) printf '%s\n' "Unsupported model profile: $1 (expected 0.5B or 1.5B)" >&2; return 2 ;;
    esac
}

c0_activate_env() {
    # Optional SOL convenience: C0_ENV names a mamba environment (docs.rc.asu.edu
    # pattern: module load mamba/latest; source activate ENV). C0_PYTHON then
    # defaults to that environment's python. Exporting C0_PYTHON directly also works.
    if [[ -n "${C0_ENV:-}" ]]; then
        module load mamba/latest
        source activate "$C0_ENV"
        : "${C0_PYTHON:=$(command -v python)}"
        export C0_PYTHON
    fi
    # Compute nodes read pinned base models from the HF cache filled by prefetch_models.sh.
    export HF_HUB_OFFLINE="${C0_HF_OFFLINE:-1}"
    export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
}

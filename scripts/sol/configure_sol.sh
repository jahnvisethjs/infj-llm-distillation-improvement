#!/bin/bash
# Generate four SOL-local configurations only. Does not submit or run experiments.
set -euo pipefail
: "${C0_REPO:?Export the actual capstone repository directory on SOL}"
source "$C0_REPO/scripts/sol/common.sh"
c0_activate_env
: "${C0_PYTHON:?Export the compatible capstone Python executable, or C0_ENV for a mamba environment}"
: "${C0_DATA:?Export the V4 release directory containing metadata/release_manifest.json}"
: "${C0_PARENTS:?Export the parent package directory containing parent_index.json}"
: "${C0_INHERITED:?Export the inherited release directory containing metadata/ and code/}"
: "${C0_OUTPUT:?Export the results directory outside immutable input packages}"
cd -- "$C0_REPO"
[[ -x "$C0_PYTHON" ]] || { printf '%s\n' 'C0_PYTHON is not executable' >&2; exit 2; }
config_dir="${C0_CONFIG_DIR:-$C0_REPO/configs}"
for profile in 0.5B 1.5B; do
    for seed in 17 123; do
        "$C0_PYTHON" -B -m capstone_person4.v4_config \
          --release "$C0_DATA" --inherited-release "$C0_INHERITED" --parent-package "$C0_PARENTS" \
          --output-root "$C0_OUTPUT" --profile "$profile" --parent-seed "$seed" \
          --config-output "$config_dir/c0.local.sol.${profile,,}.seed${seed}.json"
    done
done

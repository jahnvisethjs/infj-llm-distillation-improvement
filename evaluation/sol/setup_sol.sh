#!/bin/bash
# One-time Sol setup (run in the Jupyter/OnDemand terminal, NOT as a job):
#   bash sol/setup_sol.sh <full 40-character Qwen2.5-1.5B revision from Person 2's run_config.json>
set -euo pipefail
REV15=${1:?usage: bash sol/setup_sol.sh FULL_1.5B_REVISION}
[ ${#REV15} -eq 40 ] || { echo "revision must be the full 40-character hash"; exit 1; }
REV05=060db6499f32faf8b98477b0a26969ef7d8b9987
cd "$(dirname "$0")/.."            # evaluation folder

echo "== 1. environment"
source sol/p3_env.sh
if ! conda env list | grep -q '^p3eval '; then mamba create -n p3eval python=3.11 -y; fi
source activate p3eval
pip install -q -r requirements.txt
python -c "import torch;print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

echo "== 2. tests + data"
python -m pytest -q test_answer_parser.py | tail -1        # expect: 59 passed
python check_eval_data.py | tail -1                         # expect: ALL CHECKS PASSED

echo "== 3. results folder, registry, pins"
R=/scratch/$USER/eval_results; mkdir -p "$R" logs
cat > "$R/pinned_revisions.json" <<J
{"Qwen/Qwen2.5-0.5B": "$REV05", "Qwen/Qwen2.5-1.5B": "$REV15"}
J
if [ ! -f "$R/runs.json" ]; then
cat > "$R/runs.json" <<'J'
[
  {"run_id": "base_a100", "model": "Qwen/Qwen2.5-0.5B", "adapter": null, "phase": "baseline", "where": "sol", "note": "A100 fp32, cleaned prompts, parser v5"},
  {"run_id": "base15",    "model": "Qwen/Qwen2.5-1.5B", "adapter": null, "phase": "baseline", "where": "sol", "note": "A100 fp32, cleaned prompts, parser v5"}
]
J
fi
cat "$R/pinned_revisions.json"; cat "$R/runs.json"

echo "== 4. download the pinned models (compute nodes are offline)"
export HF_HOME=/scratch/$USER/hf_cache
python - <<P
from huggingface_hub import snapshot_download
for m, r in {"Qwen/Qwen2.5-0.5B": "$REV05", "Qwen/Qwen2.5-1.5B": "$REV15"}.items():
    snapshot_download(m, revision=r); print("cached", m, r)
P
echo "== setup done. Next: nohup bash sol/run_interactive.sh > logs/run_$(date +%m%d_%H%M).log 2>&1 &"

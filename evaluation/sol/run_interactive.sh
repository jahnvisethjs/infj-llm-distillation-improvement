#!/bin/bash
# Run evaluations directly in an interactive GPU session (Jupyter with an A100), no sbatch.
#   bash sol/run_interactive.sh                 # every unfinished run in runs.json
#   bash sol/run_interactive.sh base15          # just one run
# Best started in the background so a closed browser tab does not stop it:
#   nohup bash sol/run_interactive.sh > logs/run_$(date +%m%d_%H%M).log 2>&1 &
#   tail -f logs/run_*.log
# If the session time runs out, start it again: finished runs are skipped, unfinished ones resume.
set -euo pipefail
cd "$(dirname "$0")/.."            # evaluation folder
source sol/p3_env.sh
mkdir -p logs

export HF_HUB_OFFLINE=1
ONLY=${1:-}

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)
echo "== GPU: $GPU"
[[ "$GPU" == *A100* ]] || { echo "STOP: not an A100. Request an A100 so every evaluation uses the same GPU type."; exit 1; }
echo "== code version: $(git log --oneline -1)"
python -m pytest -q test_answer_parser.py | tail -1          # expect 67 passed
python check_eval_data.py | tail -1                           # expect ALL CHECKS PASSED

# one-time check per model on this GPU: batch 16 vs 1 must give identical outputs (about 5 min each)
for M in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B; do
  TAG=a100_$(echo "$M" | sed 's/.*-//; s/\.//')
  if [ ! -f "$R/batch_check_$TAG.ok" ]; then
    REV=$(python -c "import json;print(json.load(open('$R/pinned_revisions.json'))['$M'])")
    python sol/batch_check.py --model "$M" --revision "$REV" --results "$R" --tag "$TAG"
    echo "$GPU" > "$R/batch_check_$TAG.ok"
  else
    echo "batch check $M already passed on: $(cat "$R/batch_check_$TAG.ok")"
  fi
done

SEL=(); [ -n "$ONLY" ] && SEL=(--only "$ONLY")
echo "== smoke test (10 problems per run)"
python run_eval.py --runs "$R/runs.json" --results "$R" --where sol --smoke "${SEL[@]}"
echo "== full evaluation"
python run_eval.py --runs "$R/runs.json" --results "$R" --where sol "${SEL[@]}"
echo "== done; summary at $R/summary.csv"
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
  echo "== backing up results to Drive ($DRIVE_RESULTS)"
  rclone copy "$R" "$DRIVE_RESULTS" --exclude "*.bak" && echo "backup done"
fi

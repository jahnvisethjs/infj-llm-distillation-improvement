# Shared settings for every Sol script and notebook cell:  source sol/p3_env.sh
# Where things live (all of it survives the end of a Jupyter session):
#   code + git repo   $HOME/infj-llm-distillation-improvement   (home: permanent)
#   results           /scratch/$USER/eval_results               (scratch: kept, but purged after long inactivity -> Drive backup)
#   adapters          /scratch/$USER/adapters
#   model cache       /scratch/$USER/hf_cache
_p3_flags=$-; set +u              # conda/module scripts use unset variables
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true
type module >/dev/null 2>&1 || for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /usr/share/lmod/lmod/init/bash; do
  [ -f "$f" ] && source "$f" && break; done
module load mamba/latest >/dev/null 2>&1 || echo "WARNING: could not load mamba module"
module load rclone >/dev/null 2>&1 || true        # Sol provides rclone as a module
# use the p3eval environment's python directly (works in notebooks, nohup and batch jobs alike)
P3ENV=""
for c in "$(conda env list 2>/dev/null | awk '$1=="p3eval"{print $NF}')" \
         "$(mamba env list 2>/dev/null | awk '$1=="p3eval"{print $NF}')" \
         "$HOME/.conda/envs/p3eval" "$HOME/.local/share/mamba/envs/p3eval" "$HOME/micromamba/envs/p3eval"; do
  if [ -n "$c" ] && [ -x "$c/bin/python" ]; then P3ENV=$c; break; fi
done
if [ -n "$P3ENV" ]; then
  export PATH="$P3ENV/bin:$PATH"; export CONDA_PREFIX="$P3ENV"
else
  echo "NOTE: p3eval environment not found yet (fine during first setup)"
fi
if [[ $_p3_flags == *u* ]]; then set -u; fi
export REPO=$HOME/infj-llm-distillation-improvement
export EVAL=$REPO/evaluation
export R=/scratch/$USER/eval_results
export ADAPTERS=/scratch/$USER/adapters
export HF_HOME=/scratch/$USER/hf_cache
export PATH=$HOME/bin:$PATH                      # rclone lives here if installed by hand
export DRIVE_RESULTS="gdrive:eval_results_sol"   # Drive copy of $R (separate from Colab's eval_results)
export DRIVE_ADAPTERS="gdrive:adapters"          # Drive folder with Person 2's run folders
mkdir -p "$R" "$ADAPTERS" "$HF_HOME"

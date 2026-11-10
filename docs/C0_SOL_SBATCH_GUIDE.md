# ASU SOL batch commands for C0/B

For C1 on the same 6,409-row bank, use the separate
[C1 sbatch guide](C1_SOL_SBATCH_GUIDE.md) after completing matching C0 runs.
Set `C0_B_BANK` before generating configs to keep Method B's common universe.

Updated 5 November 2026. The scripts target the class allocation supplied for this capstone:

```bash
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_fse57081354fall2026
#SBATCH --gpus=a100:1        # GPU stages only
```

`public` is the partition on which SOL accepts the `class` QoS ([partition/QoS table](https://docs.rc.asu.edu/partitions-and-qos/)); `general` does not accept it. Research Computing requests GPUs with `-G`/`--gpus` (for example `-G a100:1`, optionally `-C a100_80` for the 80 GB variant). Each stage is a separate job; no script runs accuracy evaluation, changes packages or deletes artifacts. All four model/parent configurations use the same scripts with different configuration files.

## Resource requests and concurrency

| Script in `scripts/sol/` | CPUs | Host RAM | A100s | Walltime cap |
| --- | ---: | ---: | ---: | --- |
| `prepare.sbatch` | 4 | 32 GB | 0 | 1 hour |
| `score_smoke.sbatch` | 4 | 32 GB | 1 | 30 minutes |
| `score_reference.sbatch` | 4 | 32 GB | 1 | 4 hours |
| `select.sbatch` | 4 | 32 GB | 0 | 30 minutes |
| `train_smoke.sbatch` | 4 | 32 GB | 1 | 30 minutes |
| `train.sbatch` | 4 | 32 GB | 1 | 4 hours per arm |
| `finalize.sbatch` | 4 | 32 GB | 0 | 15 minutes |

These are initial caps, not measured estimates (Person 2's R1000 0.5B pass took about 6.5 minutes on an A100). The `class` QoS limits are **2 running jobs**, **10 running plus pending jobs**, **24 hours per job** and **960 GPU running minutes per user** (sum of running jobs' time limits). Two 4-hour one-GPU jobs fit (480 minutes); two 8-hour jobs fit exactly; one 12-hour job leaves no room for a second. CPU jobs use running-job slots but no GPU minutes. One full eight-job chain stays under ten active submissions; do not queue two complete chains at once.

## 1. One-time setup on a login node

```bash
module load mamba/latest
mamba create -n c0 python=3.12 -y
source activate c0
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-person4.txt
```

`requirements-person4.txt` pins the stack recorded in the supplied parent package, including `huggingface-hub<1.0` (an unpinned 1.x hub broke tokenizer loading locally). Then export the capstone locations and generate the four SOL configurations:

```bash
export C0_REPO="/scratch/$USER/infj-llm-distillation-improvement"
export C0_ENV=c0                       # or: export C0_PYTHON=/path/to/env/bin/python
export C0_DATA="/scratch/$USER/data"   # contains metadata/release_manifest.json
export C0_PARENTS="/scratch/$USER/R1000_parents_seeds17_123_20261101T023416637853Z"
export C0_INHERITED="/scratch/$USER/person1_v3_20261012_ready"   # metadata/ and verified code/
export C0_OUTPUT="/scratch/$USER/c0-results"
export HF_HOME="/scratch/$USER/hf-cache"
cd "$C0_REPO"
bash scripts/sol/configure_sol.sh
bash scripts/sol/prefetch_models.sh configs/c0.local.sol.0.5b.seed17.json
```

`configure_sol.sh` writes `configs/c0.local.sol.<0.5b|1.5b>.seed<17|123>.json` (no scheduler commands). `prefetch_models.sh` downloads the pinned `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B` revisions into `HF_HOME` once; batch jobs run with `HF_HUB_OFFLINE=1` so compute nodes never depend on Hub access (set `C0_HF_OFFLINE=0` to allow it). Tokenizers are read from the checksummed parent package, never downloaded. Keep `C0_REPO`, `C0_ENV`/`C0_PYTHON` and `HF_HOME` exported in the shell that submits jobs: Slurm propagates the environment and `batch_stage.sh` activates the mamba environment inside each job.

Optional CPU preflight before any submission:

```bash
bash scripts/sol/preflight.sh configs/c0.local.sol.0.5b.seed17.json
```

## 2. Submit the whole chain for one configuration

```bash
CFG=configs/c0.local.sol.0.5b.seed17.json
bash scripts/sol/submit_pipeline.sh "$CFG" q05-s17 pilot01 smoke01
```

This submits seven jobs with `afterok` dependencies: prepare → score-smoke → score → select → train-smoke → C0 → finalize. B (random baseline) and A (replay) are trained by other team members, so B is not trained here by default; the select job still writes the matched-budget B manifest (`selection/B/manifest.jsonl`, same 318,845 ± 0.5% token target, same frozen pool) for whoever trains B, and the handoff records the budget B must match. Set `C0_ARMS="B C0"` to train B through this pipeline as well (eight jobs). A failed job stops its dependents; inspect its `<tag>-<stage>.<jobid>.err`, fix the cause and resubmit the remaining stages individually (section 3). Extra `sbatch` options for every job (for example `--mail-user=you@asu.edu --mail-type=END,FAIL`) go in `C0_SBATCH_ARGS`.

| Model / parent seed | `CFG` | tag |
| --- | --- | --- |
| 0.5B / 17 | `configs/c0.local.sol.0.5b.seed17.json` | `q05-s17` |
| 0.5B / 123 | `configs/c0.local.sol.0.5b.seed123.json` | `q05-s123` |
| 1.5B / 17 | `configs/c0.local.sol.1.5b.seed17.json` | `q15-s17` |
| 1.5B / 123 | `configs/c0.local.sol.1.5b.seed123.json` | `q15-s123` |

Run the 0.5B seed-17 chain first and inspect it before the other three; each combination gets its own measurements, smoke evidence, checkpoints and handoff.

## 3. Stage-by-stage submission

Each stage validates its own prerequisites and exits nonzero when inputs are missing or stale. From the repository root, with `C0_REPO` and `C0_ENV`/`C0_PYTHON` exported:

```bash
CFG=configs/c0.local.sol.0.5b.seed17.json
TAG=q05-s17
PREP=$(sbatch --parsable --job-name="$TAG-prep" scripts/sol/prepare.sbatch "$CFG")
SMOKE=$(sbatch --parsable --job-name="$TAG-score-smoke" --dependency=afterok:"$PREP" scripts/sol/score_smoke.sbatch "$CFG")
SCORE=$(sbatch --parsable --job-name="$TAG-score" --dependency=afterok:"$SMOKE" scripts/sol/score_reference.sbatch "$CFG")
SELECT=$(sbatch --parsable --job-name="$TAG-select" --dependency=afterok:"$SCORE" scripts/sol/select.sbatch "$CFG")
TRAIN_SMOKE=$(sbatch --parsable --job-name="$TAG-train-smoke" --dependency=afterok:"$SELECT" scripts/sol/train_smoke.sbatch "$CFG" smoke01)
C0_JOB=$(sbatch --parsable --job-name="$TAG-C0" --dependency=afterok:"$TRAIN_SMOKE" scripts/sol/train.sbatch "$CFG" C0 pilot01)
FINAL=$(sbatch --parsable --job-name="$TAG-finalize" --dependency=afterok:"$C0_JOB" scripts/sol/finalize.sbatch "$CFG" pilot01)
# Optional, only if B is trained here too: submit it after TRAIN_SMOKE and add it to FINAL's dependency.
# B_JOB=$(sbatch --parsable --job-name="$TAG-B" --dependency=afterok:"$TRAIN_SMOKE" scripts/sol/train.sbatch "$CFG" B pilot01)
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.10l %R"
```

What each stage does and what to inspect:

- **prepare** (CPU): file-only audit, protection scan of the 10,000-row selection bank, 5,000-row stratified freeze, re-encoding of every candidate and parent row with the package tokenizer. The replay anchor must reconcile to 318,845 supervised tokens. Output: `<C0_OUTPUT>/<profile>/<prepare-fingerprint>/prepared/preparation.json`.
- **score-smoke** (GPU): eight rows scored singly and in padded batches; finite per-token losses, causal counts, unchanged parent, measured throughput and peak memory in `reference/runtime/*.json`. The padded-batch gate compares each row's mean NLL (`scoring.atol` 0.05 nats, `rtol` 0.05); the per-token maximum is only recorded. On Sol single tokens differed by up to 0.44 nats from bfloat16 kernel noise while row means differed by under 0.02; masking bugs shift rows by whole nats and still fail.
- **score** (GPU): all 5,000 candidates, resumable per-row cache under `<C0_OUTPUT>/reference_cache/`. If no smoke proof exists for this exact GPU/runtime the job runs the smoke itself first, so a smoke that ran on a different A100 variant does not block scoring.
- **select** (CPU): `diagnose`, then B and C0 manifests under the shared allocator; each must total 317,251–320,439 supervised tokens. `selection/pair_budget_report.json` records the pairwise difference.
- **train-smoke** (GPU): disposable continuation on at most four C0 rows; verifies adapter activation, fresh optimizer, changed LoRA tensors and exact token accounting.
- **C0** (GPU, and B if requested): continuation from the original parent, seed 42, one pass, accumulation 8, learning rate 1e-4 with 10 warmup updates. Each arm writes `token_reconciliation.json` next to its checkpoint.
- **finalize** (CPU): reconciliation, checkpoint hash and parent identity of C0 (and B when it was trained here); writes `handoffs/<id>/evaluation_handoff.json` with `evaluation_status: pending` and, for an external B, the token target and tolerance it must match.

To validate the account/partition combination without submitting: `sbatch --test-only scripts/sol/score_smoke.sbatch "$CFG"`. Command-line options override script directives (for example `--time=08:00:00` or `-C a100_80`).

## Monitoring, longer caps and interrupted work

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.10l %R"
sacct -j "$B_JOB,$C0_JOB" --format=JobID,JobName,State,Elapsed,Timelimit,AllocTRES,MaxRSS
tail -f "$TAG-B.$B_JOB.out"
```

Logs land in the directory from which you submitted (`<job-name>.<jobid>.out/.err`). After measured throughput, raise a cap with a new attempt name:

```bash
sbatch --time=08:00:00 --job-name="$TAG-B" scripts/sol/train.sbatch "$CFG" B pilot02
```

An interrupted scoring pass reuses valid cached rows on resubmission. Interrupted training resumes only from that arm's own periodic checkpoint, into a new attempt:

```bash
sbatch --job-name="$TAG-C0-resume" scripts/sol/train.sbatch "$CFG" C0 pilot01_resume01 "" "$C0_OUTPUT/.../runs/C0/pilot01/checkpoint-000100"
```

If a B arm trained here ends with a different attempt name than C0, use the main CLI instead of the same-attempt batch helper (`--b-run` is optional):

```bash
"$C0_PYTHON" -B -m capstone_person4 finalize --config "$CFG" --c0-run <C0 run dir> [--b-run <B run dir>]
```

Timeouts or interrupted checkpoints never count as completed arms. Outputs are additive: an existing artifact with different content stops a stage instead of being overwritten, so use a new attempt name or output root when changing a frozen setting.

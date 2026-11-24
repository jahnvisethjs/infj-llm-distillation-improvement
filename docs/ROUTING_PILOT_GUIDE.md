# Small routing pilot before the full experiment

This is a separate exploratory screen, not the full frozen routing experiment.
The original opportunity analysis, question pool, adapters, native Python, and
interrupted full label outputs remain unchanged. No full experiment starts
automatically after the pilot. No new adapters are trained.

Default scope: parent 17, continuation seed 42, policy pool
`{A42,B42,C1_42}` versus A-only control `{A42,A137,A2026}`. These share five
unique checkpoints. The pilot takes approximately 300 original router-training
questions and 150 original router-validation questions, balanced over GSM/MATH.
It samples whole families by question-only hashes; counts can differ slightly
when a family boundary prevents an exact target. Sampling does not inspect
correctness or select checkpoints by observed performance.

The maximum is approximately **2,250 answers** instead of 43,200 training and
validation answers for the full experiment. Based on the observed SOL chunk
times, allow roughly 45–60 minutes of GPU generation plus setup, fitting, and
queue time. The batch allocation is capped at two hours. This is an estimate,
not a runtime guarantee.

## Deploy and prepare on the SOL login node

Build the new ZIP locally; keep previous releases:

```powershell
& 'C:/softwares/pythonAnaconda/python.exe' -B scripts/package_routing.py --output outputs/person4/routing_release/routing_pilot_v3_batchfix01.zip
scp "C:\Users\jahnv\OneDrive\Desktop\Jahnvi\ASU\fse capstone\infj-llm-distillation-improvement\outputs\person4\routing_release\routing_pilot_v3_batchfix01.zip" jseth2@sol.asu.edu:/scratch/jseth2/
```

If the long labeling job is still queued/running, identify and cancel only that
job, then wait until it has stopped. Do not use the completed smoke job's ID:

```bash
squeue -u "$USER"
# Replace LABEL_JOB_ID with the actual long labeling job ID:
scancel LABEL_JOB_ID
```

Use a new code directory. An existing directory is preserved by `mkdir` failing;
do not extract over an installed version:

```bash
export ROUTING_DIR=/scratch/jseth2/routing_20261123_203728
export ROUTING_PY=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python
export ROUTING_PILOT_CODE=/scratch/jseth2/routing_pilot_code_v3_batchfix01

mkdir "$ROUTING_PILOT_CODE" && unzip -q /scratch/jseth2/routing_pilot_v3_batchfix01.zip -d "$ROUTING_PILOT_CODE"
cd "$ROUTING_PILOT_CODE"
bash scripts/sol/routing_pilot_setup.sh
```

Setup creates `venv_pilot_v3` inside the original experiment directory with a
CPU-only PyTorch 2.6.0 build and the pinned router requirements. Dependencies
are installed in that separate environment; the historical evaluator
environment is not modified. This CPU build is for the exploratory pilot;
the full frozen live-answer experiment must retain its documented encoder
runtime compatibility requirements.
The fixed MiniLM encoder is prefetched on CPU into
`inputs/pilot_encoder_cache`, separate from the shared Qwen cache. No GPU job is
submitted during setup. The original exposure-screened `results/pool01` is
reused; no new dataset download or exposure scan is needed. Check the printed
counts and maximum answers before submission. The plan is sealed at
`results/pilot_plan01` before any pilot labels are inspected.

## Submit one detached job

```bash
bash scripts/sol/routing_pilot_submit.sh
```

After `Submitted batch job ...`, the terminal may be closed. The worker uses
the original evaluator Python for generation and the separate pilot Python for
CPU embeddings/classifiers. It reuses matching, sealed native chunks from
`results/labels_train01` and `results/labels_train02` if present. Unfinished
chunks are preserved and ignored. It verifies the complete original question
file, sidecar, checkpoint identity, runtime, and saved prediction contract
before taking a subset; it never rewrites an original sidecar to claim another
question-file hash. Conflicting overlapping saved answers are rejected rather
than selecting the better attempt. No final-test predictions can be imported.

The submitter exports `ROUTING_PILOT_CODE`; the worker imports the package from
that directory (falling back to the working directory set with `--chdir`). It
must not derive the code directory from `BASH_SOURCE`: Slurm executes a copied
script in its spool directory. The original v3 worker made that mistake and
failed immediately with `No module named capstone_routing` before generating
answers. A regression test executes a spool copy using a temporary deployed
package and covers explicit-directory and working-directory fallback behavior.
This batch-only fix leaves all package `.py` hashes unchanged; existing pilot
plans and prepared environments remain usable. Existing v3 installations can
submit a separately uploaded fixed worker with `sbatch --chdir=CODE --export=ALL`
and the original plan/output settings, without repeating setup. Preserve the
original release and use a different worker filename for that recovery.

Generated pilot chunks retain native batch size 16 and all frozen decoding
settings. Batch membership can differ from the original full labeling run.
Recorded answers are reused as recorded; this screen does not establish parity
over every possible batch composition or checkpoint.

The worker generates only missing pilot answers, then fits classifiers with
the same fixed, pinned MiniLM question embeddings as the full experiment.
It chooses C and switching threshold with three-fold family-grouped out-of-fold
predictions inside the 300 training questions. It refits on those 300 questions
and evaluates once on the reserved 150 validation questions. The fixed adapter
baseline is selected on training questions as well. Each policy/control pool
retains three candidates; both use the same questions, features, search grid,
and generation procedure. The original 600 final-test questions are never
generated, scored, or used for fitting. Original test question manifests may
be read by upstream family-integrity checks. Test gold is never parsed for
labeling, fitting, or scoring; receipt verification hashes all upstream files.

## Read the result before deciding on more compute

```bash
cat "$ROUTING_DIR/results/pilot_report01/report.md"
```

The report compares A, the router, the training-selected best fixed adapter,
random routing at the router's selection frequencies, and oracle accuracy.
It includes successful/harmful switches, rescues/risks, per-benchmark results,
question-level correctness, a policy-minus-seed contrast, and paired family
bootstrap intervals. Full construction metadata is preserved in `summary.json`.

`promising_pilot_signal` means at least the predeclared 2 percentage-point gain
over A, plus positive gains over the fixed and random baselines for at least
one pool. This practical screening flag is **not a significance test**, a
generalization claim, or an automatic go decision. The threshold can be set
before preparing a new plan with `prepare --minimum-gain-pp VALUE`.

If neither pool clears this screen, inspect the oracle, switch balance, and
intervals before committing to the long run. Small samples can miss useful
modest gains. If only the A-seed pool improves, that supports investigating
checkpoint diversity before attributing anything to policy diversity. A
positive result still needs confirmation across matched blocks and the
untouched full routing test after freezing. No variation across run blocks is
estimated from this one-block pilot. Bootstrap intervals condition on the
fitted models and checkpoints and exclude fitting uncertainty.

Files:

- `results/pilot_plan01`: fixed question/family manifests, selected checkpoints,
  lineage, and predeclared screening settings.
- `results/pilot_labels01`: native chunks, source references, correctness labels,
  and counts of reused versus newly generated answers.
- `results/pilot_report01`: `report.md`, `summary.json`, validation metrics,
  paired intervals, rescues/risks, correctness matrix, switches, training CV
  trials/folds, chosen models, and embeddings.
- `logs/routing-pilot-JOB_ID.log`: batch progress and terminal summary.

Check progress after reconnecting:

```bash
squeue -u "$USER"
tail -n 30 "$ROUTING_DIR/logs/routing-pilot-JOB_ID.log"
```

If interrupted, keep the existing output and request a new attempt, reusing
sealed pilot chunks in addition to the original full-run chunks:

```bash
export PILOT_REUSE="$ROUTING_DIR/results/pilot_labels01"
export PILOT_LABELS="$ROUTING_DIR/results/pilot_labels02"
export PILOT_REPORT="$ROUTING_DIR/results/pilot_report02"
bash scripts/sol/routing_pilot_submit.sh
```

Before a later full experiment, choose its configuration using development and
validation evidence only. Keep the existing final-test split untouched and
do not call pilot validation a final test.

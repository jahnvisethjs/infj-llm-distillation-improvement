# C1 on SOL: shared-bank continuation after C0

For the required Person 3 inputs/outputs, evaluator compatibility checks and
running candidate generation on Jahnvi's SOL account, see
[C1 generation requirements](C1_PERSON3_REQUIREMENTS.md).

The C1 scripts use the current repository's explicit **6,409-row Method B bank**.
They do not use the overnight notebook's 4,955-row screened bank. Each invocation
covers one model/parent and trains continuation seeds **42, 137, 2026** by default.
Run all four configurations separately to obtain **12 C1 runs**:

| Model | Parent seed | SOL config | Suggested tag |
| --- | --- | --- | --- |
| 0.5B | 17 | `configs/c0.local.sol.0.5b.seed17.json` | `q05-s17-c1` |
| 0.5B | 123 | `configs/c0.local.sol.0.5b.seed123.json` | `q05-s123-c1` |
| 1.5B | 17 | `configs/c0.local.sol.1.5b.seed17.json` | `q15-s17-c1` |
| 1.5B | 123 | `configs/c0.local.sol.1.5b.seed123.json` | `q15-s123-c1` |

## Prerequisites

Use the environment, input packages, model cache and exports from the
[C0 SOL guide](C0_SOL_SBATCH_GUIDE.md). When generating SOL configs, **export
`C0_B_BANK` to the actual `B_method_B_document_v3` directory first**:

```bash
export C0_B_BANK="/scratch/$USER/capstone_approach_b/outputs/B_method_B_document_v3"
bash scripts/sol/configure_sol.sh
```

Set that path to the real shared location when B belongs to another account.
Without `C0_B_BANK`, the generator produces the legacy 5,000-row setup, which
the C1 batch helper refuses. Export `C0_REPO`, `C0_PYTHON` or `C0_ENV`, and the
same `HF_HOME` used for C0 in the submission shell. Confirm access to the
scripts' account `class_fse57081354fall2026` on SOL.

Before C1 submission, finish the matching C0 runs for the same parent and all
requested continuation seeds. Preparation, reference scores, B/C0 selections,
and C0 token reconciliation/checkpoint files must remain available under the
same output root. Reuse the same config and scientific source files; changing
fingerprinted modules changes the artifact paths. These new batch wrappers do
not change any fingerprinted C0 module.

## Export generation requests

From the repository root on SOL, after C0 preparation:

```bash
CFG=configs/c0.local.sol.0.5b.seed17.json
sbatch --job-name=q05-s17-c1-request scripts/sol/c1_export.sbatch "$CFG"
```

The export job prints a `request_directory` containing `questions.jsonl` and
`request.json`. Give both to Person 3. Repeat once per model/parent: **four
requests**, each with 6,409 candidates. Generation is shared across the three
continuation seeds, not regenerated per seed.

Person 3 generates and grades with the original matching R1000 parent under
the frozen request contract (question-only inference, greedy fp32, 1,024 new
tokens, merged adapter, parser v6). Return a complete predictions JSONL and
the evaluator run's config JSON for each request. Use the actual emitted file
names; do not rename another report into an evaluator config. The ingest job
checks the parent, model revision, generation settings, parser and exact row
coverage. Execution errors must be retried externally, not counted as mistakes.

This pipeline schedules **no candidate generation GPU job**: it preserves the
repository's Person 3 handoff. It also schedules no benchmark evaluation or B
training. Candidate generations measure training-selection mistakes.

## Submit C1 after predictions arrive

```bash
CFG=configs/c0.local.sol.0.5b.seed17.json
PREDICTIONS=/actual/path/to/parent17_predictions.jsonl
REPORT=/actual/path/to/parent17_run_config.json
bash scripts/sol/submit_c1_pipeline.sh \
  "$CFG" q05-s17-c1 "$PREDICTIONS" "$REPORT" pilot01 c1pilot01 c1smoke01
```

The last three arguments are **C0 full attempt**, **C1 full attempt**, and
**C1 smoke attempt**. The C0 attempt must match the completed C0 runs (the C0
submitter defaults to `pilot01`). The default C1 names are `c1pilot01` and
`c1smoke01`; use new names for a fresh training attempt. Existing artifacts
are preserved. Use absolute prediction/report paths; the submitter resolves
them before submission.

Before the first scheduler call, a read-only preflight verifies prepared data,
reference scores, B selection, completed C0 evidence for every requested seed,
the matching exported request and existence of the supplied generation files.
The ingest job then performs full prediction validation. Failed ingest or
diagnostics blocks downstream jobs through `afterok` dependencies.

The default submission is seven jobs:

```text
ingest -> diagnostic gate + C1 selection -> C1 smoke
                                         -> train seed42   --+
                                         -> train seed137  --+-> finalize all three
                                         -> train seed2026 --+
```

The diagnostic gate blocks training when confirmed mistakes are below 5% or
above 95%, or C0/C1 selection Jaccard overlap is at least 95%. It retains the
diagnostic evidence. Passing gates bind the actual outcomes, C0/C1 selections,
reference measurement and checker/batch code. Smoke and full C1 jobs verify
those bindings again. No gate or budget is automatically relaxed.

Set `C1_SEEDS="42"` for a single continuation seed; by default all three are
submitted. A seed must have a matching completed C0 run. `C1_SBATCH_ARGS` adds
common scheduler options, falling back to `C0_SBATCH_ARGS`. Submit one parent
chain at a time and account for your other queued/running jobs. The submitter
limits its own chain to ten jobs but does not inspect current queue capacity.
If submission is interrupted or partially refused, inspect the printed job
IDs and `squeue` before resubmitting; duplicate submission is not suppressed.

## Resources, outputs and recovery

| Job | CPU cores / RAM | A100s | Walltime cap |
| --- | --- | --- | --- |
| Request export | 4 / 32 GB | 0 | 15 minutes |
| Ingest | 4 / 32 GB | 0 | 30 minutes |
| Diagnostics + selection | 4 / 32 GB | 0 | 30 minutes |
| C1 smoke | 4 / 32 GB | 1 | 30 minutes |
| C1 full training, per seed | 4 / 32 GB | 1 | 1 hour |
| Finalize, all requested seeds | 4 / 32 GB | 0 | 15 minutes |

All use `public`/`class` and the course account above. Caps are requests, not
measured runtimes. SOL resource/account availability still requires a live
preflight and GPU smoke. Stdout/stderr files are `%x.%j.out` / `%x.%j.err` in the
submission directory.

The helper reuses C1's existing fit-times-confirmed-mistake utilities, selector,
fresh-optimizer continuation trainer and token budget. Each run starts from
its original parent, never from a completed C0 or another C1 adapter. Only
generation outcomes and selections are shared across continuation seeds.

C1 data and gate evidence appear under the select-stage artifact root;
checkpoints appear under each seed's train-stage root at
`runs/C1/<C1-attempt>/`. Finalization requires reconciled **both C0 and C1** for
each seed; it cannot silently publish a C0-only handoff if C1 is missing.
The handoff records `evaluation_status: pending`.

For individual reruns, the wrappers accept the same config first:

```bash
sbatch scripts/sol/c1_select.sbatch "$CFG"
sbatch scripts/sol/c1_train_smoke.sbatch "$CFG" \
  --c0-attempt pilot01 --attempt c1smoke02 --training-seed 42
# Submit full runs only after the matching selection and smoke succeed.
sbatch scripts/sol/c1_train.sbatch "$CFG" \
  --c0-attempt pilot01 --attempt c1pilot02 --training-seed 42
sbatch scripts/sol/c1_finalize.sbatch "$CFG" \
  --c0-attempt pilot01 --attempt c1pilot02 --seeds 42
```

Use scheduler dependencies when queuing individual stages together. The C1
wrappers preserve failed/partial attempts and do not automatically resume them.
Request export and ingest may reuse identical immutable outputs. Any different
existing output requires resolving the changed inputs or a new output root.

## Local verification

```bash
python -B -m unittest discover -s tests/person4 -p 'test*c1*sbatch.py' -v
python -B -m unittest discover -s tests/person4 -p test_sbatch.py -v
python -B -m unittest discover -s capstone_person4_approachC1/tests -t . -v
```

Tests use recording scheduler/Python stubs and synthetic fixtures. They do
not submit to SOL, generate candidate answers or train real models.

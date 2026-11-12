# Complete the screened C0 pilot on Sol

Scope: Qwen2.5-0.5B, **parent seed 17**, **continuation seed 42**, one matched
B/C0 comparison. Seed 17 identifies the existing R1000 parent. Seed 42 preserves
the existing continuation and selection recipe; it is a different seed role.

Suraj authorized this pilot after the successful independent-screen preview.
`protocol_decision.json` records that scope. The original missing matcher audit
is still unresolved. This is a separately documented screening protocol.

## Start in the existing Sol Jupyter notebook

1. Upload `C0_Screened_05B_Seed17_20261110.zip` into
   `/scratch/sshind38/capstone_c0_c1_work/`.
2. Run the code in `LAUNCH_CELL.py` as one new notebook cell. The same code is in
   the first code cell of `C0_05B_Seed17_Run.ipynb`.
3. The cell prints the Slurm job ID and the exact `.out` / `.err` log paths.
   Share that output if submission fails. The job can continue after the notebook
   disconnects.
4. Use the notebook's status cell to inspect progress. Submit success is not a
   completed experiment; only the final completion report establishes completion.

The launch cell checks the archive's files, extracts without replacing different
existing files, verifies the already produced preview, freezes its exact bank and
selections in a separate output directory, and submits one A100 job. No package
installation, GPU rescoring or new constrained solve is required.

## What the job runs

The stages run sequentially, in fresh Python processes so GPU models are released
between stages:

1. B training smoke: 32 selected examples, four updates, save/reload check.
2. C0 training smoke: the same checks under its own selection.
3. Full B continuation: 1,000 examples, 317,486 supervised tokens, 125 updates.
4. Full C0 continuation: 1,000 examples, 320,273 supervised tokens, 125 updates.
5. Untouched parent evaluation on 500 development GSM and 500 development MATH questions.
6. The same two evaluations for the matched B and C0 checkpoints.
7. Training reconciliation, paired comparisons, and a results ZIP with per-question
   predictions, settings, selected manifests, logs, lineage and checkpoint locations.

The B/Person2 training function is reused with only its reporting label changed.
The loop retains QLoRA, explicit next-token loss, completion-plus-EOS labels,
token-weighted accumulation, one epoch, microbatch one, accumulation eight,
learning rate 1e-4, ten warmup updates, clipping, original parent weights and a
fresh optimizer/scheduler. Original B files and checkpoints remain immutable.

Both arms preserve 50 examples in each of 20 source/length strata, ±5% stratum
token bounds and ±0.5% of the 318,845-token global target. The actual selected
row content, order and tokenization must match the verified preview and inherited
actual-label proof immediately before training.

The job requests the previously used class account, public partition and class
QoS, one A100, four CPUs, 32 GB RAM, and a **four-hour allocation cap**. Four hours
is a requested cap, not a runtime prediction. Sol establishes scheduler availability.

## Outputs and checking completion

`prepare` prints the actual output directory, also saved in `RUN_LOCATION.json`:

```text
/scratch/sshind38/capstone_c0_c1_work/outputs/C0_screened_pilot_v1/0.5B_seed17/<new-identity>/
```

The identity binds the new runner, declared decision, original bridge/runtime,
screening evidence, bank, cached measurements, tokenizer proof and selections.
The source preview remains under its original `35f8a5ecbf57cd37` directory.

Important output locations relative to the new directory:

| File or directory | Meaning |
| --- | --- |
| `freeze.json` | New protocol, exact evidence hashes and disclosed limitations |
| `selection/B/train.jsonl` | The newly screened B control |
| `selection/C0/train.jsonl` | The screened reference-fit selection |
| `runs/B/seed42_pilot01/final/adapter/` | Full B adapter |
| `runs/C0/seed42_pilot01/final/adapter/` | Full C0 adapter |
| `development/pilot01/{parent,B,C0}/` | Six prediction files and their native evaluator settings |
| `reports/pilot01/completion_report.json` | Reconciled training, results and comparison evidence |
| `reports/pilot01/comparison.md` | Readable parent/B/C0 accuracy table |
| `PILOT_COMPLETE_pilot01.json` | Written only after both training runs and all evaluations validate |
| `C0_05B_parent17_cont42_pilot01_results.zip` | Upload this small results bundle to this conversation |

The results ZIP excludes model weights and optimizer states. Their verified hashes
and exact checkpoint paths remain in the report; adapters remain on Sol.

## If the job is interrupted

Rerunning the launch cell first checks whether the previous job is still pending
or running; it does not submit a duplicate active job. Once the job ends, it can
submit another allocation for the same attempt. Completed training stages are
validated and reused. Partial development predictions resume under the native
evaluator's unchanged checkpoint/settings/platform checks.

A partially trained run without a valid completion receipt is preserved. The
runner stops instead of overwriting it or guessing optimizer state. If training
was interrupted, change `ATTEMPT = 'pilot01'` to `ATTEMPT = 'pilot02'` in the launch
cell. This restarts both methods from the original parent with the same controls
and writes separate outputs. Do not delete earlier attempt files.

## Scientific scope

- This screen retains 4,955 of 6,409 candidates (3,313 Orca, 1,642 Numina).
- The 1,454 quarantines are operational exclusions, not 1,454 proven contaminated
  or invalid examples. Conservative cosine rules remove some distinct problems
  and can miss below-threshold semantic repetitions.
- The previous 25-exclusion/572-quarantine audit was not reproduced or resolved.
  No `status: resolved` report for the old protocol is created.
- Seven inherited parent/development relationships remain disclosed. Results are
  exposure-limited development measurements; this screening does not repair the
  trained parent or establish uncontaminated benchmark generalization.
- Corpus-wide mathematical correctness and unprovided reserve populations are
  not certified. See the included original screen documentation for scope.
- Old B results are historical context. The new B run supplies the matched control.
- Paired intervals describe item variation for these saved checkpoints, not
  training-seed variability. Accuracy improvement is not assumed.
- Reserved final GSM8K/MATH500 evaluation, C1, other parents and other sizes are
  separate milestones. `screened_c0_pilot_complete` applies only to this pilot.

## Validation performed when packaging

See `LOCAL_VALIDATION.json` and `local_validation.log`. CPU checks cover quarantine
application on the supplied question projection and targeted references, quotas,
global/stratum budgets, changed content and order, consumption and warmup, parent
optimizer isolation, smoke/full separation, mixed evaluation identities, checker
errors, paired comparisons, and launch behavior when Slurm copies the batch script.

The full 6,409-row Sol manifest, cached GPU measurement files, actual tokenizer,
real Qwen loading/training/evaluation and the Sol scheduler are **not available in
this execution environment**. Their checks execute on Sol; local validation is
not a real-model completion receipt.

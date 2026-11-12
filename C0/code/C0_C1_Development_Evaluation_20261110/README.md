# C0/C1 development evaluation on Sol

Deliverable: one self-contained `C0_C1_Development_Evaluation_Overnight.ipynb`.
Upload it to Sol, open it in Jupyter, and run configuration, embedded install and
launch (sections 1–3). A CPU Jupyter session of one hour is sufficient for launch.
The existing B evaluation environment, model caches, trained adapters and original
training evidence must remain at their existing paths.

## Evaluation scope

- C0: both development benchmarks for all 12 saved adapters.
- C1: both development benchmarks for all 12 saved adapters.
- Each benchmark has exactly 500 frozen questions: `dev_gsm` and `dev_math`.
- Compatible C0 pilot predictions are reused; normally this leaves 23 C0/C1
  adapters and 46 benchmark files requiring new generation.
- The four matching original parents and screened B pilot are included for
  comparisons. Their compatible existing predictions are reused where available;
  missing compatible helper evaluations are generated under the same contract.
- Original B predictions are imported when their complete per-question outputs
  and settings are compatible. They remain historical context for the original
  bank, not matched controls for the new screening protocol.
- No training, candidate rescoring, new selection or reserved final-test evaluation.

There are 58 required prediction files: 48 C0/C1, eight parent, and two screened
B-pilot files. Reused files count toward completion. Up to 24 original B files
are imported separately and do not require new GPU generation.

## Frozen contract and identity checks

The exact Person 3 evaluator and parser from the completed pilot are reused:
fp32, greedy, batch 16, 1,024 new tokens, merged adapters, `p1_clean`, EOS 151643,
parser `p3-parser-v6`, math-verify 0.9.0, PyTorch 2.6.0+cu124, transformers 4.57.1,
peft 0.17.1, CUDA 12.4, and NVIDIA A100-SXM4-80GB. No package upgrades occur.

Checkpoint weights, saved model/revision/seed/manifest, LoRA scaling/targeting and
the approved training metadata are verified. The actual saved parent can retain
its historical `preliminary` label: exact weight identity and training metadata
determine its identity. Smoke/debug adapters are refused.

Prediction reuse requires all 500 unique frozen question IDs, the correct
checkpoint/model/manifest, exact configuration hash, gold answers, parser/source
hashes, software and GPU platform. Partial files cannot be treated as full
results. Missing/unparseable answers count incorrect and are reported separately.
Checker execution errors are retried on saved text using the exact same parser,
with the original outputs and retry evidence preserved. Unresolved checker
errors block that result; they are never silently counted as mathematical errors.

## Disconnection and recovery

The launch cell validates sources and caches before submitting. Six four-hour
Slurm allocations run serially, one A100 at a time. Once job IDs are printed,
the notebook/laptop can disconnect. The 24-hour total allocation cap is not a
runtime prediction. Actual runtime depends on existing result reuse, generation
speed, queueing and recovery; the earlier 10–16 GPU-hour estimate is approximate.

Evaluation saves native per-question predictions as it goes. Compatible partial
files resume. A truncated JSON record is preserved and a new bounded attempt is
used. Complete native files can be published after a disconnect without loading
the model again. Results are published atomically after validation.

The controller hands unfinished work to the next pre-submitted allocation and
reuses hash-checked completed files. Attempts are bounded to three. It protects
against duplicate active submissions, recovers held jobs after a launch-cell
disconnect, and checks scheduler accounting when old jobs leave the queue.
Unused later allocations are cancelled after terminal success or failure.

## Progress and collection

Section 4 prints:

- Slurm jobs and states;
- C0 and C1 models evaluated out of 12 (both benchmarks required per model);
- completed benchmark files out of 58;
- the current model/benchmark and live `16/500`, `32/500`, … progress from its log;
- log modification time and any blocked/failed stages.

These are evaluation counters, distinct from the already completed training.
Section 5 collects the final handoff ZIP into `/home/sshind38/`; it prints the
exact filename and can be downloaded through Sol Files → Change directory →
`/home/sshind38/`. Upload that ZIP back to this conversation for independent
verification and interpretation. The ZIP contains predictions, configs, source
identities, metrics and comparison tables, with no model weights.

If the chain is exhausted, rerun the launch cell for a new bounded chain that
reuses validated results. For a blocked integrity/checker stage, inspect its log;
do not edit hashes, parser rules, question sets or runtime settings to force
completion. A Slurm `COMPLETED` state alone is insufficient: look for
`EVALUATION_COMPLETE.json` and `status: evaluation_complete`.

## Automatic comparisons

- All 12 C1-minus-C0 comparisons on each benchmark, paired on identical questions.
- All C0/C1-versus-parent comparisons.
- C0/C1-versus-screened-B contrasts for 0.5B / parent17 / continuation42.
- Accuracy, valid-answer rate, truncation, repetition and output length separately
  for GSM and MATH.
- Per-seed equal-weight benchmark averages; three-seed means and sample SD.
- Paired item-bootstrap intervals: 2,000 resamples, seed zero, for fixed saved
  checkpoints. They are not training-seed uncertainty intervals. No best-seed
  selection or automatic formal significance claim.

Files include `DEVELOPMENT_RESULTS.csv`, `PAIRED_COMPARISONS.csv`,
`PER_SEED_AGGREGATE.csv`, `SEED_SUMMARY.csv`, `HISTORICAL_B_CONTEXT.csv`,
`MISSING_MATCHED_B.csv`, `comparison.md` and the complete JSON report.

Eleven additional screened B training/evaluation controls are still missing for
the full matched B/C0/C1 matrix. This notebook evaluates existing models only.
The original B bank differs; its scores are labeled historical and excluded from
matched C-minus-B effects. A completed C0/C1 evaluation marker does not claim that
the full three-method matched matrix or the final study is complete.

Inherited parent/development exposure and conservative-screen limitations remain
disclosed. These are development measurements; no contamination-free
generalization, final-test result or accuracy improvement is assumed.

## Local validation

CPU tests cover prediction identity/coverage/gold/configuration checks, historical
parent labels, LoRA scaling, parser failures, native partial/full recovery,
atomic publication, paired statistics, historical-bank exclusion, interruption
across Slurm allocations, blocked completion and duplicate/held-job recovery.
Full orchestration uses simulated GPU compute. All six existing real pilot
prediction files were additionally checked for compatibility. No new Qwen GPU
evaluation or live Sol scheduler execution was performed here; those checks run
on Sol when the notebook is launched.

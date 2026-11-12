# Screened B training and development evaluation

Upload the updated `B_Screened_4955_Training_Overnight.ipynb` into your Sol home
folder. Open this updated copy and choose **Run > Run All Cells**. It contains
both packages and performs CPU preflight before submitting GPU jobs.

The original B training package is embedded byte for byte unchanged. Existing
completed or active B training runs are reused, and the original B pilot is
preserved. Do not cancel or resubmit an older training job manually just to use
this notebook. The submission code checks active recorded jobs before starting
another training chain.

## Fully unattended sequence

1. Complete the eleven remaining screened B training configurations, reusing
   the completed 0.5B / parent 17 / continuation 42 pilot. Every full run starts
   from its matching original parent, uses a fresh optimizer, and trains the
   fixed 1,000-row B selection from the same 4,955-row bank used by C0/C1.
2. Wait for all active recorded training jobs to end. Evaluation starts only
   when all twelve screened B checkpoints pass the training completion checks.
3. Evaluate each new B checkpoint on the exact **500 GSM + 500 MATH development
   questions** used by C0/C1. Reuse the pilot's two compatible completed files.
4. Produce twenty-four validated development files for twelve B configurations,
   a twelve-row B summary CSV, raw predictions, receipts, and a combined
   training/evaluation handoff ZIP. There are **22 new GPU evaluation files**
   (11,000 new question predictions) and **2 reused pilot files**.

No reserved final-test questions are used. No new C0 or C1 training/evaluation
is scheduled: those development evaluations were already completed.

Wait until the notebook submission cell returns real training/evaluation job
IDs and `all_jobs_released: true`. Afterward your laptop, notebook, browser and
SSH connection may disconnect. Slurm continues the jobs without the notebook.
The default is two serial four-hour training allocations followed by three
serial four-hour evaluation allocations, one A100 each. These are upper limits,
not a runtime estimate or overnight guarantee. Queue time is additional.
Unused future queued allocations belonging to this submission are cancelled
once work is complete or no runnable tasks remain.

## Same evaluation settings

The unchanged native evaluator and answer parser are fingerprinted against the
working B/C0/C1 runtime. The frozen contract uses greedy fp32 generation,
batch size 16, at most 1,024 generated tokens, merged adapters, the same prompt,
the same model revisions, and parser `p3-parser-v6` with math-verify 0.9.0.
Question-file hashes, checkpoint hashes, training-manifest hashes, saved LoRA
configuration and per-prediction identities are verified. Missing or duplicate
questions, final-test rows, changed settings and unresolved checker errors block
completion. Runtime/GPU drift is reported; settings are not silently relaxed.

## Progress and recovery

The Status cell shows both training and evaluation Slurm IDs, independent model
counts, and the latest active-stage output. Dependency-pending evaluation jobs
are expected during training. Completion requires the actual receipts and
checkpoints, not merely Slurm COMPLETED states. Evaluation ends with
`B_screened_models_evaluated_of_12: 12`, `completed_evaluation_files: 24`,
and a validated `EVALUATION_COMPLETE.json`.

Training partials restart from the original parent in a new preserved attempt.
Evaluation resumes compatible partial predictions by question ID. Malformed
or incompatible attempts are preserved; attempts are bounded. If a chain ends
incomplete, rerun submission to reuse completed work and continue. If sbatch's
outcome is uncertain, inspect the saved ledger and queue instead of blindly
submitting again. Do not change scientific inputs to bypass an integrity error.

## Exact output locations

Training remains under `outputs/B_screened_4955_controls_v1/<identity>/`.
Evaluation uses `outputs/B_screened_4955_development_evaluation_v1/<identity>/`.
Both are inside `/scratch/sshind38/capstone_c0_c1_work`. Notebook output prints
the exact directories and logs. Code is added in the new evaluation package;
original B, C0/C1 and old evaluation files are read without changing them.

After completion, rerun the Download cell to copy the combined handoff ZIP into
`/home/sshind38` and print its exact path. It includes prediction files, configs,
metrics, the B summary CSV, checkpoint index, and training receipts. Weights stay
on Sol. Use these screened B results alongside screened C0/C1 for the fair
candidate-pool comparison; legacy B remains separate. A new complete B/C0/C1
comparative analysis is a subsequent step, not fabricated in advance here.

The independent conservative screen does not certify absence of contamination.
The historical 25-exclusion/572-quarantine audit remains unresolved in reviewed
copies, inherited parent/development exposure remains disclosed, and C1 remains
the existing exploratory refinement with its unchanged diagnostic gate.

Local validation used real archived pilot predictions and frozen development
questions, unchanged native-function comparisons, partial-file recovery tests,
mocked Slurm dependency/duplicate checks, and synthetic complete/blocked worker
tests. These local checks do not constitute a real Sol GPU submission.

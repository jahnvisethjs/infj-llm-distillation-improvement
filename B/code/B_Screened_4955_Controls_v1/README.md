# Screened B: eleven remaining controls

Upload **B_Screened_4955_Training_Overnight.ipynb** to `/home/sshind38` on Sol,
open it in Jupyter, and use **Run > Run All Cells**. The notebook contains this
entire package. A CPU-only preflight runs before any GPU submission.

Wait for the submission cell to return **real job IDs** and
`all_jobs_released: true`. Then your laptop, Jupyter session and SSH connection
can disconnect. The jobs run under Slurm. Two serial four-hour A100 allocations
are submitted by default; these are limits, not a prediction of runtime or a
guarantee that all training finishes overnight. Queue time is additional.
The second allocation continues unfinished runs; unused future queued jobs are
cancelled after all requested work is finished. No browser/kernel needs to run
the controller after submission.

## What is trained

* Eleven new B controls, bringing screened B to twelve configurations when the
  existing 0.5B / parent 17 / continuation 42 pilot is included.
* Sizes 0.5B and 1.5B; parents 17 and 123; continuation seeds 42, 137 and 2026.
* **1,000 selected examples per model**, from the same **4,955-row eligible
  bank** used by the completed C0/C1 runs. Do not train all 4,955 examples.
* Exact completed screened B pilot selection, reused across all configurations:
  317,486 supervised tokens versus the common 318,845 target, within 0.5%.
  Twenty source/length strata, fifty rows each, with the same stratum budgets.
* Original matching parent adapters, fresh optimizer/scheduler, one epoch,
  learning rate 1e-4, ten warmup updates, microbatch one, accumulation eight.
  Native B loss, quantization, update, saving and reload-probe behavior retained.
* A separate 32-row smoke check precedes each full run and starts from the
  original parent. Smoke weights never initialize full training.

**Training only.** No candidate scoring/generation, development evaluation or
reserved final testing is submitted. Evaluate the eleven new B models on the
same frozen 500 GSM + 500 MATH development questions afterward. The existing
screened B pilot already has development evaluation evidence. Preserve legacy B
as a separate condition; compare the screened controls against screened C0/C1.

## Required existing Sol files (validated without changing them)

* `/scratch/sshind38/capstone_approach_b`: original B runtime, environments,
  parents and offline pinned base models.
* `/scratch/sshind38/capstone_c0_c1_work/C0_C1_Overnight_Training_20261110`:
  exact previously used runner, bundled checks and provenance pins.
* Its four completed profile tokenization proofs under
  `outputs/C0_C1_overnight_training_v1/7376bbd91d80e726/`.
* The 4,955-row bank at
  `outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17/35f8a5ecbf57cd37/candidate_bank.jsonl`.
* Original screened pilot package and B checkpoint at
  `outputs/C0_screened_pilot_v1/0.5B_seed17/be46c67690fb33ba/`.

Any drift in code, scientific inputs, parents, bank or selected manifest stops
the run rather than relaxing eligibility or substituting another bank. The
independent screen does not resolve the historical 25-exclusion/572-quarantine
audit, and inherited parent/development exposure remains disclosed. C1's
exploratory rule and diagnostic gate are unchanged.

## Progress, recovery and outputs

New code is installed only in `B_Screened_4955_Controls_v1` under the C0/C1 work
root. New results live in `outputs/B_screened_4955_controls_v1/<identity>/`.
The submission prints the exact output directory, Slurm log paths and job IDs.
Re-run only the notebook's **Status** cell to check progress; submission reruns
also refuse to duplicate active jobs. A scheduler state of COMPLETED alone is
insufficient. Success requires `TRAINING_COMPLETE.json`, eleven validated new
training receipts and twelve checkpoint entries including the pilot.

Interrupted training restarts in a new attempt from the original parent with a
fresh optimizer. Partial attempts remain intact; no optimizer state is guessed
or mixed between runs. There are at most three attempts per run. Failed data or
recipe integrity checks block that configuration and are reported explicitly.

If the allocation chain ends before all runs finish, the notebook will report
incompleteness. Re-run the submission cell to submit a new chain and reuse valid
completed runs. If sbatch's outcome is uncertain, the saved submission ledger
blocks blind resubmission: inspect the displayed ledger and your queue first.
Do not launch a second copy with modified scientific inputs to bypass checks.

After verified completion, rerun the final notebook cell. It copies a lightweight
training handoff ZIP into `/home/sshind38` and prints the exact path. It contains
receipts, configs, full training-consumption logs, selection metadata and an
evaluation-ready checkpoint index. Adapter weights stay on Sol. The next step
is development evaluation, followed by a matched-pool B/C0/C1 comparison.

Local CPU tests exercised real archived pilot data, manifest hashes, consumption
and optimizer receipts, unchanged native training functions, scheduler chaining,
held-job release, duplicate suppression and preservation of interrupted runs.
No Sol job or GPU training was submitted while generating these files.

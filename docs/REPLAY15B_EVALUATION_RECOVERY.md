# Recover the existing 1.5B parent17 evaluation

This is an evaluation-only recovery for the completed parent17 training identity
`5a0577ec24d78630b0594107aa6d093b599ad4900bea85047d181df4a027c513`.
The original evaluation job `2244344` stopped when the old result validator rejected
the legitimate parser status `empty_marker`. Finalization job `2244345` was then
cancelled. All three trained parent17 adapters are reused.

The original 0.5B and 1.5B packages and scripts remain byte-for-byte unchanged.
This separate tool checks the exact v1 source hashes before loading the original
configuration, all control/parent checks, and all three training completion
receipts. It refuses parent123: that parent's completed report stays in place.

## Transfer from Windows

Run in PowerShell:

```powershell
scp "C:\Users\jahnv\OneDrive\Desktop\Jahnvi\ASU\fse capstone\infj-llm-distillation-improvement\outputs\person4\replay_c1_release\replay15b_eval_recovery_v1.zip" jseth2@sol.asu.edu:/scratch/jseth2/
```

Connect the ASU VPN first if required for SSH access.

## Extract separately on Sol

```bash
mkdir /scratch/jseth2/replay15b_eval_recovery_v1 && \
  unzip /scratch/jseth2/replay15b_eval_recovery_v1.zip \
    -d /scratch/jseth2/replay15b_eval_recovery_v1

cd /scratch/jseth2/replay15b_eval_recovery_v1
export REPLAY15B_PYTHON=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python
export REPLAY15B_ORIGINAL=/scratch/jseth2/replay_c1_15b_code_v1
export REPLAY15B_CONFIG=/scratch/jseth2/replay15b_parent17_config.json
```

The ZIP contains the recovery tool and its worker, documentation, tests and
checksums. It does not replace the original training package. Keep the original
code directory and parent17 configuration at their current locations.

## Inspect first; no GPU work or writes

```bash
"$REPLAY15B_PYTHON" -B -m capstone_replay15b_recovery inspect \
  --repo "$REPLAY15B_ORIGINAL" --config "$REPLAY15B_CONFIG" --summary-only
```

Expect `"ready":true`. The output reports `reuse_benchmark_files` and
`generate_benchmark_files`, totalling 18: three methods, three continuation seeds
and two frozen 500-question benchmarks. Omit `--summary-only` for the per-file
inventory and parser-status counts. A blocked record names the affected method,
seed, benchmark and failed check. Inspect performs no GPU execution and submits
no jobs.

The recovery validates canonical `eval01` receipts first. Their hashes and full
runtime/question/checkpoint contracts must pass. It also checks complete raw
outputs in `<method>_seed<seed>/<benchmark>/native/raw.jsonl` with their sidecars.
Unfinished raw files are preserved and evaluated afresh. Complete C1 results are
reused from the already verified native/published records.

## Submit one evaluation/reporting job

After inspection returns ready, run:

```bash
"$REPLAY15B_PYTHON" -B -m capstone_replay15b_recovery submit \
  --repo "$REPLAY15B_ORIGINAL" --config "$REPLAY15B_CONFIG" --authorize-gpu

squeue -u jseth2 -o "%.18i %.32j %.2t %.10M %R"
```

This submits one A100 job named `recover15b-p17-eval`. It reuses validated outputs,
generates only the missing/unusable benchmark files and writes all three-seed
comparison reports. The job uses the original frozen evaluator, parser, decoding,
question hashes and runtime. It never invokes training. Submission is explicit;
packaging/local tests do not authorize or execute this GPU work.

If inspection says `generate_benchmark_files` is zero, the same recovery can
instead run entirely on the CPU without submitting a job:

```bash
"$REPLAY15B_PYTHON" -B -m capstone_replay15b_recovery run \
  --repo "$REPLAY15B_ORIGINAL" --config "$REPLAY15B_CONFIG"
```

## How statuses are handled

The exact frozen parser has SHA256
`3361e3c81864072961026fca4d7459a17c7544ed5ffec144575ab20eeac93d5d`.
The regression suite exercises every status it emits, including both `empty` and
`empty_marker`. These two statuses are recognized as no-answer outcomes only in
an ephemeral validation view. Saved raw and canonical rows keep their original
status, output text and `correct=False`. The frozen parser and accuracy policy do
not change; all 500 questions remain in each benchmark's accuracy denominator.

`verify_error`, `error` and `execution_error` are genuine checker failures. They
are never accepted as wrong answers. If an unreceipted raw benchmark contains
one, recovery generates one clean full-benchmark evaluation under the same
checkpoint/runtime. A repeated checker error stops recovery and records failure
evidence. Unknown statuses, inconsistent correctness, corrupt canonical receipts,
duplicate/unknown question IDs and mismatched checkpoint/runtime contracts stop
recovery instead of being silently replaced.

## Read the recovered results

The new attempt is `eval02_recovery`. After the returned recovery job ID shows
`COMPLETED` with exit code `0:0`, run:

```bash
cat /scratch/jseth2/replay_c1_15b_4955_pilot/1.5B_parent17/5a0577ec24d78630/reports/eval02_recovery/accuracy.csv
```

Its directory also contains `paired_differences.csv`, `run_index.csv`,
`summary.json`, original-training `provenance.json`, and a separate `recovery.json`
with the recovery code/policy hashes and per-file origin ledger. Completion is
recorded after validation in:

```text
/scratch/jseth2/replay_c1_15b_4955_pilot/1.5B_parent17/5a0577ec24d78630/recovery/eval02_recovery/completion.json
```

Submission details and `recover.<jobid>.out/.err` logs are in that same recovery
directory. Original `evaluation/eval01`, training checkpoints, and the parent123
report are preserved. Parent123 results remain at:

```text
/scratch/jseth2/replay_c1_15b_4955_pilot/1.5B_parent123/dd521be5e744f065/reports/eval01/accuracy.csv
```

## Interrupted recovery

The submission ledger prevents duplicate submission of the same recovery attempt.
An execution claim preserves interrupted work. Read the job's logs and
`failure.json` before choosing a new recovery attempt. A later clean recovery can
reuse completed files from the previous recovery by explicitly setting:

```bash
"$REPLAY15B_PYTHON" -B -m capstone_replay15b_recovery inspect \
  --repo "$REPLAY15B_ORIGINAL" --config "$REPLAY15B_CONFIG" \
  --source-evaluation-attempt eval02_recovery \
  --evaluation-attempt eval03_recovery --summary-only
```

Use those same two attempt flags with `submit --authorize-gpu` only after the
previous recovery job has ended and the new inspection passes. No files are
deleted, overwritten or resumed by appending to an interrupted JSONL.

## Local checks

From the full development repository:

```bash
python -B -m unittest discover -s capstone_replay15b_recovery/tests -v
bash -n scripts/sol/recover_replay15b_worker.sh
```

Tests use explicit synthetic weights and native-shaped files; model generation
is mocked. Test fixtures contain the exact hash-pinned parser/evaluator sources
for contract checks. Local success does not establish live Sol file access or
finish the real evaluation; run inspect and then the recovery on Sol.

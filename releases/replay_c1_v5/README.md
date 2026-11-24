# Existing replay+C1 launcher: 0.5B parent123 profile

Release v5 extends the same `capstone_person4_approachReplayC1` command to accept
parent seeds 17 and 123. Use the commands below for **parent123 only**, with
continuation seeds **42, 137 and 2026**. The original parent17, 1.5B and recovery
deployments remain unchanged. This release is a versioned copy of the existing
launcher; no new training algorithm is introduced.

## Transfer from Windows PowerShell

```powershell
scp "C:\Users\jahnv\OneDrive\Desktop\Jahnvi\ASU\fse capstone\infj-llm-distillation-improvement\outputs\person4\replay_c1_release\replay_c1_pilot_v5.zip" jseth2@sol.asu.edu:/scratch/jseth2/
```

## Extract and configure on Sol

```bash
mkdir /scratch/jseth2/replay_c1_code_v5 && \
  unzip /scratch/jseth2/replay_c1_pilot_v5.zip -d /scratch/jseth2/replay_c1_code_v5

export REPLAY_C1_REPO=/scratch/jseth2/replay_c1_code_v5
export REPLAY_C1_PYTHON=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python
cd "$REPLAY_C1_REPO"

"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 configure \
  --parent-seed 123 \
  --config-output /scratch/jseth2/replay_c1_05b_parent123_config.json \
  --output-root /scratch/jseth2/replay_c1_4955_pilot

"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 preflight \
  --config /scratch/jseth2/replay_c1_05b_parent123_config.json
```

Configuration/preflight submit no jobs and execute no GPU models. Preflight must
return `"passed":true` before submission. It verifies all three A and all three
C1 controls, the screened 4955-row bank, original 1000-row parent exposure,
model-specific tokenization proof, cached reference scores and generation
outcomes, native source fingerprints, and development evaluation contracts.
Missing permissions or inputs stop at the named path.

If A's original policy file is not discoverable under its run, B data or parent
records, configure with `--a-policy /actual/original/tokenization_policy.json`
using a fresh configuration filename. Existing configurations are immutable.

## Submit the three remaining combinations

After preflight passes:

```bash
bash scripts/sol/replay_c1_submit.sh \
  /scratch/jseth2/replay_c1_05b_parent123_config.json \
  --attempt attempt01 --evaluation-attempt eval01 --authorize-gpu

squeue -u jseth2 -o "%.18i %.32j %.2t %.10M %R"
```

This submits seven serial `afterok` stages: selection, smoke, train42, train137,
train2026, evaluation, and final reporting. It creates three new full hybrid
adapters from the original 0.5B parent123 adapter, with one disposable 16-example
smoke first. A is evaluated from its existing adapters in the common runtime;
matching native C1 evaluations are reused. A and C1 are not retrained.

Each full run uses the same fixed 50/50 supervised-token setting, with a target
of 318845 additional supervised tokens. Whole-example halves each contain
158626–160219 supervised tokens and 500 examples, with the same source/length
quotas and per-stratum token bounds as the completed runs. Replay uses the
original parent examples, and fresh C1 selection uses the complete 4955-bank
cached scores/outcomes for parent123. The selection is fixed across continuation
seeds. The original optimizer reset, warmup, encoding, bf16/LoRA/NF4 and training
recipe remain in force. No ratio sweep, reference rescoring or new candidate
generation is performed.

The parser-status fix is included: `empty` and `empty_marker` remain false
no-answer outcomes without modifying saved records. Unknown statuses and genuine
checker errors still block evaluation. Tests exercise every outcome emitted by
the exact hash-pinned parser.

## Verified profile paths

| Input | Location |
|---|---|
| Original parent | `/scratch/sshind38/capstone_approach_b/parents/0.5B_seed123/final/adapter` |
| Native C1 profile | `/scratch/sshind38/capstone_c0_c1_work/outputs/C0_C1_overnight_training_v1/7376bbd91d80e726/profiles/0.5B_seed123/50234d324ee09a25` |
| C1 controls | `<native profile>/runs/C1/seed<42 or 137 or 2026>_attempt01/final/adapter` |
| A sidecars | `/scratch/msures15/eval_results/A_0.5B_p123_s<42 or 137 or 2026>_gsm.config.json` |
| A adapters | Exact `checkpoint_id` read from each sidecar |
| Published C1 evaluation | `/scratch/sshind38/github_screened_4955_latest/C1/evaluation/0.5B_parent123_seed<seed>/<benchmark>` |

The model is `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. The parent123 adapter SHA256 is
`aa11429355543bcfe35e47856a3c6ada4e5e0eb7f4647318c1c68aee5bf1c3a2`.
The shared bank and parent-manifest hashes, native runner and evaluator hashes
are unchanged. The profile was checked against the published native provenance
at GitHub commit `2c88b654a904ae6e1884d5b7ac1771e02a2eda7e`.

## Final results

Preflight prints the exact new identity and output directory:

```text
/scratch/jseth2/replay_c1_4955_pilot/0.5B_parent123/<identity>/
```

Once the final reporting job shows `COMPLETED` and exit code `0:0`, locate the CSV:

```bash
find /scratch/jseth2/replay_c1_4955_pilot/0.5B_parent123 \
  -path '*/reports/eval01/accuracy.csv' -print -exec cat {} \;
```

Its report directory contains `accuracy.csv`, `paired_differences.csv`,
`run_index.csv`, `summary.json` and `provenance.json`. Each full run must have a
passed, non-smoke completion receipt with reconciled replay/fresh label counts
and a matching saved-adapter reload probe. The native exposure and historical
matcher-audit limitations remain disclosed in the reports.

## Validation

Local tests use synthetic fixtures and preserve their artifacts. A pinned
OR-Tools check is skipped when the separate solver environment is unavailable;
the real selection and GPU checks still take place on Sol. No live Sol training
or evaluation was executed during release preparation.

# Replay + C1: separate 1.5B deployment

This package runs the same fixed 50/50 supervised-token pilot for
`Qwen/Qwen2.5-1.5B`, with original R1000 parent seeds **17 and 123**, and continuation
seeds **42, 137, and 2026** for each parent. It creates six full hybrid runs and two
disposable smokes. Implementation and CPU tests do not establish live Sol readiness
or an accuracy improvement. No GPU job was executed or submitted during packaging.

## Transfer and extract

Run in Windows PowerShell from the repository directory:

```powershell
scp '.\outputs\person4\replay_c1_release\replay_c1_15b_v1.zip' jseth2@sol.asu.edu:/scratch/jseth2/
```

Run on Sol. Use the new directory so the running 0.5B pipeline stays unchanged:

```bash
mkdir /scratch/jseth2/replay_c1_15b_code_v1 && \
  unzip /scratch/jseth2/replay_c1_15b_v1.zip -d /scratch/jseth2/replay_c1_15b_code_v1

export REPLAY15B_REPO=/scratch/jseth2/replay_c1_15b_code_v1
export REPLAY15B_PYTHON=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python
cd "$REPLAY15B_REPO"
```

The archive contains only the separate package, scripts, examples, this guide and
checksums. Data, original parents, cached scores and control adapters stay on Sol.
Do not extract it over the 0.5B code or the native C1 runtime. Do not change package
files after selection/submission: code content forms part of the run identity.

## Create both configurations and check inputs

These commands write new configuration files, then perform read-only preflight.
They do not load GPU models or submit jobs. Run them from the new code directory:

```bash
for parent in 17 123; do
  "$REPLAY15B_PYTHON" -B -m capstone_person4_replay_c1_15b configure \
    --parent-seed "$parent" \
    --config-output "/scratch/jseth2/replay15b_parent${parent}_config.json" \
    --output-root /scratch/jseth2/replay_c1_15b_4955_pilot || break
done

for parent in 17 123; do
  "$REPLAY15B_PYTHON" -B -m capstone_person4_replay_c1_15b preflight \
    --config "/scratch/jseth2/replay15b_parent${parent}_config.json" || break
done
```

Both preflights must return `"passed":true` before submission. The output names
the parent, all six verified A/C1 controls, the identity and exact output directory.
The 1.5B A adapters may need their owner to grant read/traverse access, just as the
0.5B controls did. A permission error identifies the actual inaccessible path.

`configure` preserves an existing identical file and refuses to overwrite a
different file. If the native runner is installed elsewhere, specify
`--native-runner /actual/path/nightly.py`. If A's exact recorded policy cannot be
found under its run, B data, or parent records, use
`--a-policy /actual/original/tokenization_policy.json` with a new config filename.
Do not change expected hashes to bypass preflight.

## Input paths and model-specific binding

| Input | Sol location |
|---|---|
| B root | `/scratch/sshind38/capstone_approach_b` |
| Work root | `/scratch/sshind38/capstone_c0_c1_work` |
| Native runner | `<work>/C0_C1_Overnight_Training_20261110/nightly.py` |
| Parent examples | `<B root>/data/train_n1000.jsonl` |
| Parent adapter | `<B root>/parents/1.5B_seed<parent>/final/adapter` |
| Parent17 native profile | `<work>/outputs/C0_C1_overnight_training_v1/7376bbd91d80e726/profiles/1.5B_seed17/c68a89948de82f90` |
| Parent123 native profile | `<work>/outputs/C0_C1_overnight_training_v1/7376bbd91d80e726/profiles/1.5B_seed123/2e8494a4fd013ecd` |
| Fresh bank, shared across profiles | `<work>/outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17/35f8a5ecbf57cd37/candidate_bank.jsonl` |
| Cached reference scores | Each native profile's verified `reference/` cache, discovered by the pinned native loader |
| Cached C1 outcomes | Each native profile's `generation/` chunks and receipts |
| C1 training controls | Each native profile's `runs/C1/seed<seed>_attempt01` |
| A sidecars | `/scratch/msures15/eval_results/A_1.5B_p<parent>_s<seed>_gsm.config.json` |
| A adapters | Exact `checkpoint_id` read from each A sidecar; timestamps are not guessed |
| Native development evaluation | `<work>/outputs/C0_C1_development_evaluation_v1/931e8fb821c7d70f` |
| Published C1 results fallback | `/scratch/sshind38/github_screened_4955_latest/C1/evaluation/1.5B_parent<parent>_seed<seed>/<dev_gsm or dev_math>` |

The shared bank's historical directory contains `0.5B_seed17`; its immutable
**4955 question/reference rows** are shared. The 1.5B pipeline uses each 1.5B
parent's own reference scores, independent-answer outcomes, selection receipts,
tokenizer proof and training controls. It does not reuse 0.5B model measurements
or exchange measurements between parents.

The native code and published records are pinned to
[GitHub commit 2c88b654](https://github.com/jahnvisethjs/infj-llm-distillation-improvement/tree/2c88b654a904ae6e1884d5b7ac1771e02a2eda7e).
Model revision: `8faed761d45a263340a0528343f099c05c9a4323`.

| Immutable input | SHA256 |
|---|---|
| Parent17 adapter | `e0e419f2963dae8a6ae8f03637fdf5c255c8634ddf882360164b6f6843c30a75` |
| Parent123 adapter | `849ca63fb2ebffc489144c68f00ac8513f1b9cae0c827c9fe4586cf6b38d370d` |
| Shared 1000-row parent manifest | `efa258c9c4a8bffa85bb3be2026c187e9035f3918fe044967c4dafadcfa7f303` |
| Shared 4955-row fresh bank | `ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f` |
| Native runner | `50809b9e374fcc05a9b637c381396b46223ab85abcda3ca07cd4b6a5a4af882f` |

Native dependency, measurement, generation and actual-label checks remain active.
An absent native evaluation receipt permits the known published fallback. A
present but corrupt or inaccessible native receipt stops the run. Compressed
predictions are checked against the native hash of their uncompressed bytes.

## What each parent runs

The continuation target is **318845 supervised tokens**. Each whole-example half
must have **158626–160219** supervised tokens, within 0.5% of half that target.
Both halves have 500 examples: 25 per frozen source/solution-length stratum, with
stratum tokens within 5% of the corresponding half-parent tokens. A feasible
selection, rather than an exact 50% example count alone, establishes token balance.

Replay uses deterministic hash priorities over the original parent examples.
Fresh selection recomputes native fit ranks over all 4955 candidates and maximizes
`round(1000000 * fit) * (1 + confirmed_mistake)`, using only cached measurements.
The chosen mixed manifest stays fixed across the three continuation seeds for
that parent. No ratio sweep or new scoring/generation is performed.

Every full run starts from its original R1000 parent, with fresh optimizer and
scheduler state, the existing one-epoch recipe, learning rate 0.0001, 10 warmup
updates, accumulation 8, microbatch 1, bf16 and no packing. The native LoRA/NF4,
masking, clipping, loss and checkpoint behavior are retained. A 16-example smoke
(8 replay + 8 fresh) must pass first. Success receipts require completed token
consumption, an adapter update, unchanged original parent and a matched saved
adapter reload probe. Incomplete attempts are preserved; use a new attempt name.

## Submit after successful preflight

The following commands explicitly authorize GPU execution when **you run them**.
Each parent gets seven Slurm stages chained with `afterok`:
selection → smoke → train42 → train137 → train2026 → evaluation → finalization.
Both parent chains may run concurrently, using at most one A100 per chain.

```bash
for parent in 17 123; do
  bash scripts/sol/replay15b_submit.sh \
    "/scratch/jseth2/replay15b_parent${parent}_config.json" \
    --attempt attempt01 --evaluation-attempt eval01 --authorize-gpu || break
done

squeue -u "$USER"
```

Defaults are account `class_fse57081354fall2026`, partition `public`, QOS `class`,
one A100 for GPU stages, 32G host memory and four hours per stage. Override account,
partition and QOS during configuration if required by your allocation. Existing
active submissions are rejected; submission/job ledgers are preserved. Failed
jobs prevent dependent stages from running. No automatic cancellation, retry or
package installation is performed.

Evaluation reuses 12 matching C1 benchmark files across the two parents. It
evaluates the six existing A adapters and six new hybrid adapters on both frozen
500-question development benchmarks: 24 new benchmark files. The common runtime
is fp32 merged adapters, greedy decoding, 1024 new tokens, batch16, parser v6,
Torch2.6.0+cu124, Transformers4.57.1 and PEFT0.17.1 on an A100. Earlier A accuracy
rows from a different runtime are not substituted. A and C1 are not retrained.

## Results and checks

Each preflight prints its exact identity. Results stay under:

```text
/scratch/jseth2/replay_c1_15b_4955_pilot/1.5B_parent<17 or 123>/<identity>/
  selection/{selection.json,replay.jsonl,fresh.jsonl,train.jsonl,membership.jsonl}
  smokes/seed42_attempt01/completion.json
  runs/seed<42 or 137 or 2026>_attempt01/completion.json
  evaluation/eval01/<method>_seed<seed>/<benchmark>/completion.json
  submissions/attempt01_eval01/{submission.json,job_*.json,*.out,*.err}
  reports/eval01/{accuracy.csv,paired_differences.csv,run_index.csv,summary.json,provenance.json}
```

```bash
find /scratch/jseth2/replay_c1_15b_4955_pilot -path '*/reports/eval01/accuracy.csv' -print
find /scratch/jseth2/replay_c1_15b_4955_pilot -path '*/runs/*/completion.json' -print
```

Each parent's report includes all three methods, all three continuation seeds,
both benchmarks, mean accuracy, training-seed sample SD and matched hybrid-minus-A
and hybrid-minus-C1 contrasts. Question-paired bootstrap intervals describe fixed
checkpoints; they do not estimate training-seed uncertainty. Report the parents
separately before pooling them. Improvement is a hypothesis, not a guaranteed
outcome. Inherited parent/development exposure and the unresolved historical
matcher audit remain disclosed. Reserved final tests are not used.

## Local verification

```bash
python -B -m unittest discover -s capstone_person4_replay_c1_15b/tests -v
```

Tests use explicit synthetic fixtures and preserve their artifacts. The true
OR-Tools solver test requires the pinned Sol solver environment; a local skip
does not validate a real selection. Live permissions, all native inputs and
model availability must still pass preflight and the GPU smoke on Sol.

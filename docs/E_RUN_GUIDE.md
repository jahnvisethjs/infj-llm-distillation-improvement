# Approach E: mistake-only continuation

## Required outputs from the completed C1 SOL run

Keep the four original SOL configs and their output tree accessible. Before E,
retain the exact code revision/source hashes used for C1 and, for each parent:

- `prepared/candidates.jsonl`, `preparation.json`, `parent_rebinned.jsonl`,
  token verification, protection and parent/release binding evidence.
- `generation/request/<request-name>/{questions.jsonl,request.json}`,
  `generation/outcomes.jsonl`, and `generation/coverage.json`.
- B/C0/C1 selected manifests and allocation reports, plus
  `selection/C1/batch_gate.json` linking C1 selection to the exact outcome hashes.
- Completed C0 and C1 runs at seeds 42/137/2026: run configs, training summaries,
  token reconciliation, checkpoints, environment records and evaluation handoffs.

The Windows workspace has the original parent package and four valid raw grading
pairs in `../c1_candidates/`, but not the completed shared-bank SOL output tree.
Do not rebuild that tree locally or regenerate the original parents. If a small
metadata bundle is needed for inspection, copy the records above; large weights
and reference-loss caches can remain on SOL. The read-only preflight prints the
specific missing/stale file or failed gate and exits 2. It does not launch jobs.

## What E reuses

E uses C1's **original-parent candidate grading**, not benchmark predictions from
C0-trained adapters. Only the existing classifier's `incorrect` status gets
`w=1`. Missing answers, ambiguity and truncation without a checked answer stay
distinct, and execution/checker errors must be resolved before ingestion.

The full matrix is both Qwen sizes (0.5B/1.5B), parent seeds 17/123, and
continuation seeds 42/137/2026: **12 new E runs**. All four parents are included,
without conditioning inclusion on favorable C1 development accuracy. This
supersedes the E execution restriction in the older C1 implementation plan.

Selection uses the frozen 6,409-row Method B bank and its existing CP-SAT solver:
1,000 rows, 50 per source/length cell, 318,845 supervised tokens +/-0.5%, and
per-cell tokens +/-5%. It maximizes confirmed-mistake count, then Method B's
seeded random utility, then the existing solver hash tie-break. A dominance
multiplier larger than any possible total secondary utility encodes the first
two objectives exactly. A constant mistake term is omitted at fixed quotas so
constant-indicator E sends B's exact objective to the solver. Reports retain
the solver status/gap; a feasible incumbent is not called an optimum.

All eligible rows remain available. In particular, the 1.5B parents have only
22/28 confirmed mistakes in the 50-row `orca|0` cell, so non-mistake examples
are needed for coverage. Token constraints may require additional such rows.
Reference-loss caches, fit ranks and lambda never enter E selection.

## Separate entry point and immutable artifacts

Use `python -m capstone_person4_approachE`, not the reserved E option in the
historical core CLI. The fingerprinted core and C1 generation module are
unchanged. E reads their artifacts through the existing validators and writes
under `<output_root>/<profile>/e-<fingerprint>/`:

```text
selection/E/manifest.jsonl
selection/E/ordered_ids.json
selection/E/selection_table.csv
selection/E/allocation_report.json
training_smoke/E/esmoke01/
runs/E/epilot01/
handoffs/<identity>/evaluation_handoff.json
```

Selection identity binds parent/candidate/outcome hashes, reused code, E code,
the classifier and allocator settings. Training additionally binds the selected
report and training recipe/seed. E's process-scoped trainer adapter restores all
shared bindings even on failure. Its smoke key excludes the continuation seed
but includes the E selection and code, so one E smoke may cover three seeds;
C0/C1 smoke records cannot unlock E. Source/config mismatches are hard failures,
not reasons to rewrite saved fingerprints or rerun historical experiments.

Exact reruns reuse saved selections without solving again. Different existing
outputs are preserved. Changed training attempts need new names. The wrapper
does not automatically resume interrupted training or submit duplicate-safe
jobs; inspect existing jobs/artifacts before any recovery submission.

## SOL commands

Use the same `C0_REPO`, `C0_PYTHON` or `C0_ENV`, and `HF_HOME` exports as C1.
Use the existing SOL configs; do not regenerate them with a different pool.

```bash
cd "$C0_REPO"
CFG=configs/c0.local.sol.0.5b.seed17.json
"$C0_PYTHON" -B -m capstone_person4_approachE preflight --config "$CFG" \
  --c0-attempt pilot01 --c1-attempt c1pilot01 --seeds 42 137 2026
```

Substitute actual completed C0/C1 attempt names. The preflight validates the
prepared bank, grading, controls and all requested completed control runs.
GPU readiness is verified separately by the E smoke. OR-Tools must have the
existing pinned version 9.14.6206; no new scientific dependencies are added.

After explicit GPU-experiment authorization:

```bash
bash scripts/sol/submit_e_pipeline.sh "$CFG" q05-p17-e \
  pilot01 c1pilot01 epilot01 esmoke01
```

This submits six jobs for one parent:

```text
selection -> E smoke -> seed42   --+
                    -> seed137  --+-> finalization
                    -> seed2026 --+
```

Jobs use the existing course account/public/class setup, four CPUs, 32 GB RAM,
and one A100 for smoke/training. Caps: 30 minutes for selection/smoke, one hour
per training run, 15 minutes for finalization. These are resource requests, not
new throughput measurements. `E_SEEDS="42"` selects one continuation seed;
`E_SBATCH_ARGS` adds scheduler options. Failed prerequisites block downstream
jobs through `afterok`. No generation, reference scoring or benchmark job is
submitted. Validate 0.5B/parent17 first, then repeat for the other three configs.

Individual stages and finalization recovery:

```bash
"$C0_PYTHON" -B -m capstone_person4_approachE select --config "$CFG"
"$C0_PYTHON" -B -m capstone_person4_approachE train --config "$CFG" \
  --training-seed 42 --attempt esmoke01 --smoke
"$C0_PYTHON" -B -m capstone_person4_approachE train --config "$CFG" \
  --training-seed 42 --attempt epilot01
"$C0_PYTHON" -B -m capstone_person4_approachE finalize --config "$CFG" \
  --attempt epilot01 --c0-attempt pilot01 --c1-attempt c1pilot01 --seeds 42 137 2026
```

The unchanged Person 2 trainer reloads the original parent, starts a fresh
optimizer/scheduler, trains once on original correct completions and reconciles
all selected exposures/tokens. Finalization rejects smoke/incomplete runs and
emits an `e_run_index.csv` path. It can be rerun without training again.

## Additive handoff to Megha

Package the four emitted indexes after all runs reconcile:

```bash
"$C0_PYTHON" -B -m capstone_person4_approachE package \
  --run-index /actual/q05p17/e_run_index.csv /actual/q05p123/e_run_index.csv \
              /actual/q15p17/e_run_index.csv /actual/q15p123/e_run_index.csv \
  --output /actual/new/E_megha_handoff
```

The new directory contains `E_evaluation.zip` and its relative checksum. The
archive includes adapters/configs, selected manifests/audits, reconciled run
records, available loss logs, frozen generation coverage, evaluation handoffs,
portable run index, source-to-package path mapping, code hashes/revision,
inventory and relative file checksums. Partial matrices are explicitly labeled.
No optimizer states, intermediate checkpoints or duplicate baseline weights are
included. Original records keep their historical paths. The checkpoint metadata
is also supplied at `final/run_config.json` for Megha's evaluator lookup.

Give Megha this additional package and the exact shared location/read access.
She can reuse the compatible C0/C1/parent/B files already provided. Ask for E
evaluation on the same frozen development GSM8K/MATH questions and protocol,
per-question predictions plus actual evaluator configs, and matched C1-minus-E,
E-minus-C0, E-minus-B, E-minus-parent comparisons. Each E adapter's evaluator
guard must use its **own E continuation-manifest hash** from the index. Keep
benchmarks/model sizes separate and distinguish paired-question uncertainty
from continuation-seed variability. Reserved final tests need separate approval.

## Local validation

```bash
python -B -m unittest discover -s capstone_person4_approachE/tests -t . -v
python -B -m unittest discover -s capstone_person4_approachC1/tests -t . -v
python -B -m unittest discover -s tests/person4 -v
python -B scripts/validate_e.py
```

Tests use visibly synthetic fixtures, an exhaustive small-case selection oracle,
the actual shared solver when installed, the unchanged continuation boundary,
and recording scheduler stubs. They do not run a real model or submit GPU jobs.
Missing OR-Tools skips only the explicitly marked real-solver test. Test artifacts
are additive under `.test_artifacts/`; these fixtures cannot satisfy real gates.
The validation runner retains JSON/results logs. Its optional `--solver-path`
uses an existing isolated dependency directory without changing system packages.

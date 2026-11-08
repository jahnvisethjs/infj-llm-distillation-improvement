# C0 implementation plan: reference-fit selection for continuation training

Date: 29 October 2026 (America/Phoenix)  
Branch: `person4/c0-reference-fit`  
Starting branch/commit: `person2_training` / `13a320f`  
Status: C0/B infrastructure implemented; see [the run guide](C0_RUN_GUIDE.md). Real-parent scoring and continuation remain gated by required assets and runtime. The sections below retain the original design and acceptance criteria.

Current planning update, 5 November 2026: use [the V4 implementation plan](C0_V4_IMPLEMENTATION_PLAN.md) for the supplied data and R1000 parents, the newly identified candidate-pool filtering work, and execution through continuation with accuracy evaluation deferred. The missing-parent/data statements below describe the earlier inspection and are superseded by that update.

## 1. Intended outcome

Build C0: select fresh mathematical question–solution pairs that fit an already fine-tuned Qwen parent, then continue training from that same parent under a declared additional supervised-token budget.

The first research comparison is **C0 versus fresh random selection (B)**. Both arms use the same parent, frozen candidate pool, source/length coverage, budget allocator, continuation recipe, and evaluation protocol. Evaluate the untouched parent as the no-additional-training control.

This establishes a reusable baseline for the later **C1 versus C0** experiment. C1's independent-answer correctness signal is outside this implementation phase.

This document refines the supplied [student-aware implementation brief](../../Student_Aware_Selection_Codex_Implementation_Plan.md) for C0. That brief currently lives outside the Git repository; this plan records the decisions needed here without requiring that workspace path at runtime.

## 2. Method and research boundary

[GRAPE](https://arxiv.org/html/2502.04194v3#S3) motivates choosing supervision with higher length-normalized likelihood. Its main method selects among responses to a fixed instruction using the pretrained model. Our C0 instead selects question–solution pairs using a fine-tuned parent. Describe C0 as a **GRAPE-inspired reference-fit baseline**, rather than a reproduction. The paper explicitly discusses [response versus instance selection](https://arxiv.org/html/2502.04194v3#S7).

For each candidate, compute teacher-forced completion loss:

```text
L_i = -sum(log p_parent(y_i,t | question_i, y_i,<t)) / N_i
```

Here `N_i` is the actual number of completion labels supervised after causal shifting, including exactly one EOS under the verified training contract. Exclude prompt and padding labels. Use natural logarithms; loss units are nats/token. Lower `L_i` means better reference fit, not proven independent correctness or learnability.

Within each fixed `(dataset_id, length_bin)` stratum:

```text
a_i = average ascending rank of L_i, with tied losses sharing a rank
r_i = (a_i - 0.5) / number_of_rows_in_stratum
f_i = 1 - r_i
C0 priority = descending f_i, then SHA256(selection_seed, row_id)
```

Singleton strata have `f_i = 0.5`. Do not discard the lowest-loss quartile. A middle-band method would be a separate `C0_band` experiment. Use priority only to select rows; continue training on the unchanged references with the existing unweighted objective.

## 3. What the existing code supplies

Evidence comes from static inspection, not a fresh model run.

| Existing component | Reuse path | Required adaptation or verification |
| --- | --- | --- |
| [Person 2 notebook](../Person2_SOL_Seeds17_123_UpTo4000.ipynb), cell writing `trainer_core.py` | `build_model`, training loop, checkpoint saving, reload checks, token logging | Make definitions importable without executing setup, installation, or historical runs |
| Same notebook, cell writing `person2.py` | Model profiles, `select_model`, `encode`, `collate`, release/fingerprint checks | Expose a manifest-to-continuation interface |
| Person 1 `person1_data_contract` from the verified release | Combined-text encoding, prompt boundary, positional padding, EOS and manifest validation | Pin the code and release hashes actually associated with the parent |
| [Saved training archive](../seeds17_123_training_results_small.zip) | Run configurations, training summaries and exposure evidence | Tie the chosen records and manifest to the actual saved adapter |
| Workspace `eval_results/` | Per-question outputs, run metadata, generation settings and evaluator hashes | Recover matching evaluator/parser source and frozen question manifests before new comparisons |

The notebook names pinned Base models:

| Profile | Model | Recorded revision |
| --- | --- | --- |
| 0.5B | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| 1.5B | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |

The shared encoding is plain text, without a chat template; prompt labels are ignored, completion labels and one EOS are supervised. Verify those token IDs against the selected parent rather than treating this inspection as runtime validation.

### Integration differences that must be handled explicitly

1. Existing `branch` training restores the parent's optimizer/scaler and skips new warmup. This plan requires reloading the trained adapter with a **fresh optimizer and schedule for each continuation arm**. Add an explicit continuation mode; preserve historical branch behavior.
2. Existing `parent_budget_check` requires a completed 1,000-row `curve` parent and checks per-source token totals within 1%. The new gate must bind the actual configured parent, use the shared total-token budget tolerance below, and report source row/token shares separately.
3. The wrapper's training entry point sets `a.parent = None`, so calling it unchanged would start another base-initialized run. A dedicated adapter must pass the verified parent to the reused training engine.
4. The notebook refers to a verified seeds-17/123 release, while another workspace handoff describes `person1_v3_20261012_ready`. Do not mix their manifests, summaries, or hashes. Resolve the release associated with the chosen checkpoint.
5. The inspected small training archive contains no adapter weights. Evaluation metadata records SOL adapter paths and digests, but those paths do not establish local availability.

## 4. Proposed implementation layout

Names are proposed interfaces, not commands or modules that already exist.

```text
capstone_person4/
  __main__.py             # stage CLI
  config.py               # validated profiles and unresolved-input checks
  audit.py                # evidence-backed readiness and stage gates
  person2_adapter.py      # verified loader/encoder/trainer integration
  data.py                 # fresh pool, split controls, frozen manifests
  reference_scoring.py    # per-token and per-example completion NLL
  selection.py            # C0 and B priorities
  allocation.py           # shared coverage and whole-example token budget
  continuation.py         # parent reload, fresh optimizer, consumed tokens
  evaluation.py           # adapter to Person 3's evaluator and paired reports
  provenance.py           # hashes, cache keys, atomic writes and resume
configs/c0.example.json
tests/person4/
docs/C0_RUN_GUIDE.md
```

Extract only the notebook's required source definitions into importable modules, recording source cell/content hashes. Avoid a second independent training loop. Do not execute notebook setup or training cells to reach the loader. Keep the original notebook unchanged unless a later approved change is necessary.

Treat Person 1 as a versioned external input supplied through configuration. The existing untracked `capstone_person1/` directory is not evidence of a complete, committed dependency. Do not copy it wholesale or stage its cache files.

Keep heavy model dependencies lazy so audit/data/selection fixtures can run without a GPU. Start with Person 2's pinned dependency stack; propose changes only when a concrete compatibility issue requires them.

## 5. Configuration and readiness

Use one resolved configuration with separate 0.5B and 1.5B profiles. Unavailable inputs remain `null` in the example, rather than plausible placeholder values.

Required fields include:

- Parent run ID, base model/revision, adapter path/hashes, adapter settings, and complete training-exposure evidence.
- Release path/hash, canonical manifest, tokenizer/template/mask/EOS fingerprints, and context limit.
- Development/final question manifests, group-exclusion evidence, and evaluator/parser source hashes.
- Candidate/split/selection/training seeds, candidate count, source/length definitions and quota proportions.
- Reference-scoring dtype, quantization, batch settings, numeric tolerance and cache location.
- Token target, its verified or provisional origin, relative tolerance, and allocator search settings.
- Continuation learning rate, warmup/schedule policy, accumulation, precision, packing policy, and output paths.
- Evaluation decoding, maximum output tokens, stopping rules and checker version.

Each audit check records `PASS`, `MISSING`, `MISMATCH`, `UNVERIFIED`, or `WARNING`, evidence, affected stages and remedy. Write both `readiness_report.json` and `readiness_report.md`.

| Stage | Evidence required to unlock it |
| --- | --- |
| Synthetic dry run | Only clearly marked synthetic fixtures; cannot unlock real stages |
| Prepare real fresh pool | Valid canonical release, complete parent-seen groups, and declared dev/eval exclusions |
| Scientific reference scoring | Verified parent/base/tokenizer/serialization/mask and fresh candidate manifest; compatible runtime |
| C0/B selection | Complete common eligible universe, valid reference-score coverage for C0, fixed quotas and budget |
| Continuation training | Verified selected counts, matching parent, resolved continuation recipe and runtime |
| Evaluation/comparison | Frozen shared questions, evaluator/checker, decoding protocol and valid run provenance |

A missing answer checker must not block reference scoring; a missing checkpoint must not block synthetic allocator tests. Missing development inputs block tuning and controlled comparison claims, while explicitly limited frozen-rule diagnostics can still proceed. Scientific scoring must not silently substitute base weights for the trained parent.

## 6. Data preparation

1. Load the immutable canonical release actually associated with the parent.
2. Resolve all parent training exposures from checkpoint-bound manifests, summaries and logs, including earlier stages and repeats. A run name alone is insufficient.
3. Remove all parent-seen, development and final-evaluation problem groups; reuse the release's quality and exact/near-duplicate controls. Apply those controls across any newly introduced split boundary.
4. Verify nonempty questions/references, unique row/group IDs, source metadata and whole-example context eligibility. Preserve uncertainty about corpus-wide reference correctness.
5. Freeze up to **5,000 fresh candidates** deterministically using source/length strata before seeing model scores. Use the existing selection bank where its release and freshness evidence match. If fewer are available, record the shortfall.
6. Keep the same eligible universe for C0 and B. Retry or commonly exclude failed measurements before comparison; do not silently shrink only one arm's pool.
7. Preserve release length-bin edges. Do not rebucket after observing C0 scores or claim topic balancing when only source/length metadata exists.

Persist canonical rows, IDs/group IDs, split membership, references, token counts, text/render hashes, revisions and exclusion reasons. A held-out reference-loss bank is not automatically a frozen mathematical-accuracy development set.

## 7. Reference scoring

Run a small real-parent smoke check before the 5,000-row pass. The first planned smoke uses 8–32 eligible rows spanning sources and lengths; its size is an engineering choice.

Scoring procedure:

1. Verify and load Person 2's base plus trained adapter with `trainable=False`; confirm adapter activation. Enter evaluation and inference mode.
2. Encode the exact question/reference combination using the shared contract. Reject overlength rows instead of truncating their solutions.
3. Align logits at position `t-1` with labels at `t`. Supervise only nonignored completion/EOS labels after this shift. Mask padding by position/attention, preserving genuine EOS when pad and EOS IDs match.
4. Compute unreduced next-token cross entropy, accumulating per-row sums/counts and means. Avoid using the model's scalar batch loss for row ranking. Use numerically stable float32 loss arithmetic on the configured model logits.
5. Store every supervised token's ID, sequence position and NLL in a linked compressed artifact for the first experiment. Verify its sum and mean reconstruct the row scores.
6. Compare single-example and padded-batch scores under a declared tolerance; establish the tolerance from dtype and smoke evidence before scaling. Verify the parent was not updated.
7. Write atomic, resumable score shards keyed by parent, tokenizer, serialization, mask/EOS, rendered text, precision/quantization and code fingerprints. Partial failures must not replace valid records.

Each row record includes `row_id`, parent/render fingerprints, mean NLL, total NLL, effective supervised count, per-token artifact reference and status. Keep raw losses alongside normalized ranks. Record throughput, token counts, GPU/memory and wall time.

C0 does not need candidate-wide answer generation. At this stage the model measures the supplied references; generation is needed later for evaluation and C1.

## 8. Shared selection and token allocator

Define `T` as the re-encoded, verified supervised-token total of one additional pass over the chosen parent's unique-row dataset when replay is the anchor. Record historical repeats separately. Otherwise require an explicitly configured pilot budget labeled provisional.

Do not use the older brief's 333,328-token R1000 figure automatically. The inspected v3 handoff lists 318,845 reference supervised tokens for its R1000 manifest, but neither figure is the new experiment's budget until reconciled with its actual parent and labels.

Use a configurable **0.5% relative tolerance** around `T`. This is an engineering tolerance, not a scientific constant. Match actual supervised tokens; row counts may differ between arms.

Proposed deterministic allocation:

1. Fix row-allocation proportions over source/length strata from verified parent coverage, where feasible. Declare a shared alternative before selection if that coverage cannot be supported.
2. Estimate a feasible selected row count from `T` and candidate lengths, then explore a recorded bounded range of row counts, expanding the range deterministically when necessary.
3. Convert proportions to integer quotas using largest-remainder allocation with stable stratum ties and explicit capacity checks.
4. For each quota vector, start with the highest-priority rows within each stratum. Use a configured bounded boundary-swap search to approach `T`, keeping quotas fixed and recording any priority sacrifice needed for the budget.
5. Prefer feasible solutions with the least departure from within-stratum priority, then smaller budget error, then stable row-ID ties. Define the priority cost using ordering/ranks so uniform positive utility rescaling cannot change allocation. Fix and test the precise objective before experiment execution.
6. If the search finds no valid solution, fail with target/coverage deficits and the nearest solution found. Distinguish proven capacity infeasibility from exhaustion of a bounded search. Do not silently widen tolerance or relax coverage.

Use the identical allocator for B, replacing fit utility with deterministic seeded random priorities. Use whole examples once each, with no padding examples, artificial target tokens, partial references or repeated fresh rows to fill a gap. Record requested/realized quotas, selected rows and source token shares.

Keep the interface compatible with later C1 priorities. Tests must establish that zero mistake bonus and constant mistake indicators reproduce C0's selected IDs and order under uniform score rescaling.

## 9. Continuation and evaluation

Add a dedicated continuation entry point around Person 2's training engine:

- Reload the same verified trained adapter independently for B and C0, with its existing LoRA configuration and intended trainable parameters.
- Create fresh optimizer/scaler/schedule state. Do not use historical `branch` defaults or restore the parent's optimizer.
- Start with Person 2's inspected recipe as a proposal: AdamW, learning rate `1e-4`, microbatch 1, accumulation 8, no packing, and the verified model precision. Resolve one shared fresh warmup/schedule policy before running either arm; the existing non-branch engine uses 10 warmup updates. These are proposed continuation settings, not evidence of an optimal recipe.
- Expose each selected row once. Preserve the final incomplete accumulation group and token-weighted loss behavior. Log actual post-shift supervised tokens, seen IDs, updates and learning rates.
- Save the resulting adapter, configuration, manifest hashes and token reconciliation. A valid run must consume exactly its prescribed selected exposure schedule; the selected manifest's total must already satisfy the target tolerance.
- Treat resume of an interrupted new arm separately from initializing an arm: resume restores that arm's own state; starting another arm always reloads the parent with fresh optimizer state.

Recover Person 3's actual evaluator/parser rather than reconstructing it from aggregate scores. Evaluate untouched parent, B and C0 on identical frozen questions using identical question-only prompts, greedy decoding, stop rules, output cap, checker and precision policy. Report GSM and MATH separately and retain per-question outcomes, parse/truncation rates and prediction artifacts.

The inspected workspace export explicitly describes its questions as **development sets**. Use them only after confirming their identities and permitted role; historical exports are not new final-test evidence. Freeze method choices before final evaluation. Keep old aggregate tables separate from new controlled comparisons.

## 10. Execution order and deliverables

| Milestone | Work | Reviewable output |
| --- | --- | --- |
| M1 | Extract importable interfaces; audit assets; validate configuration and gates | Reuse map, readiness reports, exact missing-input template |
| M2 | Prepare data contracts; implement C0/B and allocator with synthetic fixtures | Deterministic dry run, selected manifests, scientific-integrity tests |
| M3 | Implement reference scoring; run bounded real smoke when assets/runtime are available | Verified token losses, batch comparison, parent integrity and throughput report |
| M4 | Separately execute the authorized 5,000-candidate reference pass | Complete score cache, per-token records, source/length loss diagnostics |
| M5 | Allocate C0/B at a declared pilot token budget | Trainer-ready JSONL, selection audit tables, matching/feasibility report |
| M6 | Integrate continuation; validate a disposable tiny training smoke | Parent isolation, fresh optimizer and actual token-consumption evidence |
| M7 | Separately execute B/C0 continuation and shared evaluation | Saved adapters, per-question predictions, paired comparison and compute report |
| M8 | Repeat promising comparisons; prepare later C1 extension | Training-seed variability, frozen method and extension contract |

Do not rerun historical learning-curve sweeps, retrain a parent, or regenerate Person 1's entire release to complete these milestones. Full scoring and experimental training are separate execution stages, not interface tests.

Proposed commands are `audit`, `prepare`, `score-reference`, `select --method B|C0`, `train`, `evaluate`, and `compare`, with `--config` and bounded `--smoke` options where appropriate. The run guide must contain runnable examples once those interfaces exist; no such commands are executable yet.

Runtime artifacts will live under a configured output root outside immutable input releases:

```text
readiness_report.{json,md}
resolved_config.json
splits/ and candidates.jsonl
reference_scores/ and per_token_losses/
diagnostics/reference_fit.json
selection/{B,C0}/manifest.jsonl
selection/{B,C0}/selection_table.csv
selection/{B,C0}/allocation_report.json
runs/{B,C0}/...adapter, logs, provenance, token_reconciliation...
evaluation/...per-question predictions...
comparison.{json,md}
```

## 11. Expected results and how to interpret them

### Expected implementation results

| Result | Expectation / acceptance condition |
| --- | --- |
| Reproducibility | Repeating selection with the same frozen inputs/configuration gives the same selected IDs and order |
| Reference fit | C0 prioritizes lower losses within each source/length stratum, subject to recorded budget adjustments; global unstratified mean loss need not always be lower |
| Freshness | No known parent/dev/eval group overlap; limitations of the duplicate detector remain documented |
| Budget control | B and C0 each satisfy the declared target/tolerance and coverage; actual training consumption matches their manifests |
| Provenance | Every score, selected row and trained adapter is tied to the correct parent, release, configuration and code |
| Operational behavior | Missing inputs cause actionable stage-specific failures; synthetic output never passes real experiment gates |

### Research expectations are hypotheses

- **Working hypothesis:** C0 may improve held-out accuracy over B by selecting supervision better aligned with the parent.
- **Plausible null result:** C0 and B may perform similarly because fresh data or existing coverage controls account for most benefit.
- **Plausible negative result:** C0 may underperform if it emphasizes already predictable material and misses useful difficult examples.
- Both continuation arms may improve over, tie, or regress relative to the parent. Lower selected reference loss alone does not establish better mathematical reasoning.

There is **no justified numerical accuracy forecast** before the pilot. Do not transfer GRAPE's reported accuracy differences to this setting or interpret historical base-to-fine-tuned gains as predicted continuation gains.

Report `C0 - B` and each arm minus untouched parent in percentage points, separately per benchmark. Use paired uncertainty over the same questions and report training-seed variability separately. Repeat with matched continuation seeds where practical; a single favorable run is pilot evidence, not a robust improvement claim.

Report both additional training tokens and total scoring-plus-training cost. Reference scoring consumes forward passes over the candidate references; estimate its 5,000-row wall time from measured token throughput and actual candidate lengths. Include evaluation, model-loading overhead, hardware and memory. No GPU-hour or speedup promise is justified before the smoke measurement.

Implementation success means correct, auditable execution or accurate blockers. Research success would require reproducible held-out improvement under the declared controls; these are separate acceptance criteria.

## 12. Focused validation before scaling

- Hand-verifiable causal loss fixture, prompt/padding masks, genuine EOS and per-token reconstruction.
- Unbatched versus padded-batch agreement, finite losses and unchanged parent parameters.
- Deterministic tied/singleton stratum ranks, random priorities and allocation under rescaling.
- Whole-example budget feasibility, insufficient strata, failed bounded search and exact exposure accounting.
- Group-disjoint splits and common eligibility/score coverage across B and C0.
- Cache invalidation after adapter, reference, template, mask or precision changes; atomic resume without duplicated rows.
- Continuation reloads the trained parent, uses fresh state per arm and preserves all selected rows exactly once.
- Shared evaluator question IDs/settings and paired predictions; no final-test-driven tuning.

These tests are planned. No model scoring, training, or runtime validation was performed when writing this plan.

## 13. Inputs to recover during M1

1. The chosen parent adapter/checkpoint files and checkpoint-bound run config, actual training manifest and full exposure/group evidence. Start with a saved 0.5B parent; use R500 only if its real artifacts are available. Do not recreate it to satisfy a preferred row count.
2. The exact data release and encoder files matching that parent, including source revisions, selection-bank metadata and tokenizer artifacts.
3. Person 3's evaluator/parser source and frozen development/final question manifests, including role and overlap evidence. Existing result metadata provides hashes and leads, not the complete interface.
4. A compatible Python/model runtime and allocated GPU for later real-model stages. The earlier workspace `python` invocation pointed to a missing interpreter; resolve an actual executable instead of assuming the shell command works. Person 2 records a Python 3.12 / PyTorch 2.6 CUDA stack, which should be checked on the execution machine.

Recover what the repository, archive and available handoffs already provide before requesting missing inputs. No unresolved input blocks writing the model-independent portions of the implementation.

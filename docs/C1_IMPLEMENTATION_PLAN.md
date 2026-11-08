# C1 implementation plan: independent correctness added to reference-fit selection

Date: 6 November 2026 (America/Phoenix)
Branch: `person4/c0-reference-fit`
Status: plan only. Nothing in this document has been implemented or executed.

## 1. Task and intended result

Two phases, in order:

- **Phase 0.** Rerun C0 on the candidate bank that Method B (Suraj Shinde) selected from, with B's coverage strata and B's three continuation seeds, for all four R1000 parents. That bank is then frozen for every later selector.
- **Phase 1.** Add the C1 selector (reference fit times a confirmed-mistake bonus) and train C1 arms on the same bank, parents and seeds.

The E (mistake-only) ablation is Phase 2 and reuses every Phase 1 stage; it is specified in section 9 so the code leaves room for it.

Decisions already made:

| Decision | Value | Reason |
| --- | --- | --- |
| Parents | Four: `0.5B/seed17`, `0.5B/seed123`, `1.5B/seed17`, `1.5B/seed123` | The parents that exist; B trained on all four |
| Candidate universe | B's main mixed bank, 6,409 rows (4,412 Orca + 1,997 Numina), ingested as-is | Identical universe to B; our protection scan reports but does not drop |
| Coverage strata | B's per-source decile intervals of parent supervised-token length, right-closed, 50 parent rows per stratum | Coverage matched to B; the bank already carries this `length_bin`, verified by recomputation |
| Allocator | Suraj's CP-SAT selector `select_b.py` (SHA `5c924123…`), vendored as `b_select.py` with a `utilities` parameter; reproduced B's 1,000 rows, 317,506 tokens and objective exactly on 8 November 2026 | One allocator for B, C0, C1 and E; see C0 rerun section 2.4 |
| Continuation seeds | 42, 137, 2026 per parent, same as B | Training-seed variability comparable to B's three-seed means |
| Selection and freeze seeds | 42, unchanged | Fixed selection; not redrawn per training seed |
| Token target | 318,845 ± 0.5% | Verified one-pass parent budget; B realized 317,506 |
| Mistake measurement | Person 3 generates and grades under the frozen contract (fp32, greedy, 1,024 new tokens, parser v6), once per parent | Same checker for scoring and evaluation |
| Lambda | 1.0, from config | Declared pilot value, not tuned |
| Per-parent measurements | Each parent gets its own reference scores and generation outcomes | No cross-model score reuse; the 1.5B tokenizer guard stays |

The primary contrast is C1 minus C0 at matched supervised tokens, same parent, same continuation seed. With B's runs this gives, per parent, three matched triples (B, C0, C1) at seeds 42, 137 and 2026. Nothing is pooled across model sizes.

Existing C0 runs on the 5,000-row pool are superseded. They stay on disk under their old artifact roots as a record, and no plan step depends on them. The `frozen_runs` pin from the earlier draft is therefore dropped.

## 2. Inputs from Method B (received 8 November 2026, verified)

Bundle `b_inputs_20261108T063441Z.tar.gz` in the workspace root, extracted to `b_inputs/`, 892 files, all SHA256s verified. Paths below are relative to `.../B_method_B_document_v3/` inside it. Read-only checks, no model loaded.

| # | Item | File | Verified |
| --- | --- | --- | --- |
| 1 | Bank | `shared/mixed_candidate_bank.jsonl`, SHA `806dd78f…` (matches `bank_audit.json` and every run receipt's `shared_universe_sha256`) | 6,409 rows, 4,412 Orca + 1,997 Numina, unique IDs, every ID in `data/manifests/selection_bank.jsonl`; every field identical to the selection bank except `length_bin` (31 rows differ, see item 3) |
| 2 | Filters | `shared/bank_audit.json`, `shared/parent_seen_groups.csv` | Screened pool 9,909; 1,000 dev-reference and 2,500 transfer rows reserved; parent-seen overlap 0; exact protected overlap 0; 7 parent-related development groups disclosed |
| 3 | Strata | `shared/parent_length_bins.json` (per-source decile cut points as fractions, `bisect_left`, right-closed); `shared/parent_rebinned_reference.jsonl` | Recomputing bins from the cut points reproduces the stored `length_bin` on all 6,409 bank rows, the 1,000 parent rows and the 1,000 selected rows with zero mismatches. Parent has exactly 50 rows per stratum. Parent ID set equals `train_n1000.jsonl` |
| 4 | Selected manifest | `mixed/budget_1x/1x/train_B.jsonl` (training order), `selected_ids.txt`, `budget_report.json`, `candidate_priorities.csv` | 1,000 rows, all in the bank, supervised tokens sum to 317,506 (target 318,845, −0.42%), 50 rows per stratum, `quota_deviations: 0`, `tolerance_relaxed: false` |
| 5 | Run receipts | `mixed/budget_1x/runs/B_<size>_p<parent>_t<seed>_1x/run_config.json` (12 full runs) plus 12 smokes; `runtime/config.json`, `runtime/execution_protocol.json` | Recipe: AdamW 1e-4, warmup 10, accumulation 8, microbatch 1, one epoch, bf16, no packing, fresh optimizer, weight decay 0, clip 1.0, 4-bit NF4 double quantization, LoRA inherited. Same environment as our C0 runs (torch 2.6.0+cu124, transformers 4.57.1, peft 0.17.1, bitsandbytes 0.48.1). Parent adapter SHAs match ours. Person 2's notebook uses the same NF4, clip 1.0 and weight decay 0, so the recipes are identical |
| 6 | Tokenization | `mixed/budget_1x/training_inputs/tokenization_policy.json`, `prompt_template.json` | Same prompt template and policy content as our C0 `training_inputs` (the policy hashes differ only in provenance layout) |
| 7 | Allocator | `mixed/budget_1x/selection_audit.json`; `runtime/allocator.py` | **Gap.** B's manifest came from a CP-SAT solve (`selection_implementation: "document Appendix A CP-SAT"`, ortools 9.14.6206, OPTIMAL, objective 912,175,166 = sum of selected `utility_integer`), with exactly 1,000 rows, 50 per stratum, per-stratum tokens ± 5% of the parent's, global ± 0.5%. The bundled `allocate()` is a different heuristic and does not reproduce B's rows (1,002 rows, 181 of 1,000 IDs). The CP-SAT code is not in the bundle. See [C0 rerun section 2.4](C0_B_BANK_RERUN.md) for the options |
| 8 | Protection parity | our old scan under `results/0.5B/17408cafafa059e7/protection/` | Our matcher would quarantine 572 and exclude 25 of B's 6,409 rows (79 of them are in B's 1,000). Keeping them keeps the universe symmetric; it is disclosed, not hidden. Normalised exact match of bank problems against all five evaluation files: 0 hits each. 0 rows over 2,048 tokens, 0 in `loss_dev` groups, 0 sharing a group with `train_n1000` |

Nothing further is needed from Suraj. B's dev evaluations used his own evaluator with a Person 3-compatible contract; Person 3 should re-evaluate the B adapters alongside C0 and C1 under the frozen contract for the final comparison (section 8).

## 3. What exists and is reused

| Asset | Location | Use |
| --- | --- | --- |
| Selection bank (10,000 rows) | `data/manifests/selection_bank.jsonl` | Source of rows for B's IDs |
| Protection scan | `protection.protect` | Reports dispositions of B's rows; no drops |
| Token verification and freeze | `data.prepare` | Explicit-universe mode added |
| Reference scoring and per-row cache | `reference_scoring.score_reference` | Rescored on the new bank |
| Allocator | `allocation.allocate` | Quotas from B's parent counts |
| Question-only prompt rendering | `Person2Adapter.render_question` | Request export |
| C1 formula seam | `selection.combined_priorities` | C1 utility |
| Trainer wrapper, reconciliation, smoke gate | `continuation.train` | Unchanged except whitelists |
| Prediction schema | `evaluation.validate_predictions`, `STATUSES` | Ingest validation |
| SOL chain | `scripts/sol/submit_pipeline.sh` and `.sbatch` files | One chain per config, profile and seed |

## 4. Code changes that touch fingerprinted files

`stage_fingerprint` hashes `selection.py`, `allocation.py`, `continuation.py`, `data.py` and others. Phase 0 changes `data.py` (explicit universe) and `selection.py` (`stratum`), so every existing artifact root is invalidated by design. That is acceptable because Phase 0 regenerates all of them. No pinning is needed.

## 5. Configuration additions

Added to all four explicit configs. New keys inside `data` and `allocation` enter the prepare and select fingerprints, which is intended.

```json
"data": {
  "...": "existing keys",
  "universe_manifest": "../../b_inputs/.../shared/mixed_candidate_bank.jsonl",
  "universe_sha256": "806dd78f38b37774391efc37d98318f0610e95dbeb9d53f810b54a03baf435ae",
  "universe_origin": "method-B main mixed bank, 6,409 rows",
  "length_bins": "../../b_inputs/.../shared/parent_length_bins.json",
  "length_bins_sha256": "024b651bc18084c715b0b77a1db78c82e9e20040dc5ec671e7597a2604dd51e6",
  "candidate_count": 6409
},
"allocation": {
  "...": "existing keys",
  "b_allocator_sha256": "<sha of the vendored allocator.py>",
  "quota_seed": 1729, "swap_window": 24, "max_priority_swaps": 8,
  "stratum_tolerance": 0.05,
  "b_selected_manifest": "../../b_inputs/.../mixed/budget_1x/1x/train_B.jsonl",
  "b_selected_sha256": "378316a7e9485ba0f0e089f674aadf00efa3c2b3dc8c174911f089a6d0fb5579"
},
"generation": {
  "contract_sha256": "<sha of data/inputs/evaluation/handoff/eval_contract.json>",
  "prompt_sha256": "22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db",
  "precision": "fp32", "decoding": "greedy", "max_new_tokens": 1024,
  "eos_token_id": 151643, "add_special_tokens": false,
  "parser_version": "p3-parser-v6", "executor": "person3-external"
},
"selection": { "mistake_bonus": 1.0 }
```

Continuation seed: `training.seed` is overridden per run with a new `--training-seed` CLI option on `train`, `finalize`, `evaluate` and `compare`, so one config serves three seeds. Because the `training` section is in the train fingerprint, each seed gets its own artifact root, and `method_definition` already excludes the seed.

## 6. Phase 0 tasks: rerun C0 on B's bank

### Task 0.1 Explicit universe in `prepare`

- When `data.universe_manifest` is set, `prepare` reads B's bank JSONL, verifies its SHA256, and freezes exactly those rows in row-ID order. No sampling. `candidate_count` must equal the row count.
- Every field of every row except `length_bin` must equal the selection-bank row with the same ID, so the pool is byte-identical in content to what B used and still traceable to Person 1's release. A differing field, an ID absent from the bank, or a duplicate is a hard stop.
- The protection scan runs and writes `protection/universe_dispositions.json`: for each of B's rows, eligible, excluded or quarantined under our rules, with reasons. Rows are not dropped. Counts are reported in the preparation summary and the handoff.
- Token verification of all 6,409 rows and the replay anchor runs as today.
- Tests: an ID absent from the bank or duplicated raises; a changed non-bin field raises; the frozen pool equals the file exactly; the dispositions report is written and the pool size is unchanged by it.

### Task 0.2 B's strata

- The bank already carries B's `length_bin`, so `selection.stratum` needs no change: `dataset_id|length_bin` is B's stratum.
- `prepare` recomputes each row's bin from `data.length_bins` (B's `parent_length_bins.json`, SHA pinned) with `bisect_left` on `supervised_tokens` against the source's cut points, and requires it to equal the stored `length_bin` on every bank row and every parent row. The parent rows are re-binned the same way for the replay anchor and the per-stratum counts; the plan expects exactly 50 per stratum and stops on anything else.
- Tests: cut-point values land in the right-closed interval; a fixture with one wrong stored bin raises; parent counts of 50 per stratum are reproduced from a fixture.

### Task 0.2b Vendor B's allocator

- Copy `allocator.py` from the bundle to `capstone_person4/b_allocator.py` unchanged except for the module docstring, record its SHA256 in the config as `allocation.b_allocator_sha256`, and add it to `STAGE_SOURCES["select"]`. A hash mismatch at import is a hard stop.
- `continuation.select` calls `b_allocator.allocate(rows, priorities, parent_counts, target, tolerance, quota_seed, swap_window, max_priority_swaps)` with B's recorded settings (`quota_seed` 1729, `swap_window` 24, `max_priority_swaps` 8, tolerance 0.005, per-stratum 0.05). Priorities are ascending, so C0 passes `1 - fit` and C1 passes `1 / utility`, each with our tie key appended as the deterministic secondary key; B's own random priority is reproduced by its `random_priorities` with seed 42 for the pair report.
- Our `allocation.allocate` stays in the package for the dry run and the identity tests, and is no longer used for real selection.
- Tests: calling the vendored allocator with B's random priorities on the real bank reproduces B's 1,000 selected IDs and 317,506 tokens exactly (this is the one real-data test in the suite and the proof the vendoring is faithful); bonus 0 reproduces C0 through the same function; a changed allocator file fails the hash check.

### Task 0.3 Reference scoring on the new bank

- `score-reference --smoke` then full, per parent. Four GPU jobs. The per-row cache is keyed by pool hash, so rows shared with the old 5,000 are rescored; that is a forward pass only and cheaper than special-casing.

### Task 0.4 Selection and B cross-check

- `select --method C0` and `select --method B` per parent, with B's quotas. The B manifest is only for the pair budget report.
- New `check-b-manifest` CLI: verifies every row in Suraj's manifest is in the universe, recomputes its supervised tokens under our encoder, compares the total to 317,506 and to the target tolerance, and records overlap with the C0 selection. Output goes into the handoff. A token mismatch is reported, not corrected.

### Task 0.5 C0 training, three seeds

- `train --method C0 --smoke` once per parent, then `train --method C0 --attempt <name> --training-seed S` for S in 42, 137, 2026. Twelve full runs. The smoke proof is per fingerprint and parent, so one smoke per parent covers all three seeds only if the smoke itself is run at each seed's fingerprint; simplest is one smoke per seed, which is small.
- GPU authorization required before submission.

### Task 0.6 Recipe record and handoff

- The recipe comparison is already done (section 2, item 5): identical. `continuation.train` additionally records quantization, clip and weight decay from the trainer in `run_config.json` so the match is machine-checkable, and `finalize` copies B's receipt fields next to ours in the handoff.
- `finalize` per parent and seed, `C0_EVALUATION_HANDOFF.md` regenerated for the twelve runs.

## 7. Phase 1 tasks: C1

### Task 1.1 Request export (`export-generation-request`)

CLI: `python -m capstone_person4 export-generation-request --config CFG --profile P`

- Reads the frozen universe (Task 0.1). Does not need reference scores, so it can run as soon as Task 0.1 is done and Person 3 can generate while C0 rescoring and training proceed.
- Writes `generation/request/<profile>-seed<parentseed>/questions.jsonl` with `problem_id` (= `row_id`) and `problem` only, and `request.json`: interface `person4-generation-request-v1`, parent adapter path and SHA, model, revision, contract SHA, prompt SHA, decoding settings, `candidate_sha256`, `generation_key`, row count, questions SHA. A request whose adapter SHA does not match the config's parent is refused.
- Test: no `completion`, `final_answer`, `answer` or `solution` field and no reference text substring in the exported file.

### Task 1.2 Ingest (`ingest-generation`)

CLI: `python -m capstone_person4 ingest-generation --config CFG --profile P --predictions FILE --report FILE`

- Validates with the `validate_predictions` rules: exact coverage once each, status in the six values, boolean `correct` consistent with status, required fields present.
- Verifies Person 3's report against `request.json`: adapter SHA, prompt SHA, contract SHA, parser version, decoding settings, model, revision. Any mismatch stops; this is what keeps a 0.5B file out of a 1.5B config and seed 17 outcomes out of seed 123.
- Rejects any `execution_error` row with the IDs to retry. Partial coverage is never accepted.
- Writes `generation/outcomes.jsonl` and `generation/coverage.json` keyed by `generation_key`. `mistake = (status == "incorrect")`; everything else is `false` with its status retained.
- `load_generation(cfg)` re-verifies hashes and coverage before use.
- Tests: missing, duplicate, unknown status, execution error, wrong adapter SHA, wrong parser version each raise; a valid fixture round-trips; changing `parser_version` or `prompt_sha256` changes `generation_key`.

### Task 1.3 Diagnostic gate (`diagnose` extension)

Adds a `correctness` section when outcomes exist: status counts by stratum; mistake rate per fit quartile; confirmed mistakes with `fit >= 0.5`; C0 versus C1 overlap at the budget (both through `allocate`); degeneracy flags `mistake_share < 0.05`, `> 0.95`, `overlap_fraction >= 0.95`. A flag sets `proceed_to_training: false` with the reason. Decided per parent. Tests: fixtures for each flag and a mixed case with none.

### Task 1.4 C1 selection (`select --method C1`)

`load_scores`, `load_generation`, `fit_priorities`, `combined_priorities(bonus=cfg["selection"]["mistake_bonus"])`, `allocate` with B's quotas. Report records `version: "mistake-fit-v1"`, `generation_key`, bonus, mistakes selected, C0 overlap. Tests: bonus 0 reproduces C0's exact ordered IDs; all-false and all-true vectors reproduce C0; repeated runs are byte-identical; a mixed vector changes the selection and the report counts its mistakes.

### Task 1.5 C1 training

`train --method C1` at seeds 42, 137, 2026 per parent. Twelve runs. SOL: `C1` added to the arm handling in `submit_pipeline.sh`, `select.sbatch` and `train.sbatch`. GPU authorization required.

### Task 1.6 Handoff and comparison

- `finalize --c1-run` binds C1 to the C0 run with the same parent and seed; `evaluation_handoff.json` gains `arms.C1`. `C1_EVALUATION_HANDOFF.md` lists the twelve adapters and checksums.
- `compare` accepts `{parent, B, C0, C1}` with B optional, matches arms by continuation seed as it does today, reports `C1_minus_C0` as the primary contrast plus the others, and summarises the three seeds within a parent as training-seed SD, the same statistic B reports.

## 8. Execution sequence

| Step | Who | Needs | Output |
| --- | --- | --- | --- |
| Receive and verify B's bundle | Person 4 | | Done 8 November 2026; see section 2 |
| Tasks 0.1, 0.2, 0.2b, 0.4 (check), 1.1 to 1.4, 1.6 implemented; existing and new tests pass | Person 4 | CPU | Code on the branch |
| Prepare the universe, export four requests | Person 4 | CPU | Frozen bank, request folders |
| Generate and grade, four runs of 6,409 rows | Person 3 | A100, see cost table | Predictions and report per parent |
| Score reference on the bank, four jobs | Person 4 | SOL GPU, authorization | Scores per parent |
| Select C0 and B, check B's manifest | Person 4 | CPU | Manifests, pair report, cross-check |
| Train C0, twelve runs, finalize | Person 4 | SOL GPU, authorization | C0 adapters and handoffs |
| Ingest, diagnose, per parent | Person 4 | CPU | Outcomes, flags |
| Read flags together | Team | | Go or stop per parent |
| Select C1, train C1, twelve runs, finalize | Person 4 | SOL GPU, authorization | C1 adapters and handoffs |
| Evaluate parent, C0, C1 on dev | Person 3 | A100 | Predictions |
| Compare, per parent | Person 4 | CPU | Four comparison reports |

Projected Person 3 generation cost, from their dev-set ledger rates (A100, fp32):

| Parent | Rows | Approximate A100 hours |
| --- | ---: | ---: |
| 0.5B seed 17 | 6,409 | 2 to 2.5 |
| 0.5B seed 123 | 6,409 | 2 to 2.5 |
| 1.5B seed 17 | 6,409 | 2.5 to 3.5 |
| 1.5B seed 123 | 6,409 | 2.5 to 3.5 |
| Total | 25,636 | 9 to 12 |

Projections, not measurements; Numina rows are longer than dev MATH.

SOL training: 24 full runs (12 C0, 12 C1) plus smokes. The existing C0 runs took 123 to 124 optimizer updates each; use their recorded wall times to size the requests.

Order within the phase: run 0.5B seed 17 end to end, including `compare`, before submitting the other parents' chains, so a stage defect is found once.

## 9. Phase 2: E (mistake-only ablation)

Deferred until C1 dev results exist. `select --method E`: utility `w` with the `random_priorities` tie key; same allocator and quotas; version `mistake-only-v1`. `train --method E` at the same three seeds, `finalize --e-run`, `compare` with `C1_minus_E` and `E_minus_C0`. Tests: all-false mistakes make E identical to B; a mixed vector ranks every mistake above every non-mistake within each stratum; ties are deterministic. Trained only for parents where C1 beats C0 on dev.

## 10. What this plan cannot claim

- B was trained with Suraj's wrapper around the same Person 2 trainer family; the recipe, quantization and environment match ours. His dev accuracies come from his own evaluator run, so the final B-versus-C0-versus-C1 comparison needs Person 3 to evaluate B's adapters under the frozen contract too.
- Our protection scan may flag rows B used. They stay in the universe by decision; the counts are reported.
- Two model sizes are two experiments. Four parents give two replicates of parent variation per size; three seeds give training-seed SD per parent. Neither is a confidence interval.
- Without E, a C1 gain cannot be attributed to the fit term versus mistake targeting.
- One greedy generation under one decoding policy is an observation, not a mastery estimate.

## 11. Later phases, not planned here

`C0_band`/`C1_band`, cross-model score transfer, and an extra held-out benchmark for generalization.

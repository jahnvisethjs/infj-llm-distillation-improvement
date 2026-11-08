# C0 rerun on Method B's bank: what changes, what reruns

Date: 8 November 2026 (America/Phoenix)
Branch: `person4/c0-reference-fit`
Status (8 November 2026, evening): **sections 2.1 to 2.10 are implemented on the branch, uncommitted.** 76 CPU tests pass under the project interpreter (one visible skip: the solver test, which passes with the other seven in an OR-Tools 9.14.6206 environment). Local evidence: the file-only audit for `0.5B/seed17` passes with the new `universe` check; the protection scan reran under the new prepare fingerprint (`outputs/person4/v4/0.5B/f1154634dcfd94d1/protection`, 9,241 eligible / 54 excluded / 705 quarantined of 10,000, identical to the earlier scan); `explicit_universe` on B's bank gives 5,812 eligible / 572 quarantined / 25 excluded by our matcher (921 / 78 / 1 among B's 1,000 selected rows), no row dropped; the re-binned parent is exactly 50 rows in each of the 20 strata. Token re-encoding (`prepare`), scoring, selection on Sol and all training remain to run. Section 3 is the rerun list.

## 1. Target state

| Setting | Current C0 | Rerun |
| --- | --- | --- |
| Candidate universe | 5,000 rows frozen by `freeze_stratified` from the protected selection bank, seed 42 | B's `mixed_candidate_bank.jsonl`, 6,409 rows, taken as-is |
| Strata | `dataset_id` × Person 1 `length_bin` | `dataset_id` × B's `length_bin` (per-source parent deciles; the bank already carries it; 31 rows differ from Person 1's) |
| Parent proportions | Counter of parent rows under Person 1 bins | Parent rows re-binned under B's cut points: exactly 50 per stratum |
| Allocator | `allocation.allocate` (ours) | B's `allocate()` vendored, same settings as B: quota seed 1729, tie seed 42, count adjustment 0.05, swap window 24, max swaps 8 |
| Token target | 318,845 ± 0.5% (replay anchor) | unchanged |
| Continuation seed | 42 | 42, 137, 2026 |
| Parents | four | four, unchanged |
| Recipe, trainer, environment | Person 2 trainer via `continuation_source` | unchanged; verified identical to B's |

## 2. Where the C0 implementation needs updates

Ordered by the stage chain. "Fingerprinted" means the file is in `STAGE_SOURCES`, so editing it changes artifact identities. Every stage reruns anyway (section 3), so that is acceptable here and no pinning is needed.

### 2.1 Configuration (`configs/c0.v4.*.json`, four files) and generator (`v4_config.py`)

| Key | Change | Why |
| --- | --- | --- |
| `data.candidate_manifest` | point at B's bank JSONL | `protect` reads this file; scanning B's rows directly gives the dispositions report for exactly the universe used |
| `data.universe_mode` (new) | `"explicit"` | tells `prepare_v4` to skip `freeze_stratified` |
| `data.universe_sha256` (new) | `806dd78f…` | hard check on the file |
| `data.candidate_count` | 6409 | must equal the bank size |
| `data.length_bins`, `data.length_bins_sha256` (new) | B's `parent_length_bins.json`, `024b651b…` | re-bin check and parent re-binning |
| `allocation.allocator` (new) | `"method-b-shared-v1"` | selects the vendored allocator |
| `allocation.b_allocator_sha256` (new) | SHA of `b_allocator.py` | import-time check |
| `allocation.quota_seed`, `tie_seed`, `row_count_adjustment_fraction`, `swap_window`, `max_swaps` (new) | 1729, 42, 0.05, 24, 8 | B's recorded settings |
| `allocation.boundary_window`, `max_swaps`, `max_row_counts` | left for the dry run only | our allocator no longer used for real selection |
| `allocation.b_selected_manifest`, `b_selected_sha256` (new) | B's `train_B.jsonl`, `378316a7…` | reproduction check in `select` |
| `training.seed` | stays 42 as default | overridden per run by `--training-seed` |

`v4_config.py` gets the same fields and `configure_sol.sh` a `C0_B_BANK` variable pointing at Suraj's v3 folder on Sol (readable from your account; fingerprints strip paths, so local and Sol identities match).

### 2.2 `data.py` (fingerprinted: prepare and everything after)

`prepare_v4`, explicit-universe branch:

1. `protect(cfg)` runs on the bank as today and writes its eligible, quarantined and excluded lists. New: `load_protection` variant that returns all three, not only eligible, because no row is dropped. Counts go into `prepared/universe_dispositions.json` and the preparation report.
2. The universe is the full bank in row-ID order. Hard stops: SHA mismatch, count ≠ `candidate_count`, duplicate ID, ID absent from `selection_bank.jsonl`, or any field other than `length_bin` differing from the selection-bank row.
3. New helper `rebin(rows, bins)`: `bisect_left` of `supervised_tokens` against the source's cut points (fractions). Applied to the bank as a check (must equal stored `length_bin`, zero tolerance) and to the parent rows to produce their B bins. Parent must come out at 50 per stratum.
4. Token verification of all 6,409 rows and the parent anchor: unchanged code.
5. Report adds `universe_origin`, `universe_sha256`, `length_bins_sha256`, dispositions counts, parent stratum counts.

`freeze_stratified` stays for the legacy path and its tests.

### 2.3 `selection.py` (fingerprinted: select and after)

No change. `stratum()` already reads `length_bin`, which is B's bin on bank rows. `fit_priorities` unchanged. `combined_priorities` untouched until C1.

### 2.4 New `b_allocator.py` (add to `STAGE_SOURCES["select"]`)

**Open gap found by the independent fairness audit on 8 November 2026.** B's 1,000-row manifest was not produced by `runtime/allocator.py:allocate()`. `selection_audit.json` records `selection_implementation: "document Appendix A CP-SAT"`, `ortools_version 9.14.6206`, `status OPTIMAL`, `objective 912175166`, `target_rows 1000`, per-stratum `token_bounds` = parent tokens ± 5%. The objective equals the sum of `utility_integer` from `candidate_priorities.csv` over the selected rows exactly, and all 20 strata sit inside their bounds. Running the bundled `allocate()` with B's random priorities and settings on the bank gives 1,002 rows, 319,096 tokens and only 181 of B's 1,000 IDs, so the reproduction hard stop in 2.5 would fire. The CP-SAT selector code is not in the bundle.

What B's selector actually enforced, from the audit record: exactly 1,000 rows; exactly 50 rows per stratum; per-stratum supervised tokens within ± 5% of the parent's tokens in that stratum; global tokens within [317,251, 320,439]; maximise total integer utility; a secondary tie-break solve. `allocate()` has no fixed row count and no per-stratum token bound, which is why it lands elsewhere.

Decision (8 November 2026): option 1. **Done and reproduced the same day.** Suraj's code arrived as `b_code/method_b_document_v3_code/` (from `/scratch/sshind38/capstone_approach_b/method_b_document_v3_code/`); the selector is `select_b.py`, SHA256 `5c9241231efc449fccbe464ce3198b87af20de8041aa699149f9c702f83bf7de`. Running `select_b.solve(parent_rebinned_reference, mixed_candidate_bank, 1)` under OR-Tools 9.14.6206 (isolated scratch virtualenv, Python 3.13.9) reproduced B exactly: 1,000 rows, 317,506 tokens, objective 912,175,166, identical ID set, identical training order after the `42|training-order|row_id` hash sort, byte-identical row content to `train_B.jsonl`; `parent_edges` also reproduces every stored `length_bin`. Status OPTIMAL for both the primary and the tie-break solve; 13.6 s wall locally. The selector is deterministic by construction (one search worker, fixed seed, deterministic time limit), which is why this works across machines.

What `select_b.solve` enforces: exactly the parent's row count per stratum (50 each at 1×), per-stratum supervised tokens within ± 5% of the parent's tokens in that stratum, global tokens within ± 0.5% of 318,845, maximise the sum of integer utilities; then a second solve that holds the objective fixed and minimises the sum of per-row tie hashes (`seed|tie|row_id`). The utility is hard-coded as the random hash `floor(1e6 · first32(SHA256("42|random|row_id")) / 2^32 + 0.5)`.

Vendoring plan, replacing the `allocate()` text below:

- Copy `select_b.py` into `capstone_person4/b_select.py` with one change: `solve(..., utilities=None)` where `None` keeps his random utility and a dict `row_id → int` supplies a selector's own. Record both hashes (original file and patched file) in the config; the reproduction test runs the patched module with `utilities=None` and must still give the 1,000 IDs and objective above, which proves the patch is inert for B.
- C0 utility: `round(fit · 1_000_000)`, the same integer range as B's; tied losses share a fit and therefore a utility, and the tie-break solve breaks them with his seed-42 tie hash. C1 later: `round(fit · (1 + λ·w) · 1_000_000 / (1 + λ))`, so the range stays within 0..1,000,000. B through our pipeline: `utilities=None`.
- `ortools==9.14.6206` goes into `requirements-person4.txt`, and `solve` already refuses any other version. The 1.5B tokenizer guard and everything else are unaffected: selection is CPU-only.
- Our `allocation.allocate` stays for the dry run and the lambda-zero identity tests.

The options are kept below for the record.

Resolution options, to be chosen before any code is written:

1. Obtain Suraj's CP-SAT selector (the "Appendix A" script, with the utility scaling and the tie-break solve) and run C0 through it with `1 − fit` scaled to the same integer range. The reproduction test then stands as written and "same allocator" is literally true. Preferred.
2. Re-implement the CP-SAT model in the package from the recorded constraints with `ortools` pinned at 9.14.6206, and prove it by reproducing B's objective and IDs. Same scientific result as option 1 if the reproduction succeeds; more work, and tie-breaking may not reproduce without his code.
3. Keep `allocate()` or our own allocator for C0 and report that B used a different allocator. Weakest; the reproduction hard stop must be dropped.

Until this is decided, the text below describes the `allocate()` interface and is retained only for option 3. Interface differences from ours that `select` must bridge:

| Ours | B's |
| --- | --- |
| rows carry `utility` (higher is better) and `tie_key` | `priorities` dict, ascending (lower is better); ties by `stable_key("selector-tie-v1", 42, row_id)` inside the allocator |
| `proportions` keyed `"orca|0"` | `parent_counts` keyed `("orca", 0)` tuples via `cell(row)` |
| returns rows in stratum/rank order with `priority_rank` | returns rows sorted by `row_id`; rank must be added by us from the per-stratum ordering |
| report: `target_tokens`, `actual_tokens`, `rows`, `quotas`, `relative_error`, `tolerance`, `swaps` | report: `actual_rows`, `actual_tokens`, `target_tokens`, `allowed_integer_token_range`, `relative_token_difference`, `attempts`, `swap_count` |
| row-count search over all counts, prefers zero-swap prefix, min excess-rank objective | row-count search from the token-mean estimate outward (radius 5%), first count that lands in tolerance after ≤ 8 swaps |

The last row is why our C0 selected 980 to 985 rows and B selected exactly 1,000.

### 2.5 `continuation.py` (fingerprinted: select, train and after)

- `budget_and_coverage`: re-bin parent rows with `rebin` before counting proportions; replay target computation unchanged. Return `parent_counts` as tuples for the vendored allocator.
- `select`:
  - C0 priority = `1 - fit` (ascending). B priority = B's own `random_priorities(rows, 42)` from the vendored module, so our B manifest is Suraj's.
  - Call `b_allocator.allocate(...)` with the config settings. Translate its report into the fields `load_selection`, `train` and `finalize` already read (`target_tokens`, `actual_tokens`, `tolerance`, `rows`, `manifest_sha256`, `selected_data_hash`, `score_fingerprint`, `stage_fingerprint`), and keep B's full report under `allocator_report`.
  - After selecting B: compare to `allocation.b_selected_manifest`. Same 1,000 IDs and 317,506 tokens is the proof that the vendoring is faithful; a difference is a hard stop, because "same allocator" would then be false.
  - `version` becomes `"reference-fit-v2-b-allocator"`.
- `train`: no logic change. Record `quantization`, `gradient_clip` and `weight_decay` from the trainer in `run_config.json` so the recipe match with B is machine-checkable.
- Smoke proof: today a full run requires a smoke whose `stage_fingerprint` equals the train fingerprint, which includes `training.seed`. With three seeds that means three smokes per parent. Change the proof comparison to a seed-agnostic fingerprint (train fingerprint computed with `training.seed` removed), so one smoke per parent covers all three seeds. The smoke proves trainer wiring, not the seed.

### 2.6 `__main__.py`

- `--training-seed INT` on `train`, `finalize`, `evaluate`, `compare`; sets `cfg["training"]["seed"]` after `load_config`. The `training` section is in the train fingerprint, so each seed gets its own artifact root automatically, and `method_definition` already excludes the seed.
- No new command is needed for the B cross-check; it lives inside `select`.

### 2.7 `handoff.py`

- Accept the seed via config (already does: it compares `run_config.seed` to `cfg["training"]["seed"]`). Nothing else changes for a C0-only finalize.
- `external_B` entry points at Suraj's run directory and receipt for the same parent and seed, with his adapter SHA from `ADAPTER_SHA256SUMS.txt`, instead of only our B manifest path.

### 2.8 `audit.py`

Add a `universe` check to the `prepare` and `select` stages: bank file exists and matches `universe_sha256`, length-bins file matches its SHA, vendored allocator matches `b_allocator_sha256`. Three `checked_file` calls.

### 2.9 SOL chain (`scripts/sol/`)

- `batch_stage.sh train`: fifth argument `SEED` → `--training-seed`. `finalize` stage: loop over `C0_SEEDS` inside one job.
- `submit_pipeline.sh`: for each arm, one train job per seed in `C0_SEEDS` (default `42 137 2026`), all `afterok` the single train-smoke; one finalize job after all of them. Chain per config: prepare, score-smoke, score, select, train-smoke, C0 × 3, finalize = 9 jobs, under the class QoS limit of 10 active submissions. With `C0_ARMS="B C0"` it would be 12, so B stays external.
- `configure_sol.sh`: `C0_B_BANK` → config fields in 2.1.
- `train.sbatch` time cap can drop from 4 h to 1 h: the previous full runs took 6 to 8 minutes each (section 3).

### 2.10 Tests (`tests/person4/`)

Update: fixtures in `test_pipeline.py` and `test_v4.py` that exercise `prepare_v4`, `budget_and_coverage` and `select` through our allocator; `test_sbatch.py` job-count and argument checks.

Add: explicit-universe hard stops (SHA, count, duplicate, missing ID, changed field); `rebin` boundary cases and the parent 50-per-stratum check; vendored allocator reproduces B's 1,000 IDs and 317,506 tokens on the real bank (skipped with a visible reason if `b_inputs/` is absent); allocator hash mismatch stops; `--training-seed` yields distinct train roots and the seed-agnostic smoke proof is accepted across seeds; `submit_pipeline.sh` emits 9 jobs.

The 68 existing tests must still pass; the dry run keeps using our allocator.

## 3. What must be rerun

Per parent (four parents), in chain order. Previous durations are from the recorded Sol runtime reports on the 5,000-row pool.

| Stage | Compute | Previous time | Expected now |
| --- | --- | --- | --- |
| `audit --target inputs` | CPU, local | seconds | seconds |
| `protect` on B's bank | CPU | minutes (10,000 rows scanned) | minutes |
| `prepare` | CPU, pinned env (re-encodes 6,409 + 1,000 rows) | minutes | minutes |
| `score-reference --smoke` | A100 | 30 to 50 s wall | same |
| `score-reference` full | A100 | 5,000 rows in 9 to 11 min (3,000 to 3,750 supervised tokens/s) | 6,409 rows, 12 to 15 min |
| `diagnose`, `select` B and C0 | CPU | seconds | seconds |
| `train --smoke` | A100 | ~1 min | same, once per parent |
| `train --method C0` | A100 | 6 to 8 min per run, 123 to 124 updates | 3 runs per parent, 20 to 25 min |
| `finalize` | CPU | seconds | seconds, three handoffs per parent |

GPU total for all four parents: roughly 2.5 to 3 A100 hours, plus queue time. The per-row reference cache is keyed by the pool hash, so all 6,409 rows are rescored even though 3,201 of them were in the old pool; at 12 minutes per parent that is not worth special-casing.

What is not rerun: Person 1's release, the four parents, Person 2's trainer, the protection matcher, the reference-scoring code, the 1.5B tokenizer guard.

What becomes a record only: every artifact under `results/` and `outputs/person4/v4` for the 5,000-row pool, the four `pilot01` C0 adapters on Sol, and `C0_EVALUATION_HANDOFF.md`. They stay on disk under their old fingerprint roots; Person 3 should not evaluate the old adapters as the comparison arms. A one-line note at the top of the old handoff says so.

## 4. Order of work

1. Config fields and `v4_config.py` (2.1), vendored allocator and its real-data reproduction test (2.4): this test is the first thing to run, because if B's manifest does not reproduce, the rest of the plan changes.
2. `data.py` explicit universe and `rebin` (2.2), `budget_and_coverage` and `select` (2.5), audit check (2.8), tests.
3. `--training-seed`, seed-agnostic smoke proof, handoff and SOL chain (2.5 to 2.9), tests.
4. Local: audit, protect, prepare for `0.5B seed17`; confirm the dispositions report and the 50-per-stratum parent counts.
5. Sol: one full chain for `0.5B seed17` end to end with GPU authorization, then the other three.

C1 work starts only after step 5 produces reconciled C0 handoffs on the new bank.

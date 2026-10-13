# Progress review and completion plan

> Delivered: **DATA_READY_FOR_TRAINER**, immutable `releases/person1_v3_20261012_ready/`. All **1,096,250 source rows** processed; **40K training / 10K fresh selection / 1K loss** plus nested prefixes exported; **87 tests** and independent verification of **447 artifacts / 11 manifests** passed. Person 1 current-release work is complete. Person 2 actual-label/training evidence and Person 3 evaluation freezes remain downstream inputs. [Current folder guide and remaining responsibilities](docs/CODEBASE_GUIDE_AND_STATUS.md). Earlier sections below are historical planning records. Git remains paused.

For the organized folder map, delivered data quantities and current implementation-phase assessment, see [CODEBASE_GUIDE_AND_STATUS.md](docs/CODEBASE_GUIDE_AND_STATUS.md). Tests now run through discovery under `capstone_person1/tests/`; historical commands below describe earlier milestones.

> This file records earlier implementation milestones. The supplied `Person1_Codex_Implementation_Plan.md` adds requirements and changes some ownership/readiness assumptions. Use [CODEBASE_ALIGNMENT_PLAN.md](CODEBASE_ALIGNMENT_PLAN.md) for the current implementation sequence and gaps. In particular, Person 1 can finish a `DATA_READY_FOR_TRAINER` release before Person 2 supplies actual training labels or a trained parent.

Updated October 12, 2026 (America/Phoenix): the first alignment code batch is implemented. See [FIRST_ALIGNMENT_BATCH.md](docs/milestones/FIRST_ALIGNMENT_BATCH.md) for the v2 notebook, corrected target safeguards, verified shared label encoder, 55 passing tests and the newer 5,000-row whole-snapshot diagnostic. The final corrected data release remains incomplete. [IMPLEMENTATION_STATUS.json](evidence/IMPLEMENTATION_STATUS.json) records the current verification and resume evidence.

The subsequent [Person 3 input-interface batch](docs/milestones/BENCHMARK_INTERFACE_BATCH.md) is also implemented. The current suite passes **63 tests**. The existing 1,819 reserved questions are imported with provenance and bounded overlap-check scope; development/fast-development coverage remains pending. The latest successful diagnostic is `runs/full_run_v2_20261012T181641Z_cfe8ce44`, with the same 100-row content hashes and reference budget as the preceding diagnostic. Next comes instance-level matching and final group-map export.

The sections below preserve the October 11 milestone review. They distinguish implemented code, verified behavior, completed data artifacts and external team evidence; their diagnostic counts are historical.

## Previous milestone — October 11

The next implementation milestone is complete: the approval gate is enforced across training and fresh pools, and the local rebuild now streams source batches with checkpoints and a disk-backed metadata catalog. **The final research data handoff is still incomplete.** Two smoke runs have exported draft manifests, updated reports and `handoff_readiness.json`; both correctly report `research_ready: false`.

The 16 earlier attempts remain historical incomplete runs. The old whole-source mapping path stopped at 174,755 of 200,035 Orca rows and is no longer the runner entrypoint. The replacement does not capture full datasets in mapper closures or materialize the full text corpus as pandas. Full-scale performance has not yet been established; no million-row rebuild was launched for this milestone.

### Previous milestone evidence

| Verification | Outcome |
|---|---|
| Regression and integration suite | 32 tests passed |
| Notebook syntax and embedded controls | Passed; stale execution outputs remain cleared |
| Historical tokenizer compatibility | All 1,000 preprocessing counts still match |
| 1,000-row smoke (500/source) | Completed; about 11.4 seconds after setup; observed peak RSS 199.4 MiB |
| 5,000-row smoke (2,500/source) | Completed; about 19.6 seconds after setup; observed peak RSS 214.3 MiB |
| Resume of the 5,000-row smoke | All 5,000 canonical rows reused; about 5.4 seconds after setup; manifest hashes unchanged |
| Export read-back | Ordered nesting, IDs/groups, prompt/target rendering, 100-row pool source balance, lengths and artifact hashes passed |
| Missing review/team evidence | Both actual smoke handoffs remain blocked |

Smoke directories:

- `runs/full_run_v2_20261012T004452Z_ffd466bf` — 1,000 source rows.
- `runs/full_run_v2_20261012T004834Z_c75c286d` — 5,000 source rows and verified resume.

The UTC directory date is October 12; these runs were performed October 11 in America/Phoenix. The smaller smoke subsets are 32, 64 and 100 rows, **not replacement R1000/main-run manifests**. Times above exclude initial configuration/input hashing. `rebuild_metrics.json` records the latest execution, so the larger run's current metrics describe its verified resume. Initial-run timings were captured from the successful execution output.

## Verified work

Commands rerun successfully during this review:

```powershell
& 'C:\softwares\pythonAnaconda\python.exe' -m unittest capstone_person1.test_pipeline_controls -v
& 'C:\softwares\pythonAnaconda\python.exe' -u -m capstone_person1.validate_notebook
& 'C:\softwares\pythonAnaconda\python.exe' -m capstone_person1.validate_tokenizer_adapter
```

- All 32 regression/integration tests passed: extraction and formatting, mathematical scalar equivalence, saved-review enforcement against both pools, stale evidence/changed artifacts, corrected parent identity, interrupted checkpoint resume, changed input/control/shard rejection, and disk-backed exact grouping/catalog reuse.
- Notebook code parses; its embedded controls match the tested module; stale execution outputs are cleared.
- Actual stored Orca row 71484 now extracts `6`, row 75823 extracts `1`, and row 185528 is quarantined as a suspected multi-part question. The reported `2` versus `32` pattern is covered by a passing regression test.
- The pinned local tokenizer adapter reproduces all 1,000 historical R1000 preprocessing counts, including the historical total of 333,328. This verifies adapter compatibility, **not actual training labels or the corrected budget**.

## Status against the requested fixes

| Review issue | Implemented | Remaining to finish |
|---|---|---|
| Incorrect extraction and contradictory targets | Terminal-only extraction, cautious quarantine, validated final-marker formatting | Finish source re-extraction, regenerate targets, retokenize and export corrected manifests |
| Missing documented human review | Preserved review sheets, metadata/hash-based approval interface | Generate corrected sheets, complete independent reviews and enforce every finding across all exported pools |
| Missing development protection | Frozen development manifest validation and combined overlap-check path | Person 3 supplies both development families; execute checks and regenerate downstream data |
| Unvalidated duplicate/overlap thresholds | Explicit conservative lexical-variant policy, pair classification, independent missed-match sample | Review policy/threshold behavior and missed matches; apply reviewed removals, including below-threshold discovered duplicates |
| Weak label-count reconciliation | Nonempty, exact coverage, unique IDs and one-to-one matching | Person 2 supplies actual corrected R1000 label counts and training metadata; finalize B |
| Missing fresh continuation pool | Actual parent evidence validation, cross-version ID/hash/group protection, bounded parent near-check, full balanced target requirement | Confirm corrected parent checkpoint and actual manifest; run fresh-pool generation and validation |
| Misleading exclusion descriptions | Accurate flag names and nonexclusive-count documentation | Generate updated reports from completed corrected data |
| Completeness and mathematical correctness | Conservative answer-only/multi-part exclusions; separate review checks; correctness remains unknown | Audit generated targets independently; resolve findings without representing heuristics as mathematical certification |
| Diagram and answer-conflict checks | Expanded image/Asymptote/TikZ detection and safe exact rational equivalence | Supply/use Person 3's shared symbolic checker and review unresolved conflict groups |
| Reproducibility and handoff interface | Fixed repository SHAs, new run directories, preserved evidence, prompt-spec interface, source IDs and version fields | Exercise resume behavior, record verified license provenance, finalize the prompt/training contract and approve exact output hashes |

Original v1 and pilot artifacts are preserved and marked unapproved for the main runs. Team input templates and `FIXES_AND_HANDOFF.md` are present. Their existence does not satisfy the corresponding integration tasks.

## Ordered completion plan

### 1. Finish implementation review and gate tests

Owner: coding work for Person 1.

Status: **completed for this milestone**. Tests execute the notebook's actual final gate. Pair sheets now require `manual_policy_action`; reviewed duplicate instances/protected overlaps cannot survive merely because they scored below threshold. Approval binds prompt, debug, nested-training and fresh-manifest hashes. A parent manifest must match the agreed corrected R1000 export.

- Audit the final approval gate, not only isolated helper functions. Add integration checks for incomplete/stale approval evidence and reviewed findings that remain selected.
- Enforce reviewed exclusions against **both training and fresh pools**, including source-inspection findings identified by original row index and duplicate/evaluation pairs with different ID column names.
- Reject retained reviewed duplicate instances or protected overlaps even when the similarity score is below the configured threshold. Record an explicit resolution for conservative variant removals.
- Bind the fresh-manifest and prompt-specification hashes into final readiness evidence, alongside nested training manifest hashes.
- Require the main experiment's parent checkpoint to correspond to the corrected agreed R1000 manifest. An actual but historical v1 parent must not silently pass as a corrected parent.
- Exercise notebook-to-runner integration on small data, including canonical schema compatibility, prompt/completion formatting, source balancing, nesting, token boundaries and resume behavior.

Exit condition: meaningful end-to-end checks pass and a known incomplete or inconsistent handoff cannot become research-ready.

### 2. Make the rebuild bounded in memory and restartable

Owner: coding work for Person 1.

Status: **implemented and verified on 1,000/5,000 source rows and resume**. Sources are processed into committed Parquet shards; completed rows are reused only under matching source/control/options and shard hashes. Global compact metadata lives in SQLite; only the bounded candidate text is materialized. The default run is a smoke run. `--full` is explicit and has not been run. The configured memory guard is checked during stages; observed smoke memory does not prove full-scale memory behavior.

- Inspect intermediate caches and failed-run directories; mark them interrupted so they cannot be mistaken for successful runs. Preserve v1 and existing review evidence.
- Replace full-source row mapping in the local runner with projected-column Parquet batch reads and bounded Arrow writes. Avoid closures that capture the entire execution namespace, retain only compact metadata for global duplicate grouping, and materialize full text only for the bounded candidate universe.
- Save completed stages and source/input/control hashes. Resume only when they match; do not repeat finished million-row stages after a downstream failure.
- Run approximately 1,000 rows, then 5,000 rows, measuring peak memory and elapsed time. Fix any growth before expanding to the full corpus. Keep the tokenizer adapter check against the historical R1000 fixtures.

Exit condition: small and larger smoke runs complete with bounded memory, produce drafts and correctly fail the evidence gate when inputs are missing.

### 3. Freeze complete upstream inputs and review policy

Owners: Persons 1 and 3; independent reviewer for policy.

- Obtain `frozen_development.csv` and confirmed metadata covering GSM8K and MATH development families, plus the evaluator's shared equivalence checker.
- Reload original Orca/Numina fields at the configured SHAs for the final rebuild. The saved v1 canonical snapshots omit validity fields and may contain truncated Numina answer text; snapshot reconstruction can support debugging but cannot be the final source of truth.
- Confirm the diagram, worked-solution, multi-part and lexical-overlap policy. Treat the current conservative policy as a proposed default until reviewed.
- Freeze source revisions, environment/control versions, prompt specification and protection-input hashes in a new run.

Exit condition: complete frozen source/protection inputs and a documented policy are available. Development inputs added later require downstream regeneration under a new frozen configuration.

### 4. Rebuild and review corrected draft artifacts

Owners: Person 1 and independent reviewer.

- Re-canonicalize sources; produce an extraction-change ledger distinguishing changed representations, changed targets and quarantines. Changes are not a measured dataset-wide error rate.
- Run exact-equivalence grouping, bounded near checks and development/test protection; then regenerate completions and token counts.
- Export the balanced 40,000-row working pool, debug set, ordered nested subsets and source/length summaries with hashes and updated disposition counts.
- Generate and complete source, extraction, formatting, mathematical-completeness/correctness and duplicate/overlap reviews. Correct or quarantine findings and regenerate affected artifacts; preserve evidence from each version.
- Verify nesting, unique groups, source totals, context limits, target consistency, protection coverage and reproducibility of manifest hashes.

Exit condition: corrected draft manifests and completed review evidence exist for the same exact frozen run.

### 5. Complete training and parent integration

Owner: Person 2, with Person 1 reconciling/exporting.

- Confirm the prompt, EOS, masking and truncation contract using the versioned prompt specification.
- Verify actual `labels != -100` counts for every corrected R1000 row. Require exact coverage and matching manifest/verification hashes; only then finalize continuation budget B.
- Confirm or retrain the parent on the corrected agreed manifest and provide checkpoint identity, actual training manifest and actual seen groups.
- Export the full fresh continuation pool with parent/development/test disjointness, sufficient source-balanced candidates, length summaries and hashes.

Exit condition: corrected R1000 actual-label reconciliation and a confirmed corrected parent/fresh pool pass integration checks.

### 6. Approve the final handoff

Owners: the research team.

- Save review approval bound to the exact policy, run fingerprint and review-file hashes.
- Produce `handoff_readiness.json` covering the exact training/fresh manifest and prompt-spec hashes, protected development/test inputs and verified token budget.
- Release the comparative runs only when that record has `research_ready: true` and no unresolved blocker.

## Immediate next work

Move to step 3: obtain Person 3's frozen development questions and shared equivalence checker, agree the review policy, and stage complete original fields at the pinned repository SHAs. Then rebuild/review the corrected main manifests in a new frozen run. Person 2's label and corrected-parent confirmation follows those exports. No messages have been sent to teammates and no review entries have been fabricated.

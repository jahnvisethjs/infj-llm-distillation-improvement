# Folder guide and Person 1 completion assessment

Reviewed October 12, 2026 (America/Phoenix), against the [supplied implementation brief](reference/Person1_Codex_Implementation_Plan.md) and the user's ownership clarification. Person 1 supplies data and reference contracts; **Person 3 prepares its own evaluation data**.

The codebase is organized, and **63 tests pass**. The reference encoder is implemented and independently verified with the pinned tokenizer. Several other phases have working pieces, but **Person 1's final data release is incomplete**. Current state: `DIAGNOSTIC_ONLY`, not `DATA_READY_FOR_TRAINER`. No corrected final training, selection or loss bank has been released.

## What each folder contains

| Folder/location | Contents and use |
|---|---|
| Workspace root | Preserved original notebook, current v2 notebook, navigation README and the two project folders |
| `project_documents/` | Eight proposal/charter/rubric/historical-audit reference files; these do not establish current release readiness |
| `capstone_person1/` | Eight stable runtime modules, two public CLI entrypoints, package README and the two active planning documents |
| `capstone_person1/tests/` | Seven test modules containing 63 extraction, label, snapshot/resume, catalog, integration-gate and benchmark-interface tests |
| `capstone_person1/scripts/` | Tokenizer setup, notebook synchronization, notebook/tokenizer validation, diagnostic export verification and organization utilities |
| `capstone_person1/scripts/history/` | One-time preservation/notebook migrations and historical milestone recorders; inspect before intentionally rerunning |
| `capstone_person1/docs/contracts/` | Current reference-encoder usage and Person 3 frozen-question input contract |
| `capstone_person1/docs/milestones/` | Earlier fixes/handoff notes and the first alignment/benchmark-interface milestone reports; historical commands describe their original layout |
| `capstone_person1/docs/reference/` | Exact reference copy of the downloaded implementation brief; the downloaded original remains untouched |
| `capstone_person1/evidence/` | Current implementation status, earlier verification JSONs, tokenizer verification, sampled-case diagnostics and organization records |
| `capstone_person1/configs/` | Versioned data configuration and verified prompt/tokenizer/EOS/padding contract. Final shared length-bin edges are still absent |
| `capstone_person1/inputs_from_team/` | Actual reserved-test question CSV/metadata plus unsigned examples/templates for Person 2/3 inputs; a template is not supplied evidence |
| `capstone_person1/diagnostics/` | Standalone interface checks; currently the Person 3 manifest-import diagnostic |
| `capstone_person1/runs/` | 23 preserved run directories: historical full v1, interrupted attempts and completed/failed v2 diagnostics. No final approved v2 release exists |
| `capstone_person1/outputs/` | Earlier pilot canonical data, manifests and reports, marked unsuitable for main runs |
| `capstone_person1/preserved_inputs/` | Hash-bound copies of the actual pre-alignment local notebook and selected helpers. Recovery of the untouched original Colab notebook is still pending |
| `capstone_person1/logs/` | Historical rebuild log |
| `capstone_person1/cache/` | Cached pinned Hugging Face tokenizer assets. This is not a complete upstream source-data cache |
| `capstone_person1/vendor/` | Project-local `datasketch` dependency |
| `capstone_person1/runtime_deps/` | Project-local compatible Hugging Face Hub dependency used for actual-tokenizer verification |
| `__pycache__/` directories | Automatically generated Python bytecode; not authored source or release evidence |

Every saved run has its own `outputs/processed/`, `outputs/manifests/`, and `outputs/reports/` as applicable. Streaming diagnostics also have `checkpoints/`: committed canonical Parquet batches, source progress and SQLite catalog state. Incomplete attempts may lack later folders/artifacts; folder existence is not completion.

## Main runtime files

| File in `capstone_person1/` | Responsibility |
|---|---|
| `pipeline_controls.py` | Extraction/quality flags, review safeguards, safe scalar equivalence, token reconciliation and parent-evidence checks |
| `person1_data_contract.py` | Exact prompt/completion serialization, combined-text encoding, reference labels, EOS/padding and manifest validation |
| `benchmark_manifests.py` | Person 3 frozen-question import, provenance/role checks, full/fast nesting and protection-scope export |
| `configuration.py` | Configuration validation and notebook setting mapping |
| `snapshot_stream.py` | Bounded canonicalization of historical snapshots, fixed-seed sampling and resumable committed batches |
| `snapshot_catalog.py` | Disk-backed exact grouping/equivalence and bounded candidate text materialization |
| `tokenizer_adapter.py` | Lightweight offline adapter to the pinned tokenizer assets |
| `streaming_rebuild.py` | Diagnostic orchestration; currently executes notebook stages and retains numbered-cell dependencies |
| `rebuild_from_snapshot.py` | Public diagnostic rebuild entrypoint |
| `import_person3_manifests.py` | Public frozen-question import entrypoint |

## Completion against the implementation plan

The status below distinguishes implemented code from final release acceptance. It is not an effort percentage: a verified helper and a million-row reviewed release are very different amounts of work.

| Phase | Completion status | Delivered | Still needed from Person 1 |
|---|---|---|---|
| **0 — Preservation/reproducibility** | Mostly implemented; final release freeze pending | Original local notebook preserved, separate v2, immutable runs, configured source/tokenizer revisions, seeds, optional Drive/root controls, recorded hashes/environment | Complete release-level model/source/environment metadata and all used-code/input bindings; finish the shared stage/entrypoint cleanup. Distinguish local preserved input from the unrecovered original Colab source |
| **1 — Extraction/targets** | Code fixes verified; corpus regeneration pending | Terminal extraction, ambiguity/multi-answer/NaN checks, structured evidence, preserved correct final blocks, regression cases | Reprocess complete sources and affected cohorts; resolve uncertain extraction/supply findings; regenerate full targets, counts and downstream artifacts |
| **2 — Eligibility/review** | Partial | Honest unknown-correctness fields, diagram/completeness flags, quarantine diagnostics and saved-review safeguards | Approximately 100/source stratified target reviews plus risky/duplicate/overlap cases; save actual decisions, apply findings and resolve mandatory cases |
| **3 — Protected evaluation inputs** | Person 1 interface delivered; final release checks pending | Import/role/source/hash/nesting validation; known 1,819-question union; explicit pending coverage; bounded diagnostic checks | Apply the known/supplied protected union consistently to all final banks and export final scope. Import/check later Person 3 freezes when supplied; Person 1 need not build those evaluation datasets |
| **4 — Duplicates/decontamination** | Partial; default policy alignment outstanding | Global exact grouping, bounded equivalence, LSH/Jaccard candidate reports and review hooks | Instance/near-instance policy, controlled fixtures including template relatives, reviewed decisions, final source-to-group map and group-level exclusion across all banks |
| **5 — Shared formatting/labels** | Implemented and verified | Exact plain text, pinned actual fast tokenizer, combined encoding, prompt loss mask, one supervised EOS, positional padding, label-derived counts, whole-row length exclusion, helper/docs | Include this verified contract in the final release. Person 2 owns checking its actual training collator |
| **6 — Three banks/nested manifests** | Required split not delivered | Existing single-pool/nesting infrastructure and diagnostic exports | Disjoint 40K training, 10K selection and 1K loss pools; shared source-specific length-bin edges; corrected R500/R1000/R2000/R4000, ID lists, hashes and summaries; selection export before parent binding |
| **7 — Person 2 integration/readiness** | Partial | Exact token-report coverage/uniqueness checks, stale/invalid parent rejection, negative gate tests | Standalone integration scripts and explicit verification manifests/scopes; separate data/main/continuation prerequisites so absent trainer evidence or parent does not prevent Person 1 data readiness |
| **8 — Validation/package/handoff** | Partial; final package absent | 63 passing tests, notebook checks, small snapshot diagnostics/resume, saved evidence and usage contracts | Representative pilot using complete upstream fields, one full regeneration, final review/release validation, release manifest/checksums and `PERSON2_HANDOFF.md` |

Phase 5 is the strongest completed implementation block. The other phases contain reusable work or unfinished release obligations. This is **not nine completed phases**, and current diagnostics cannot substitute for the final data package.

## How much data is actually prepared

| Artifact | Amount | Meaning |
|---|---|---|
| Historical full canonical snapshots | 1,096,250 rows: Orca 200,035; Numina 896,215 | Preserved v1 inputs; historical extraction defects and Numina field loss prevent treating them as a corrected release |
| Historical full training bank | 40,000 examples | Old unapproved bank; not the new corrected 40K deliverable |
| Latest revised diagnostic processing | 5,000 rows, sampled 2,500/source | Small snapshot-derived run, not full upstream regeneration |
| Latest diagnostic initial eligibility | 775 rows | Passed initial flags; not independently mathematically certified |
| Latest checked diagnostic candidates | 278 rows, 139/source | Bounded matching/protection universe |
| Latest exported diagnostic pool | 100 examples, 50/source | Draft examples; 31,973 reference supervised tokens including EOS |
| Latest nested exports | 32, 64, 100 | Overlapping ordered prefixes of that same 100-row pool; do not sum them |
| Known reserved final-test questions | 1,819 | Validated question inputs; explicit Person 3 confirmation/development coverage remains pending |
| Corrected final 40K training / 10K selection / 1K loss banks | **0 of 3 released** | Required final banks remain undelivered |
| Corrected final R500/R1000/R2000/R4000 | **0 of 4 released** | Historical/pilot files with these sizes do not close this requirement |

The latest successful run is `runs/full_run_v2_20261012T181641Z_cfe8ce44`. It resumes with all 5,000 source rows reused and unchanged manifest hashes. Actual pinned-tokenizer verification is reusable because its saved manifest content hashes equal those verified in the preceding run. There are no completed trained models or model evaluations in this Person 1 deliverable.

## Remaining Person 1 work, in execution order

1. **Finish instance grouping and mapping.** Separate duplicate instances from changed-number/changed-word template relatives. Review candidate and missed-match evidence; export final mappings and apply exclusions to every group member.
2. **Complete source/target preparation.** Load complete pinned upstream fields, including original Numina answers and validity labels. Improve or review unresolved cohorts, check actual per-source supply and record shortages without reusing rejected data or truncating solutions.
3. **Allocate and export all three banks.** Reserve 25,500 eligible unique groups per source: 20,000 training + 5,000 selection + 500 loss. Use fixed source-specific length-bin edges, pairwise disjoint groups and the required ordered prefixes. Export selection data with parent binding pending.
4. **Finish review evidence.** Inspect larger stratified samples, record who reviewed what, apply corrections/quarantines and reprocess systematic findings before release.
5. **Finish Person 2 integration and independent readiness.** Export token-verification inputs/checker and actual-parent binding interface; make `DATA_READY_FOR_TRAINER` attainable without an actual checkpoint or trainer-label report.
6. **Run and package the final release.** Complete the representative upstream pilot and full regeneration, read back/validate artifacts, reconcile counts/hashes and issue a release manifest, audit and `PERSON2_HANDOFF.md`.

## Other-person decisions/work

Person 3 chooses/prepares/freezes its development/fast-development sets, confirms final-test roles and implements generation/scoring. Person 1 checks provided question manifests and records missing coverage; those choices are not an instruction to generate Person 3's evaluation data.

Person 2 adopts the reference format, verifies actual training labels, executes training/save/reload, and supplies actual parent evidence. Person 1 supplies the interfaces/data; it does not execute that training. Person 4 owns FIT scores, continuation selection/control arms and research analysis.

Person 1 is complete when the corrected package and saved evidence support **`DATA_READY_FOR_TRAINER`**. Main and continuation readiness are later states with downstream prerequisites.

## Organization verification

38 files were relocated; the two active plan paths remain stable. Runtime code/notebooks retain their paths and hashes. All 643 pre-existing saved artifacts checked against the organization inventory retain their paths, sizes and modification times. The 63 tests, notebook validation and diagnostic export read-back passed after the move.

The relocation map/original checksums are in [ORGANIZATION_MANIFEST.json](../evidence/ORGANIZATION_MANIFEST.json); [IMPLEMENTATION_STATUS.json](../evidence/IMPLEMENTATION_STATUS.json) remains the current machine-readable status. No corpus regeneration or new implementation phase was claimed during organization.

# Codebase guide and current Person 1 status

Repository distribution: code, tests and [release records](../release_records/person1_v3_20261012_ready/README.md) are tracked here. All datasets and original upstream files are available through the [download/folder guide](DATA_DOWNLOADS.md); extract the archives into the repository root.

Updated October 12, 2026 (America/Phoenix). **Person 1's current data deliverables are complete: DATA_READY_FOR_TRAINER.** All **1,096,250 pinned source rows** were processed. The immutable release supplies **40,000 training groups, 10,000 fresh selection groups and 1,000 held-out loss groups**, balanced by source. **87 tests pass**, all 11 manifests were re-encoded during export, and independent read-back verified **447 artifacts** plus the complete row-status audit. Git remains paused.

Requirements are the [supplied implementation plan](reference/Person1_Codex_Implementation_Plan.md), the exact user-supplied plain-text/tokenizer/loss contract, and the clarification that **Person 3 prepares its own evaluation data**. Earlier status is preserved in [PRE_FULL_V3_STATUS.md](milestones/PRE_FULL_V3_STATUS.md).

## Folder contents

| Folder/location | Contents |
|---|---|
| Workspace notebooks | Pre-alignment local notebook unchanged; current v2 is a thin local/Colab entrypoint using shared stages |
| `capstone_person1/*.py` | Importable extraction, canonicalization, grouping, encoding, allocation, release and integration modules |
| `configs/` | Pinned sources/tokenizer, seeds, source quotas and verified formatting contract |
| `sources/` | Four complete pinned train Parquet files, source inventory and dataset cards; 1,096,250 rows total |
| `releases/` | Preserved pilots/drafts and current `person1_v3_20261012_ready` immutable DATA_READY_FOR_TRAINER release |
| `inputs_from_team/` | 1,819 known final-test question texts/metadata, import contracts and downstream templates |
| `tests/` | 87 passing extraction, encoding, grouping, sampling/resume, review, schema and integration checks |
| `scripts/` | Input download, notebook build/validation, saved-release validation and protocol fixture tools |
| `scripts/history/` | Earlier one-time migrations and milestone recorders |
| `docs/contracts/` | Formatting/labels, Person 3 question inputs, Person 2 integration and data-schema contracts |
| `docs/reference/` | Unchanged copy of the supplied implementation brief |
| `docs/milestones/` | Historical milestone/status records |
| `evidence/` | Test results, tokenizer verification, historical-defect proof, metadata profile and synthetic protocol checks |
| `preserved_inputs/` | Hashed pre-alignment inputs and former embedded v2 notebook; the local input is not claimed to be an untouched original Colab source |
| `runs/`, `outputs/` | Historical v1/snapshot diagnostics; they do not substitute for the upstream release |
| `cache/`, `vendor/`, `runtime_deps/` | Tokenizer assets and isolated local dependencies |
| `logs/` | Processing logs |
| `project_documents/` outside the package | Proposals, charter, rubric and historical audit |
| GitHub checkout outside the package | Prior branch work; paused |

## Runtime modules

| Module | Responsibility |
|---|---|
| `canonical_data.py`, `source_stream.py` | Pure canonical mapping from complete pinned fields and resumable hashed upstream shards |
| `instance_matching.py`, `missed_match_search.py` | Case-preserving identities, context-aware equivalence, MinHash plus alternate retrieval, reviewed decisions and group exclusions |
| `release_catalog.py` | Global exact checks in SQLite; bounded candidate supply and complete final row/group mapping |
| `person1_data_contract.py` | Exact prompt, combined-text boundaries, reference labels, EOS, positional padding and whole-example length rejection |
| `pool_allocation.py` | Disjoint train/selection/loss banks, common token-bin edges and deterministic ordering |
| `release_reviews.py` | Preserved, hash-bound case queues and actual decision validation |
| `quality_review_policy.py` | Complete-source conservative review filters; source cohorts, multipart targets, damaged indices and possible question-answer leakage |
| `release_pipeline.py` | Named stages shared by notebook/CLI, packaging and independent readiness gates |
| `integration_checks.py` | Exact trainer-label reconciliation and actual-parent/selection freshness checks |
| `benchmark_manifests.py` | Person 3 frozen-question imports; explicit pending coverage |
| `snapshot_stream.py`, `snapshot_catalog.py`, `streaming_rebuild.py` | Historical diagnostic path; final releases use upstream stages |

## Implementation plan assessment

| Phase | Person 1 completion and saved evidence |
|---|---|
| 0 - Preservation/reproducibility | Done: local supplied notebook preserved; pinned inputs/configuration, implementation hashes, environment and immutable package recorded. Original Colab copy was not recovered and is not claimed as preserved. |
| 1 - Extraction/targets | Done: terminal evidence/ambiguities and supported scalar types implemented; historical 2-versus-32 defect independently verified as 32; percentage display corrected or quarantined. |
| 2 - Eligibility/review | Done: full-source quality dispositions, conservative cohort exclusions and unknown mathematical-correctness fields; 208 formatted, 39 risky, 40 prior-target and 12 pair review records identify Codex. |
| 3 - Evaluation protection | Done for supplied scope: 1,819 known final questions protected; frozen question import/role/nesting contract delivered. Person 3 owns missing development/fast-development inputs and final-role confirmation. |
| 4 - Groups/decontamination | Done: global exact checks over 1,096,250 rows, bounded near checks over 105,003 candidates, reviewed pair dispositions and complete final group map; unresolved groups excluded. |
| 5 - Formatting/labels | Done: pinned actual fast tokenizer, plain-text contract v3, combined boundaries, completion-only labels, one supervised EOS, positional padding masking and whole-example length exclusions verified. |
| 6 - Three banks/prefixes | Done: disjoint balanced 40K/10K/1K banks, debug_32 and nested 500/1K/2K/4K/8K/16K/40K prefixes; hashes, label-derived budgets, source shares and common length bins saved. |
| 7 - Integration/readiness | Done: actual-label and actual-parent validators, planned R1000 manifest/seen groups, downstream templates/contracts and separate data/main/continuation gates supplied. Actual training evidence remains with Person 2. |
| 8 - Package/validation/handoff | Done: immutable package, original-field canonical files, reports, tests, code, notebook and Person 2 handoff; all 447 saved artifacts and 11 manifests passed independent read-back. |

Current release: [PERSON2_HANDOFF.md](../release_records/person1_v3_20261012_ready/PERSON2_HANDOFF.md). Readiness and checksums: [release_manifest.json](../release_records/person1_v3_20261012_ready/metadata/release_manifest.json). Independent evidence: [FINAL_RELEASE_READBACK.json](../evidence/FINAL_RELEASE_READBACK.json). Machine-readable status: [IMPLEMENTATION_STATUS.json](../evidence/IMPLEMENTATION_STATUS.json).

Processed sources: 200,035 Orca and 896,215 Numina. A complete-source conservative review scan precedes the 105,003-candidate near universe. Reference encoding retained 96,025 candidates; 42 whole examples exceeded the 2,048-token limit after final serialization. Known review rejections then exclude entire implicated groups before bank allocation. Older drafts and pilots remain preserved and unapproved.

| Data export | Rows | Source composition |
|---|---:|---|
| Training bank | 40,000 | 20,000 Orca + 20,000 Numina |
| Fresh selection bank | 10,000 | 5,000 Orca + 5,000 Numina |
| Held-out loss set | 1,000 | 500 Orca + 500 Numina |
| Required training prefixes | 500 / 1,000 / 2,000 / 4,000 | Ordered prefixes of training bank |
| Debug smoke input | 32 | Both sources and varied lengths |
| Optional prefixes | 8,000 / 16,000 / 40,000 | Data exports; no training grid has run |

The three banks have no shared groups. Prefixes reuse training-bank rows, so their sizes are not additional unique data. Every manifest summary records its exact supervised-token budget, content/ordered-ID hashes, source example/token shares and length distributions. These are verified reference-label counts; actual trainer counts remain pending.

## What is left

**No outstanding Person 1 implementation or validation work remains for this release.** Once Person 3 supplies additional frozen evaluation questions, Person 1 imports them and produces a new version with updated overlap protection before comparative main runs. The finalized release stays unchanged.

Person 2 next adopts the shared encoder, verifies all debug_32 actual collator labels, trains/saves/reloads the 32-example smoke checkpoint, then verifies actual R1000 labels and starts independent learning-curve runs from the pinned base model. Person 3 chooses/freezes evaluation sets, confirms final-test roles and validates its generation/scorer. Person 4 owns student-fit scoring, continuation-arm selection and analysis. MAIN_RUN_READY and CONTINUATION_READY remain pending those actual downstream inputs.

Review decisions are scoped Codex inspections and conservative exclusions. Mathematical correctness remains unknown corpus-wide; approximate near retrieval and detected-overlap removal do not establish perfect recall or absence of pretraining exposure.

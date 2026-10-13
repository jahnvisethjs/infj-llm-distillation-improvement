# Codebase changes required by the supplied Person 1 implementation plan

> Delivered: **DATA_READY_FOR_TRAINER**, immutable `releases/person1_v3_20261012_ready/`. All **1,096,250 source rows** processed; **40K training / 10K fresh selection / 1K loss** plus nested prefixes exported; **87 tests** and independent verification of **447 artifacts / 11 manifests** passed. Person 1 current-release work is complete. Person 2 actual-label/training evidence and Person 3 evaluation freezes remain downstream inputs. [Current folder guide and remaining responsibilities](docs/CODEBASE_GUIDE_AND_STATUS.md). Earlier sections below are historical planning records. Git remains paused.

The organized folder layout and current phase-by-phase assessment are in [CODEBASE_GUIDE_AND_STATUS.md](docs/CODEBASE_GUIDE_AND_STATUS.md). Runtime/notebook/run paths are preserved; tests, support tools, contracts, milestone documents and evidence now have dedicated folders. Use [README.md](README.md) for current commands.

This plan was written from inspection and read-only behavioral probes. The first alignment batch and the Person 3 question-manifest interface are implemented. See [FIRST_ALIGNMENT_BATCH.md](docs/milestones/FIRST_ALIGNMENT_BATCH.md), [BENCHMARK_INTERFACE_BATCH.md](docs/milestones/BENCHMARK_INTERFACE_BATCH.md) and [IMPLEMENTATION_STATUS.json](evidence/IMPLEMENTATION_STATUS.json) for evidence and remaining work. The corrected full release remains incomplete.

Requirements: `C:/Users/jahnv/Downloads/Person1_Codex_Implementation_Plan.md`, together with the user's four-person responsibility table and subsequent ownership clarification. **Person 1 supplies the training/selection data; Person 3 can prepare its own evaluation data.** This clarification takes precedence over making Person 1's provisional benchmark generation mandatory. Existing progress reports describe earlier milestones; they do not establish compliance with all release deliverables.

## Target outcome and ownership

Person 1 supplies corrected and reviewed canonical data, duplicate groups, decontamination reports for the evaluation questions actually supplied/frozen, three disjoint training-source pools, required nested manifests, a reference formatting/label encoder, checksums and a usable Person 2 handoff. The proposed 1,000-row held-out loss pool is an internal training/analysis input; it is separate from Person 3's benchmark development/test datasets.

Person 2's actual training labels and trained parent are later integration inputs. They must not prevent Person 1 from reaching `DATA_READY_FOR_TRAINER`. Person 3 owns preparation/freezing of evaluation datasets, generation/scoring, metrics and error audits. Person 4 owns student-fit scoring, budget matching and selection arms, including MATCHED_RANDOM. Person 1's data preparation can proceed while Person 3 makes evaluation decisions. Missing Person 3 development questions must be recorded as pending evaluation coverage; final comparative-run readiness requires frozen evaluation definitions and completed overlap checks. This work does not train models, implement ranking, change benchmark scoring or run experiments.

## Agreed prompt, tokenizer and training-label contract

The user supplied the following exact contract. The first batch implements the shared reference encoder and verifies the actual pinned tokenizer and saved diagnostic manifests. Person 2 still needs to integrate and verify its training collator. Person 3 uses the shared prompt for evaluation; Person 4 uses the same serialization for student-fit scoring. This does not establish a full approved data release or a completed training pipeline.

### Plain-text prompt and completion

Use this exact prompt, including the final newline after `Solution:`:

```text
Question: {problem}
Solve the problem step by step. End with: Final answer: <answer>.
Solution:
```

Its exact escaped representation is:

```text
Question: {problem}\nSolve the problem step by step. End with: Final answer: <answer>.\nSolution:\n
```

Construct the training completion as:

```text
{worked_solution}

Final answer: {verified_final_answer}
```

The final-answer value must agree with the worked solution under the documented extraction/consistency review policy. Preserve an existing correct terminal answer block instead of adding a duplicate or rewriting it unnecessarily. Quarantine/review contradictory, nonterminal or ambiguous blocks; do not delete following reasoning to make a marker look terminal. Retain the original solution as provenance. A consistency check does not independently certify all mathematical reasoning.

Use plain-text serialization throughout training, student-fit scoring and evaluation. Do not apply a chat template or introduce chat-role/control markers.

### Pinned fast tokenizer and combined-text boundary

Use the configured Qwen2.5-0.5B base tokenizer:

```python
AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    revision=PINNED_TOKENIZER_REVISION,
    use_fast=True,
)
```

The currently configured revision is `060db6499f32faf8b98477b0a26969ef7d8b9987`. Record the exact revision and tokenizer asset hashes in the release; do not resolve `main` during replay.

Tokenize the **combined prompt and completion** using `add_special_tokens=False`. Verify the completion boundary against that combined encoding, using character offsets and/or a validated prefix comparison. Separately tokenized prompt/completion IDs are not authoritative for constructing the final sequence or labels. Quarantine a boundary-crossing token when the reference encoder cannot honor the specified prompt/completion masking exactly.

If a later experiment uses Qwen2.5-1.5B, load its corresponding pinned tokenizer and verify compatibility before reusing any 0.5B token counts or encoded data. Record the compatibility result and tokenizer identity for that experiment.

### One supervised EOS

Append **one token ID** after the completion using `tokenizer.eos_token_id`. Expected values for the selected base tokenizer are:

```text
EOS token: <|endoftext|>
EOS ID:    151643
PAD token: <|endoftext|>
PAD ID:    151643
```

Verify these values in the pinned tokenizer actually loaded by the implementation. Export the verified token strings/IDs in `configs/formatting_contract.json`; a code constant or a link to `main` is not load-time verification.

`Final answer:` is ordinary text used by the answer parser. EOS marks the end of the generated completion. Append EOS explicitly after combined-text tokenization; do not add an EOS string to the prompt or rely on automatic special-token insertion.

Reference provided by the user: [Qwen2.5-0.5B tokenizer configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/tokenizer_config.json). Replay verification must use the configured pinned revision.

### Completion-only loss and budget

| Sequence part | Receives loss? | Label |
|---|---|---|
| Question and instructions | No | `-100` |
| `Solution:` and its final newline, belonging to the prompt | No | `-100` |
| Worked solution | Yes | Corresponding token ID |
| Completion separators and `Final answer:` text | Yes | Corresponding token ID |
| Final-answer value | Yes | Corresponding token ID |
| Appended EOS | Yes | `tokenizer.eos_token_id` |
| Padding | No | `-100` |

Prompt tokens remain visible through attention. Masking their labels removes their prediction loss; it does not mask prompt attention. Real prompt/completion/EOS positions have attention mask 1; padding has attention mask 0.

Compute the experiment budget from the actual reference labels:

```python
supervised_tokens = sum(label != -100 for label in labels)
```

Mask padding by **position/attention mask**, not by replacing every label whose token ID equals the pad ID. Padding and EOS share ID 151643, so masking by ID would incorrectly erase the genuine supervised EOS. Keep prompt, completion, EOS, total non-padding input and supervised-token counts distinct in exported summaries. Person 2 still checks its actual training collator against this reference contract.

Reference provided by the user: [Hugging Face Qwen2 documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2).

### Evaluation input and required verification

During evaluation, supply **only the prompt ending at `Solution:\n`**. Do not supply a reference worked solution, final-answer value or pre-appended EOS. The model generates its reasoning, final answer and EOS. Generation settings/scoring remain Person 3's responsibility; the agreed prompt serialization is shared.

Add tests for the exact prompt bytes/final newline, preservation of an already-correct answer block, contradictory/nonterminal-marker rejection, combined encoding and boundary handling, absence of chat-template application, verified EOS/pad identities, one explicitly appended supervised EOS, prompt-only loss masking, padding with the shared EOS ID, label-derived counts and prompt-only evaluation input. Tokenizer compatibility verification is required before any later 1.5B reuse.

## Historical preparation counts

Counts below record the earlier planning baseline. [FIRST_ALIGNMENT_BATCH.md](docs/milestones/FIRST_ALIGNMENT_BATCH.md) records the newer diagnostic counts and repaired marker/NaN/alternative-answer paths. Passing eligibility flags or exporting a file does not certify mathematical correctness or a completed release.

| Data/artifact | Amount currently recorded | Status |
|---|---|---|
| Preserved full source canonical snapshots | **1,096,250 rows**: 200,035 Orca + 896,215 Numina | Historical v1 data with raw text/provenance; extraction defects remain in its targets |
| Historical v1 initially eligible rows | **554,444** | Initial flag eligibility, not an approved final corpus |
| Historical v1 checked candidate universe | **60,000**, initially 30,000/source | Checked under the old bounded matching/test-only policy |
| Historical v1 training bank | **40,000 examples**, 20,000/source | Exported but unapproved; requires corrected regeneration and reviews |
| Historical v1 nested manifests | **500, 1,000, 2,000, 4,000**, plus 8,000/16,000/40,000 and debug_32 | Nested views of the same bank; do not add their counts together |
| Latest revised diagnostic source processing | **5,000 rows**, 2,500/source | Completed streaming/checkpoint path; includes the earlier 1,000-row smoke input scope, so this is not 6,000 unique processed rows |
| Latest diagnostic initial eligibility | **666 rows** | Passed current initial flags on this subset; not independently mathematically verified |
| Latest checked/formatted diagnostic candidates | **400 examples**, 200/source | Bounded diagnostic universe after the implemented checks; 100 selected and 300 outside the selected pool |
| Latest diagnostic exported training pool | **100 examples**, 50/source | Draft only; not a final research training bank |
| Latest diagnostic nested exports | **32, 64 and 100**, plus debug_32 | Prefixes of the same 100-row pool; not 196 distinct training examples |
| Reserved final-test problem-text manifest | **1,819 questions**: 1,319 GSM8K test + 500 MATH500 | Already available for decontamination; does not mean an evaluator or model scores have been delivered |
| Final corrected 40K bank and R500/R1000/R2000/R4000 | **Not yet delivered** | Historical files cannot serve as the corrected release |
| Disjoint 10K selection bank and 1K loss pool | **Not yet delivered** | Person 1 still needs to implement/export these; a trained parent is not needed to create them |
| Person 3 benchmark development/fast-development data | **No completed manifests in this workspace** | Person 3 prepares/chooses them; existing templates are not datasets |

Evidence paths: `runs/full_run_v1/outputs/reports/audit_summary_full.json`, `runs/full_run_v2_20261012T004834Z_c75c286d/outputs/reports/audit_summary_pilot.json`, `inputs_from_team/reserved_evaluation_metadata.json`, and `NEXT_STEP_VERIFICATION.json`. The 5,000-row diagnostic resumed successfully with unchanged manifest hashes. Operational verification records 32 passing tests; those tests do not cover every subsequently identified defect.

## What remains for Person 3 to decide and deliver

Person 3 can construct its own evaluation data independently. Person 1's responsibility is to expose the supplied banks/provenance/group mapping and check overlap against Person 3's frozen question manifest once available.

| Person 3 decision/work | Current starting point | What Person 1 needs for coordination |
|---|---|---|
| Choose full development datasets, origins, counts, IDs and sampling policy | The implementation brief proposes 500 GSM8K-train + 500 MATH-train questions; these are proposals for Person 3 to confirm or revise | Frozen question IDs/text, role, source split/revision, selection seed and file hash |
| Choose a smaller fast-development subset | Proposed 100 GSM + 100 MATH, nested inside full development | Nested IDs/text and hashes so the same protected union is used |
| Confirm final tests and when they are evaluated | GSM8K test and MATH500 question texts already exist here | Confirm these roles, versions and untouched-final-test policy, or supply a documented revised configuration |
| Define gold-answer storage and scoring | Person 1's manifest contains question text; no completed evaluator is supplied | A documented scoring/answer-format contract; reference labels can remain under Person 3's control |
| Define generation settings and evaluator validation | Pending Person 3 work | Consistent settings across baseline/checkpoints; validated numeric/symbolic answer extraction/equivalence and error audits |
| Freeze overlap handling for the chosen evaluation questions | Current matching scope is bounded and still needs policy alignment | Agreement on duplicate instances versus template relatives; frozen question texts for Person 1's exact/near checks |
| Run baseline/checkpoint evaluations and export metrics/predictions | No completed model evaluation is claimed here | Evaluation results are later Person 3 deliverables, not prerequisites for Person 1 to supply its data package |

Person 3 need not rebuild Person 1's training bank or create its nested training manifests. Person 1 need not build Person 3's benchmark scorer or generate model predictions. If Person 3 chooses questions that overlap the supplied banks, coordinate removal/reselection or a new version before comparisons. Until then, the data package must state exactly which known evaluation questions were protected and which coverage remains pending.

## Existing work to retain

- Stable source IDs, raw-text preservation, configured repository revisions and accurate nonexclusive flag reporting.
- Terminal-extraction fixes and safe rational equivalence, after repairing the remaining edge cases below.
- Saved-review preservation, exact coverage/unique-ID token reconciliation, artifact hashing and negative gate tests.
- Streaming Parquet checkpoints, the SQLite metadata catalog, memory observation and interruption/resume validation.
- Recorded diagnostic smoke results: 1,000 and 5,000 source rows; nested 32/64/100-row outputs; verified 5,000-row reuse; 32 passing tests at the previous milestone.

These diagnostic outputs are not corrected R500/R1000/R2000/R4000 manifests, a representative source audit, or a final data release.

## Confirmed defects and conflicts

| Finding | Current location | Required correction |
|---|---|---|
| An interior final marker is mistaken for the terminal marker | `pipeline_controls.py`: extraction, quality check and completion formatting use multiline `$` | Match the end of the entire solution; examine all candidates; preserve trailing reasoning or quarantine contradictory/nonterminal markers |
| Formatting can remove text after that interior marker | `build_completion` slices the solution at the matched marker | Never discard subsequent derivation/correction text; normalize only a verified trailing answer block |
| NaN becomes an ordinary answer string | `clean_text` / `compact_answer` | Detect nulls/NaNs before string conversion; separate missing values, sentinel values and unsupported structures |
| A worked solution ending `Final answer: 3 or 4` passes target-quality checks | Explicit marker handling accepts arbitrary marker payloads | Classify supported scalar answers versus alternatives, tuples, multi-part structures and malformed expressions |
| Default policy removes lexical parameter/word variants | Notebook policy and threshold application; review-selection enforcement | Default to instance/near-instance duplicates; make broader variant/family removal an explicit separately versioned option |
| A similarity threshold directly determines near unions/removals | Notebook grouping cell | Keep retrieval/score evidence distinct from the instance decision; apply approved instance decisions consistently across groups |
| The fresh pool waits for a parent and can overlap the rest of the training bank | Notebook fresh-pool cell | Export a selection bank disjoint from the entire training bank before any parent exists; validate the actual parent later |
| One readiness Boolean requires actual labels and a trained parent | Notebook final cell | Separate data readiness, main-run readiness and continuation readiness |
| Token counts are arithmetic estimates, with no reference labels | Tokenization cell and `tokenizer_adapter.py` | Construct actual reference labels and derive supervised counts from labels unequal to -100 |
| Local regeneration reconstructs missing upstream fields from old flags | `SnapshotCanonicalizer` / `streaming_rebuild.py` | Preserve snapshot mode for diagnostics; add a complete pinned-upstream path for the final release |
| Smoke sampling uses the first source rows | Streaming `limit` option | Keep this as a systems smoke mode; add a representative deterministic pilot with preserved original row positions |

Read-only probe results from this planning pass:

- A solution containing `Final answer: 2` followed by a correction concluding 32 returned `('2', 'explicit_final_marker')`.
- Formatting that solution with target `2` removed the following correction text.
- `compact_answer(float('nan'))` returned the text `nan`.
- A worked solution ending `Final answer: 3 or 4` returned an explicit-marker answer with no target-quality flags.

These are fixture-level defects. Their prevalence in the corpus has not been measured. Existing passing tests must be extended before regeneration.

## Ordered implementation work

### 0. Preserve the current inputs and establish the v2 entrypoint

Plan phase: 0. Touch the notebook entrypoints, configuration, runner and synchronization/validation utilities.

- Preserve the current modified notebook with its hash as the pre-alignment input. Recover the actual original notebook from an available original source if possible, and record its hash. Do not label the already-modified file as an untouched original.
- Create a distinct `Person1_Full_Run_v2.ipynb` with cleared execution outputs. Keep `full_run_v1`, old review evidence and diagnostic runs unchanged.
- Add a structured configuration file covering model/tokenizer revisions, source revisions, both seeds, source quotas, protected-set roles, overlap policy, review policy, completion/EOS policy, pilot mode and artifact schemas.
- Add explicit project-root and optional Drive-mount controls. Use release-relative paths in package metadata.
- Freeze the complete environment/dependency versions and all used code/input hashes. Validate configured revisions rather than resolving latest versions or silently substituting sources.
- Make notebook/runner paths configurable. Replace dependence on numbered notebook cells and source-text substitutions with calls to a small set of shared stage functions. Retain the current streaming/catalog infrastructure; avoid a large architecture rewrite.

Acceptance: separate preserved/corrected notebooks; reproducible configuration; notebook-format and magic-aware syntax checks; a small local smoke and equivalent Colab entrypoint using the same stage logic.

### 1. Complete extraction and eligibility semantics before any release

Plan phases: 1 and 2. Touch `pipeline_controls.py`, canonicalization shared functions, `snapshot_stream.py`, tests and notebook wrappers.

- Fix entire-solution terminal anchoring in extraction, quality diagnostics and formatting together. Detect multiple/conflicting candidates and keep all original reasoning.
- Return structured extraction metadata: candidate, extraction mode, terminal evidence text/span, confidence category, supported answer type and ambiguity reason. Retain `answer_parse_status` as a compatibility alias if useful.
- Handle numeric scalars, signed values, decimal points, fractions, percentages, units and supported symbolic scalars explicitly. Reject unsupported alternatives, partial answers, malformed/truncated structures and missing/sentinel values.
- Distinguish mathematical correctness, extraction consistency and solution completeness in the schema; keep unknown assessments unknown.
- Treat answer-only/short-solution heuristics as documented completeness diagnostics. Keep a separate review/quarantine resolution; do not equate a word-count threshold with mathematical correctness.
- Split external/missing images from inline Asymptote, SVG and TikZ flags. Default policies must document whether each form is supported or quarantined.
- Preserve original Numina answer strings and upstream validity labels in the final source path. Old 500-character answers and inferred validity must remain diagnostic-only.

Acceptance tests: every regression case listed in the supplied plan, plus interior markers with trailing corrections, multiple explicit markers, NaN/null variants, malformed boxes, unit-bearing conclusions and alternate answers after explicit markers. Formatting must preserve reasoning and must not accept a known contradiction.

### 2. Establish the shared reference encoder

Plan phase: 5. Add `person1_data_contract.py`; reuse the existing template/completion helpers and pinned tokenizer adapter.

Required public functions:

- `render_prompt(problem)`
- `build_completion(solution, final_answer, extraction_metadata)`
- `encode_example(example, tokenizer)` returning input IDs, attention mask, labels and distinct count fields
- `validate_manifest(path, release_metadata)`

Implement the exact agreed contract above. The encoder must tokenize the actual concatenated prompt/completion with `add_special_tokens=False`, validate the character/token boundary, verify the pinned fast tokenizer's EOS/pad identities, append one supervised EOS and construct completion-only labels. Preserve an existing correct terminal answer block. It must handle padding by position/attention mask so a pad ID shared with EOS does not erase the supervised EOS. Derive `supervised_tokens` from the actual reference labels. Exclude whole overlength examples without truncating reasoning. Expose the same exact prompt renderer for Person 3 evaluation and Person 4 scoring.

Export `configs/formatting_contract.json`, tokenizer/template/completion-policy versions and hashes. Make both notebook and local runner use this helper. Person 2 should import it rather than recreate serialization.

Acceptance: label masking, supervised EOS, padding/EOS shared-ID cases, empty/unsupported examples, boundary crossings and length-limit behavior are independently tested without model weights. Exported reference counts match reference labels on every selected row.

### 3. Record Person 3 evaluation decisions and apply protection checks

Code status: the import/validation interface, protected-question export and known/pending scope report are implemented and tested. See [BENCHMARK_INTERFACE_BATCH.md](docs/milestones/BENCHMARK_INTERFACE_BATCH.md). The existing 1,819 final-test questions are checked in the diagnostic universe; Person 3 development/fast-development inputs and explicit final-role confirmation remain pending.

Plan phase: 3, adjusted to the user's ownership clarification. Update the notebook protection stage and add an import/validation helper for Person 3's frozen question manifests.

- Person 3 prepares and freezes its full/fast development sets and confirms final-test roles. The proposed dataset sizes remain documented defaults for its decision, not an instruction that Person 1 must build its evaluator data.
- Accept manifests with explicit benchmark family, role, source/revision/configuration/split, selection policy and hashes. Keep the existing 1,819 final-test questions available for current protection.
- Validate supplied fast-subset nesting and development/test intersections. Run exact/near checks against the frozen protected union for all three Person 1 pools.
- Export/import benchmark metadata and `protected_questions.csv`, with the scope actually checked. Benchmark answer labels are not required for Person 1's decontamination.
- If development manifests are absent, continue Person 1 data/contract/pool work and record `person3_development_coverage=pending`. Keep final comparative-run readiness blocked until those definitions and overlap checks are complete.
- Person 1 can help construct provisional development manifests if explicitly requested later. Any such assistance must record provisional origin, train-only sampling and pending Person 3 adoption; it must not fabricate evaluation sign-off.

Acceptance: Person 1 exports an honest protected-scope report and an interface for Person 3 manifests. Once supplied, frozen roles/provenance, fast-subset nesting and overlap checks reconcile. Missing evaluation coverage remains explicit and blocks comparisons, rather than assigning Person 3's dataset preparation to Person 1. No test accuracy is used for these data decisions.

### 4. Align group matching and export the final group map

Plan phase: 4. Touch `snapshot_catalog.py`, shared near-matching functions, review enforcement and configuration.

- Keep global exact grouping and safe, bounded answer equivalence. Record unsupported symbolic comparisons as unresolved; quarantine those groups according to policy.
- Define the permitted assignment/scalar normalization context instead of treating every answer-format transformation as automatically equivalent in every problem.
- Make instance/near-instance matching the default. Similar wording, changed quantities and changed words are candidates for review, not automatically the same question. Optional variant/family policies must be separately named and versioned.
- Keep MinHash/LSH retrieval and Jaccard scores as evidence, with approximate-recall limitations. Include controlled parameter/word-variant fixtures and real near-threshold/missed-match samples.
- Apply reviewed instance/group exclusions to every group member, including exported pools and preserved source mappings. Ensure below-threshold findings trigger the appropriate correction before release.
- Export `processed/problem_group_mapping.parquet`: source row ID, original exact group, final near component, representative, disposition and decision provenance.
- All three pools must come from the checked universe. Expanding it requires the same matching/protection checks before new examples can enter a pool.

Acceptance: ELLIPSE/BALLOON is classified as a template relative under the default policy; duplicate instances group consistently; safe equivalence and unresolved conflicts are distinguished; group-level exclusions and mappings reconcile.

### 5. Implement the three disjoint pools and required prefixes

Plan phase: 6. Replace the existing working-pool/fresh-pool sequencing with a deterministic pool allocator; reuse selection ordering and manifest/hash utilities.

- Produce a 40,000-group training bank, 10,000-group selection bank and 1,000-group held-out loss set. Per source, reserve 20,000 + 5,000 + 500 = **25,500 eligible unique groups** after all corrections/protection/length exclusions.
- The present 30,000-candidate-per-source default may be sufficient, but this must be checked after exclusions. Report shortages or expand the checked universe under a recorded new configuration; never refill from rejected records.
- Calculate source-specific supervised-token length-bin edges once on the declared reference population and export `configs/length_bin_edges.json`. Apply those same edges to all pools, with explicit handling of tied lengths and endpoints.
- Use deterministic source/length allocation and validate pairwise group disjointness. Export the selection bank immediately with parent binding pending.
- Produce ordered nested R500/R1000/R2000/R4000 training prefixes and a debug set containing both sources and varied lengths. Larger exports remain optional; their existence does not authorize a training grid.
- Include raw/reference solution provenance, extraction metadata, quality/review flags, group ID, versions and reference label counts. Preserve `row_id`/`dataset_id` compatibility.
- Export bank/prefix files, ID lists, content/ordered-ID hashes, source example/token shares, length summaries, maxima and planned R1000 seen groups.

Acceptance: exact quotas when requested, unique groups, three pairwise-empty intersections, ordered prefix equality, common length-bin interpretation, valid targets and identical benchmark protection across every pool.

### 6. Complete review evidence and apply findings

Plan phase: 2, with phase 4 reviews. Touch sample generation, review schemas and application/approval helpers.

- Increase formatted-target review from 10 examples per source to approximately 100 randomly selected examples per source, stratified by extraction/answer mode. Add risky/ambiguous examples and duplicate/overlap cases.
- Preserve reviewer identity, date, sample IDs, policy version, decisions, notes and unresolved cases. Codex may record cases it actually inspected under its own identity; human approval remains distinct.
- Apply corrections/quarantines through versioned input decisions before splitting/exporting. The final gate should verify that decisions were applied, rather than being the only place unresolved decisions are detected.
- Fix systematic defects at the affected-cohort level and reprocess downstream artifacts. Review changed evidence against the correct final source/control version; do not carry approval hashes across changed inputs.

Acceptance: saved actual decisions, documented policy adoption, resolved findings in every bank, and explicit unresolved prerequisites preventing data-ready status where required.

### 7. Separate readiness and package Person 2 integration

Plan phases: 7 and 8. Refactor the final notebook gate; reuse token-reconciliation, parent-binding and artifact-integrity helpers; add small integration entrypoints.

| State | Requirements |
|---|---|
| `DIAGNOSTIC_ONLY` | Limited, snapshot-derived or incompletely reviewed outputs; clear limitations |
| `DATA_READY_FOR_TRAINER` | Complete corrected training/selection inputs, reviewed data policies, reference encoder, three disjoint pools, required manifests, checksums, package validation and explicit known/pending evaluation-protection scope |
| `MAIN_RUN_READY` | Data-ready release plus Person 2 actual-label verification and Person 3 evaluator/split contract |
| `CONTINUATION_READY` | Main-run prerequisites plus bound actual parent and verified selection-pool freshness |

- Record each prerequisite independently as passed/pending/failed, with evidence paths and hashes. Missing actual training labels or a parent must leave the data-ready state reachable when Person 1's requirements are satisfied.
- Track Person 3 dataset preparation separately: Person 1 may supply its validated data package with clearly declared pending development coverage. `MAIN_RUN_READY` requires Person 3's frozen development/test contract and completed overlap checks for that configuration.
- Export a 32-row verification-input manifest and all corrected R1000 verification inputs. Specify their expected scopes independently; do not hardcode every check to an assumed R1000 artifact in a diagnostic run.
- Provide `integration/verify_person2_tokens.py` with exact coverage, unique-ID, one-to-one, count/hash checks and a reconciliation report identifying failures. Record trainer, masking, EOS and padding contract versions.
- Provide `integration/bind_parent.py` with actual run/checkpoint identity, actual manifest hash and actual seen groups. Preserve rejection of a renamed planned template or stale evidence.
- Default to the agreed corrected R1000 parent. If Person 2 intentionally trained outside the declared train bank, classify it as a different configuration requiring explicit freshness/eligibility revalidation; do not silently treat it as the agreed parent.
- Bind all three banks, nested manifests, group map, protected questions, contracts and summaries into release metadata. Read back files when checking hashes and structural invariants.

Acceptance: valid Person 1 data can reach data-ready without a model checkpoint; missing actual-label evidence blocks main-run readiness; missing/stale parent evidence blocks continuation readiness. Invalid source/review/formatting/disjointness evidence blocks data readiness itself.

### 8. Run the required release sequence and produce the handoff

Plan phase: 8. Touch test coverage, orchestration, packaging and documentation.

1. Run targeted offline regressions, including the new confirmed defects and reference labels/pool splitting/readiness transitions.
2. Run a representative pilot using complete pinned source fields and deterministic sampling across the sources. Preserve the earlier contiguous-row smoke runs as operational evidence only.
3. Inspect pilot targets, review findings, stage counts and exported files. Resolve defects before another run.
4. Run one full regeneration when required source data/runtime are available, using the shared stage code and checkpointed processing. If access blocks it, export the precise status and evidence instead of declaring completion.
5. Finish reviews, validate/read back the complete release and write `reports/release_validation.json` and a release manifest.
6. Package the corrected notebook, shared contract, tests, configurations, canonical/group/status outputs, all banks/prefixes, benchmark inputs, integration tools, audit/review evidence and `PERSON2_HANDOFF.md`.

The handoff must include stable paths/hashes, dependencies, loading commands, reference-token totals, Person 2 verification instructions, smoke/save/reload responsibilities, base-model initialization for independent learning-curve runs, planned parent binding, selection-bank freshness, benchmark ownership and limitations. Person 1 does not execute model training as part of this package.

## File-level change map

| Existing/new file | Planned responsibility |
|---|---|
| `Person1_Full_Run_v2.ipynb` (new) | Distinct corrected Colab/local entrypoint calling shared stage helpers |
| `pipeline_controls.py` (modify) | Safe parsing, structured extraction, answer/diagram/completeness flags and reusable review/integration checks |
| `person1_data_contract.py` (new) | Exact prompt/completion, reference encoding/labels/counts and manifest validation |
| `snapshot_stream.py` (modify) | Keep diagnostic snapshots; add complete source-field mapping and deterministic pilot input selection |
| `snapshot_catalog.py` (modify) | Disk-backed exact grouping/equivalence plus final group-map export |
| `streaming_rebuild.py` / notebook stages (modify) | Shared named stage calls, complete-source mode, diagnostics versus representative pilot, structured configuration |
| Small benchmark-manifest helper (new) | Import/validate Person 3's full/fast/final definitions and export the checked protection scope; provisional generation only if separately requested |
| Small pool-allocation helper (new) | Fixed bin edges, three disjoint banks, ordered prefixes and summaries |
| Readiness/package helper (new or extracted) | Independent prerequisites, staged statuses and immutable release metadata |
| `integration/verify_person2_tokens.py`, `integration/bind_parent.py` (new) | Executable interfaces around existing validators |
| Tests (extend) | New parsing regressions, actual labels, pool allocation, policy distinctions and readiness transitions |
| `PERSON2_HANDOFF.md` (new) and existing progress/handoff docs | Final load/verification instructions and accurate ownership/completion status |

Exact helper filenames beyond the specified contract/package can be kept minimal during implementation. This plan does not call for introducing a service, database server, training framework or new selection system.

## First implementation batch — code delivered, full release pending

Preservation/configuration, target regressions, structured extraction and the shared reference encoder are implemented. The Person 3 manifest import/protection interface is also implemented. The v2 diagnostic processes 5,000 randomly sampled snapshot rows and exports a balanced 100-row pool with nested 32/64/100 manifests. These establish tested target safeguards, reference label counts and an honest protection-input interface; final upstream regeneration and release approval remain pending. Next implement instance grouping, pool splitting, larger reviews, staged readiness and packaging. Person 3 prepares its own evaluation definitions; coverage remains pending until it supplies frozen question texts.

## Definition of completion

The codebase meets the supplied plan when the package and saved evidence support `DATA_READY_FOR_TRAINER`, with every required phase acceptance check satisfied or accurately reported as a blocker. Existing test success, small draft exports and preprocessing totals cannot substitute for that release. Person 2's later actual-label/training work and Person 4's selection/analysis remain separate.

# Person 1 Codex implementation plan and Person 2 handoff

## Purpose

Complete the data work in `Person1_Full_Run.ipynb` so Person 2 can load corrected, reproducible inputs, run a training smoke test, reconcile training-label token counts, and start the learning-curve experiments. This is an implementation brief for Codex, not a claim that the fixes or new data release have already been executed.

The project asks what a small language model should train on next when the additional supervised-token budget is limited. The initial study uses Qwen2.5-0.5B base, an exploratory learning curve, and continuation from the same R1000 parent using REPEAT, RANDOM, FIT, and MATCHED_RANDOM. Person 1 prepares data; Person 2 trains; Person 3 owns evaluation; Person 4 owns selection and analysis. This brief follows the latest recommended experiment rather than reinstating every component in the original proposal.

## Copy this task instruction to Codex

Implement the phases below in order. Preserve the original notebook and original run artifacts. Produce a new corrected notebook/release and all supporting handoff files. Keep the notebook usable in Colab and a local Python environment. Run meaningful offline regression tests before any corpus regeneration, then a representative pilot, and finally one full regeneration if the required source data and environment are available. Do not stop at writing recommendations. Do not claim full processing, manual review, or readiness for experiments without saved evidence. If access prevents a stage, complete independent work, report the precise blocker, and export an honest readiness status.

Do not train a model, implement FIT ranking, construct budget-matched continuation arms, change benchmark scoring, or add a verifier/routing system. Those are other workstreams. The helper that constructs training labels using the tokenizer is within this task and does not require loading model weights.

## Inputs and current evidence

- Current supplied notebook: `/workspace/scratch/404a50156175/project_sources/05-Person1_Full_Run.ipynb`.
- Current supplied audit: `/workspace/scratch/404a50156175/project_sources/01-data_audit_full.md`.
- Notebook SHA-256 inspected for this plan: `c754efc852193112f6afcad747ca740cb574adea055121da8c82571fd74dcce9`.
- Original Colab notebook: https://colab.research.google.com/drive/1218wwpQXfrpckqtPSuuL6SzkCPgcGood
- Existing Colab project root: `/content/drive/MyDrive/capstone_person1`.
- Existing run: `runs/full_run_v1/outputs/` with `processed/`, `reports/`, and `manifests/`.
- The existing notebook has 34 cells and substantial D1-D7 implementation. It creates an initial eligible corpus, global exact duplicate groups, a bounded 60,000-example near-duplicate universe, a balanced 40,000-example working pool, and nested manifests.
- Existing audit counts are historical inputs. Recompute every count after corrections; do not preserve old counts as targets.
- Confirmed old R1000 defect: row `11901a7af64516459d2530de9a1604e411f4f5849507acbd05443670cf1440b4` solves `(x-6)/13=2` as 32 but appends `Final answer: 2`. Old file: https://drive.google.com/file/d/1f8MdbZuaesTuLBiyUBMSRTtD4qhsEJ4N/view ; line 53 in the inspected version. Locate by ID, not a permanent line number.

Read the current files afresh at execution time. If the notebook has changed, adapt this plan to its latest implementation and record the new input hash. Inspect any applicable `AGENTS.md` before editing in a repository.

## Decisions to implement

These are configurable defaults. Record them together in a versioned configuration; do not silently change them to make the output succeed.

| Decision | Default |
|---|---|
| Initial student | Qwen2.5-0.5B base; exact model/tokenizer revisions recorded |
| Training sources | microsoft/orca-math-word-problems-200k and AI-MO/NuminaMath-1.5 |
| Eligibility scope | Text-complete answer-verifiable problems under the documented word-problem policy; retain supported numeric and symbolic scalar answers |
| Training bank | 40,000 groups: 20,000 per source |
| Selection bank | 10,000 groups: 5,000 per source; disjoint from the entire training bank |
| Held-out loss set | 1,000 groups: 500 per source; disjoint from both banks |
| First manifests | debug_32 and nested train_n500, train_n1000, train_n2000, train_n4000 |
| Larger manifests | Optional existing 8K/16K/40K exports; their existence does not authorize a training grid |
| Benchmark development | 500 from GSM8K train; 500 from MATH train |
| Fast development | 100 GSM plus 100 MATH, nested inside full development sets |
| Final benchmarks | Official GSM8K test and MATH500; never development subsets |
| Length limit | 2,048 tokens for the complete training sequence, including supervised EOS |
| Seeds | Split seed 1729; manifest ordering seed 42; record both |
| Near matching | Instance/near-instance duplicate policy by default; template-family removal only under a separately documented policy |

Ensure sufficient per-source supply after all corrections. If the requested pools cannot be produced, fail the intended configuration with an explicit report or produce a separately labeled smaller diagnostic configuration. Do not fill quotas by reintroducing rejected examples, duplicating groups, or truncating solutions.

## Phase 0 — Preserve inputs and create a reproducible run

Primary cells: 2, 4, 7, 9.

1. Write the corrected notebook to a distinct path such as `Person1_Full_Run_v2.ipynb`; preserve its original source and avoid stale execution outputs suggesting new cells have run.
2. Use an immutable run directory such as `full_run_v2_<UTC timestamp>`. Do not overwrite `full_run_v1` or completed review records.
3. Export a complete configuration, input notebook/code hash, environment versions, source revisions, tokenizer revision, seeds, prompt/completion policy versions, and all input-file hashes.
4. Configure fixed revisions for replay. A recorded SHA is not sufficient if every rerun resolves the newest revision again. Validate revision loading and report inaccessible revisions rather than silently substituting newer content.
5. Keep Colab Drive mounting optional, with an explicit local project-root override. Use paths relative to the configured root for the handoff package.
6. Reuse small helpers from the notebook or extract a minimal helper module for testability. Avoid rebuilding the entire project architecture.

Acceptance: old artifacts remain intact; rerun configuration is complete; the corrected notebook passes notebook-format and code-cell syntax checks, accounting for Colab magics.

## Phase 1 — Repair answer extraction and target construction

Primary cells: 13, 14, 25.

1. Replace whole-solution answer-phrase selection with terminal-answer identification. A quoted given value or intermediate result is not a final answer merely because it follows `answer is` or `result is`.
2. Return answer candidate, extraction mode, evidence span/terminal block, confidence category, and ambiguity reason. Preserve original problem/solution/answer strings.
3. Handle explicit final markers, terminal boxed expressions, fractions, negative numbers, decimals, percentages, units, and supported symbolic expressions without selecting only the first number of an expression.
4. Treat multi-answer questions, conflicting terminal candidates, truncated answer expressions, and missing/ambiguous answers explicitly. Quarantine unsupported multi-answer cases instead of converting a total or one component into the complete answer.
5. For Numina, distinguish missing/sentinel answers from unsupported answer types. Rename `missing_or_non_numeric_final_answer`: the old code does not establish numeric-only validity. Handle nulls/NaNs and sentinel strings before converting them to ordinary text.
6. A literal answer substring check is a diagnostic, not mathematical validation. Keep extraction consistency, solution completeness, and mathematical correctness as separate fields; unknown correctness remains unknown.
7. Preserve worked reasoning. Append one final-answer block only when the extraction policy supports it. If a pre-existing final block disagrees, quarantine/review the example; do not silently append a contradictory answer or delete evidence of disagreement.
8. Reprocess all affected extraction modes, not just the known bad rows. Rebuild targets, token counts, exact-answer conflicts, and downstream manifests after the repair.

Required regression cases:

- `(x-6)/13=2` with a conclusion of 32 must emit 32, never 2.
- A solution that repeats `the answer is 11` from the question and concludes the missing number is 1 must emit 1.
- A phrase such as `the answer is 1/2 + 1/3 = 5/6` must emit the supported terminal result or quarantine; never accept 1/2 as the complete answer.
- Multiple alternatives such as `3 or 4` must not become a confident scalar 3.
- Individual lunch amounts must not become their total merely because the total appears last.
- Nested LaTeX boxed fractions, negative values, decimal points, percentages, and already-correct final blocks must be handled consistently.
- Preserve symbolic scalar answers and quarantine unsupported answer structures without relabeling them non-numeric errors.

Acceptance: fixtures pass, the historical contradiction is fixed in regenerated data or explicitly quarantined, and no known contradictory target is included in accepted manifests. Fixture success does not certify corpus-wide mathematical correctness.

## Phase 2 — Make eligibility and review evidence honest

Primary cells: 11, 13, 14, 21, 23, 25.

1. Document exactly which Numina validity labels and question types are accepted. Label the word-problem gate as a metadata gate, not proof of answerability.
2. Record non-exclusive flag counts separately from mutually exclusive row/stage dispositions. Preserve exclusions and reasons.
3. Add solution-completeness diagnostics for answer-only/very short solutions and incomplete multi-part responses. Do not impose an arbitrary minimum length as a mathematical correctness test.
4. Define diagram handling: external/missing images are ineligible; detect Markdown images with alt text, HTML images, and references. Flag inline Asymptote/SVG/TikZ separately and document whether it is accepted as model-readable input or quarantined. Do not call every inline diagram a missing external image.
5. Produce review samples stratified by source and extraction/answer mode: approximately 100 random formatted examples per source, plus risky/ambiguous cases and duplicate-threshold examples.
6. Preserve reviewed files. Record reviewer identity, review date, sample IDs, decisions, policy version, and unresolved cases. Codex can inspect records and document its own review as Codex; never label that work a human sign-off.
7. Replace the always-true `APPLY_REVIEWED_THRESHOLDS` shortcut with validation of saved review evidence. Do not auto-populate approvals without inspecting the cases. Unresolved mandatory reviews must prevent an experimental-ready release while allowing separately labeled diagnostic exports.

Acceptance: the release contains actual review decisions and accurately named flags. Systematic defects discovered in a sample are corrected for their affected cohorts before release.

## Phase 3 — Reserve development and final-test protection inputs

Primary cells: 5 and 19; coordinate with Person 3.

1. Prefer current frozen manifests supplied by Person 3, with problem IDs/text and dataset revisions. Confirm their set roles; do not infer roles from filenames.
2. If they are absent, generate provisional deterministic manifests using the defaults above and record `manifest_origin=person1_generated`. Reserve GSM development only from GSM8K train and MATH development only from MATH train, stratifying MATH by subject/level. Document that Person 3 must adopt/verify these manifests before evaluator comparisons.
3. Never derive development from GSM8K test or MATH500. If MATH train cannot be accessed, report that limitation; a separately labeled Numina-dev diagnostic does not satisfy MATH-dev readiness.
4. Export `dev_fast` as a subset of the full development manifests. Record hashes, counts, revision, split, selection seed, subject/level sampling policy, and answer-storage contract with Person 3.
5. Use question text from all development and reserved final sets for decontamination. Answer labels are not needed in the decontamination input; Person 3 controls scoring labels.
6. Check internal benchmark-set overlaps and resolve any development/test overlap according to the declared policy before freezing manifests.

Acceptance: the protected union includes development plus final tests; exact/near checks identify the scope they actually cover; no test performance has been used to make these data decisions.

## Phase 4 — Correct duplicate grouping and decontamination

Primary cells: 17, 19, 21, 23.

1. Keep global exact deduplication. Preserve numbers, signs, operators, variables, and mathematical expressions when normalizing problems.
2. Use conservative answer-equivalence checks for exact-match groups. Distinguish representations such as `9/3` and `3`, `\\frac{9}{2}` and `4.5`, or `b=2` and `2` where the problem supports equivalence. An unsupported comparison is unresolved, not a proven conflict. Use bounded parsing; do not evaluate arbitrary source strings as code.
3. Index the bounded candidate universe for near matching and document this scope. All exported train, selection, and loss pools must come from the checked universe; if it is expanded, check newly added candidates too.
4. Review candidate thresholds using real pairs and controlled fixtures. ELLIPSE versus BALLOON is a template relative, not automatically a duplicate instance. Make family-level exclusions a separate policy.
5. Record exact and near decisions with evidence, scores, and policy versions. MinHash/LSH candidate retrieval is approximate; a final Jaccard threshold does not establish perfect recall.
6. Maintain a stable final problem-group mapping for the release, including original exact groups, near components, representatives, and quarantine decisions. Apply group-level exclusions to all members.
7. Verify release pools have no retained matches under the implemented evaluation-overlap policy. Describe the result as detected-overlap removal, not proof of zero contamination or absence of pretraining exposure.

Acceptance: group mappings and pair reports are exported; matching fixtures pass; duplicate decisions are reviewed; all pools receive the same evaluation protection.

## Phase 5 — Establish a shared formatting and label contract

Primary cell: 25.

Export an importable helper, e.g. `person1_data_contract.py`, with:

```text
render_prompt(problem)
build_completion(solution, final_answer, extraction_metadata)
encode_example(example, tokenizer) -> input_ids, attention_mask, labels, counts
validate_manifest(path, release_metadata)
```

Requirements:

- Use the agreed base-model prompt serialization; no silent switch to an Instruct chat template.
- Export the exact prompt template, completion policy, EOS policy, tokenizer identifier/revision, max length, and hashes.
- Tokenize the actual concatenated training text. Verify the prompt/target boundary; handle or explicitly quarantine boundary-crossing cases.
- Mask prompt labels to -100; supervise completion and exactly the intended EOS; mask padding by position/attention mask. In particular, do not mask every token with the pad ID if pad and EOS share an ID.
- Calculate `supervised_tokens = count(labels != -100)` on reference encoded examples. Keep prompt, completion, total input, and supervised counts distinct.
- Exclude overlength whole examples, with counts and source/length summaries. Do not silently truncate reasoning to fit.
- Preserve this reference encoder for Person 2. Person 2 still verifies its actual collator/labels before main runs.

Acceptance: reference-label tests validate masking, EOS, boundaries, length handling, and counts without loading model weights. The handoff makes preprocessing counts independently reproducible.

## Phase 6 — Split the clean groups and export training manifests

Primary cells: 27, 29, 33.

1. Assign disjoint groups to the 40K training bank, 10K selection bank, and 1K loss-development pool. Use deterministic source allocation and preserve approximate within-source length distributions.
2. Define common source-specific length-bin edges once on the declared reference population and export them. Do not independently recompute incomparable deciles for each bank or selector arm.
3. Build deterministic nested training prefixes of 500, 1,000, 2,000, and 4,000 rows. `debug_32` should include both sources and varied lengths. State that Person 2 initializes every independent curve run from the original base model.
4. Preserve existing `row_id` and `dataset_id` compatibility, or export aliases `example_id` and `source` with a documented mapping. Include problem, original solution/reference, formatted prompt/completion, answer, group ID, provenance, extraction mode, flags, and token counts. Include tokenizer/template versions per row or through a required release manifest.
5. Export content hashes, ordered-ID hashes, counts, source example/token shares, length summaries, maximum lengths, and reference supervised-token totals for each manifest.
6. Export the planned R1000 manifest and planned seen-group list, clearly marked planned rather than evidence of actual training.
7. Export the selection bank now; it is disjoint from the entire training bank, so creating it does not require waiting for a trained checkpoint. Record parent binding as pending. Person 4 may score this pool only under its chosen frozen parent and scoring contract.

Acceptance: pairwise group intersections across the banks/loss set are empty, each manifest contains unique groups, nesting/order/hashes are verified, quotas and counts reconcile, and all records satisfy the formatting/length/overlap policies.

## Phase 7 — Define Person 2 integration checks without blocking Person 1 completion

Primary cells: 31 and 33.

### Training-label reconciliation

Export a verification-input manifest and a checker that accepts Person 2's `row_id, actual_supervised_tokens` report. Define the expected scope explicitly, such as the 32-row smoke batch followed by all R1000 examples before the parent run. Require unique IDs, exact expected coverage, and one-to-one matching. Reject missing, duplicated, unexpected, or mismatched rows. An empty inner join must never mean counts match. Report actual labels, prompt/EOS/padding handling, and trainer/contract versions.

### Actual parent binding

Export a validator that accepts actual parent run ID/checkpoint reference, training-manifest hash, and actual seen group IDs. Check that the actual seen groups match the actual training manifest and that the selection bank is disjoint. Do not trust a renamed planned template as actual training evidence. If Person 2 trained outside the declared train bank, revalidate freshness and pool eligibility before continuation.

No trained parent or Person 2 verification is required to label Person 1's release `DATA_READY_FOR_TRAINER`. Actual training-label reconciliation and parent binding are required for the corresponding later readiness states.

Acceptance: empty/partial/duplicate verification reports fail; valid complete reports pass; stale or inconsistent parent bindings fail. Integration scripts and exact input/output contracts are included in the release.

## Phase 8 — Package, test, and issue the handoff

Expected package, with paths relative to the release root:

```text
Person1_Full_Run_v2.ipynb
person1_data_contract.py
tests/
configs/person1_data_config.json
configs/formatting_contract.json
configs/length_bin_edges.json
metadata/release_manifest.json
metadata/source_revisions.json
metadata/environment.json
metadata/benchmark_manifests.json
processed/canonical_*.parquet
processed/problem_group_mapping.parquet
processed/row_status_audit.parquet
manifests/train_bank.jsonl
manifests/selection_bank.jsonl
manifests/loss_dev.jsonl
manifests/debug_32.jsonl
manifests/train_n500.jsonl
manifests/train_n1000.jsonl
manifests/train_n2000.jsonl
manifests/train_n4000.jsonl
manifests/*_ids.txt
manifests/*_summary.json
benchmarks/dev_gsm.csv
benchmarks/dev_math.csv
benchmarks/dev_fast.csv
benchmarks/protected_questions.csv
integration/planned_parent_seen_groups_R1000.csv
integration/token_verification_input.jsonl
integration/verify_person2_tokens.py
integration/bind_parent.py
reports/quality_audit.csv
reports/answer_extraction_review.csv
reports/formatted_target_review.csv
reports/duplicate_and_overlap_reports/
reports/review_decisions.json
reports/data_audit_v2.md
reports/regression_test_results.txt
reports/release_validation.json
PERSON2_HANDOFF.md
```

The paths are a contract proposal, not evidence these files already exist. Large canonical artifacts can remain in stable shared storage if the release records their path, size, and checksum. Keep external benchmark answer labels under the evaluator's documented contract.

### Required validation

Run targeted offline fixtures for extraction, target consistency, supported answer equivalence, group splitting, nested prefixes, overlap rules, label masking/EOS, and negative integration cases. Then run a representative pilot through the same code and record exclusions. After those pass, regenerate the full release once if source data and runtime permit. Inspect saved artifacts rather than accepting notebook print statements as certification. Report the commands actually executed, stage outcomes, and any operational failures.

### Readiness states

- `DIAGNOSTIC_ONLY`: limited or incompletely reviewed outputs; suitable only for explicitly labeled debugging with inspected examples.
- `DATA_READY_FOR_TRAINER`: corrected targets, protected splits, reviewed release policies, manifests, reference encoder, checksums, and handoff validation complete; Person 2 can start its smoke/verification work.
- `MAIN_RUN_READY`: Person 2's required training-label checks and Person 3's evaluator/split contract have additionally passed. Person 1 cannot claim this from preprocessing alone.
- `CONTINUATION_READY`: an actual trained parent has additionally been bound and fresh-pool disjointness verified.

Record every prerequisite as passed/pending/failed with evidence paths. A missing downstream prerequisite is a pending dependency, not a fabricated pass. A source-access or review blocker must be visible, and readiness must reflect it.

## PERSON2_HANDOFF.md must contain

1. Release ID and readiness status, immutable shared paths, and checksums.
2. Exact model/tokenizer family and revision, prompt/EOS/loss-mask contract, and Python/package requirements.
3. Load instructions for `debug_32` and each training manifest, using the shared reference helper rather than reconstructing targets with different logic.
4. Reference supervised-token totals by manifest/source, expressly distinguishing them from the trainer's not-yet-verified totals.
5. How to run label reconciliation and the coverage it requires.
6. How to train/reload the 32-example smoke checkpoint; Person 2 owns executing this step.
7. Independent-initialization requirement for the learning curve, and the planned R1000 parent manifest/hash.
8. Selection-bank location, freshness guarantee relative to the train bank, and actual-parent binding procedure.
9. Development/final-test identities and contacts/contracts for Person 3, without implying final tests are development inputs.
10. All exclusions, known limitations, unresolved cases, actual run/test history, and changes from v1.

## Final response required from implementing Codex

Lead with what was implemented and the actual release status. Provide the corrected notebook and handoff package locations. Summarize tests and processing actually completed, material data-count changes, and pending Person 2/3 dependencies. State the next concrete action for Person 2. Do not report historical manifest totals as corrected results or describe planned experiments as completed training.

## Suggested ownership and execution order

| Phase | Person 1 output | Other-person boundary |
|---|---|---|
| 0-2 | Corrected code, target policies, review evidence | A teammate can independently review samples |
| 3-4 | Protected benchmark union and clean groups | Person 3 supplies or adopts frozen development/evaluation inputs |
| 5-6 | Shared encoder, disjoint banks, nested manifests | Person 2 adopts the formatting/label contract |
| 7-8 | Validation scripts and data-ready release | Person 2 verifies actual labels, trains, then binds actual parent |

Person 1 is complete when `DATA_READY_FOR_TRAINER` is supported by saved evidence and the package above. Person 2 then verifies trainer labels, completes the training smoke/reload, evaluates through Person 3's contract, and starts R500/R1000/R2000/R4000 as independent runs. Student scoring, continuation-arm selection, and statistical conclusions remain outside Person 1's implementation scope.

# Person 1 corrected workflow

The project compares REPEAT, RANDOM and FIT continuation selection for mathematical reasoning. Person 1 supplies traceable worked-solution data; Person 2 owns training and actual labels; Person 3 supplies frozen development/test questions and the evaluator's shared answer-equivalence checker. Final research runs need the same corrected inputs, protected development questions and confirmed continuation budget.

`full_run_v1` and the pilot exports are historical, unapproved inputs containing confirmed extraction defects. They have been preserved for comparison. The current notebook has no saved execution outputs: old successful prints would misrepresent the modified code.

## Implemented controls

- Orca extraction reads only the terminal answer block or conclusion. It ignores earlier givens, preserves marked symbolic expressions, rejects ambiguous unmarked expressions, and quarantines unmarked single-number guesses until individually reviewed. It no longer chooses a “novel” number from a multi-number sentence.
- Suspected multi-part questions and answer-only solutions are quarantined under an explicit conservative policy. These are heuristics, not a proof that every retained question is single-part or every derivation is complete. `correctness_status` remains `unknown`.
- Formatting validates an existing terminal marker against the selected target and creates one trailing `Final answer:` block. Conflicting or unverified markers are quarantined before formatting. Targets are not truncated to 500 characters.
- Diagram exclusions include `[asy]`, TikZ, HTML images, Markdown images with alt text, reference-style images and textual missing-diagram cues. Inline diagram code is quarantined because the text-only training/evaluation interface does not render it.
- Exact rational equivalence handles `9/3 = 3`, `\\frac{9}{2} = 4.5` and `b = 2 = 2` using safe, exact arithmetic. Symbolic comparisons use Person 3's shared checker if supplied. Unknown equivalence remains quarantined and is reported separately from proven disagreement.
- Reserved-test and frozen-development inputs are validated and hash checked. Both are included in exact and bounded near-overlap checks. Missing development protection prevents approval.
- The default overlap policy conservatively removes lexical variants at the recorded Jaccard thresholds. Different parameters or words can have different answers. Removal counts do not claim identical benchmark instances or complete template-family removal. A separate parameter-normalized search supplies representative potentially missed matches for review.
- Review files preserve decisions on identical generated inputs and refuse stale evidence. Approval requires structured decisions, reviewer, UTC date, policy version, resolution, independent missed-match notes and hashes of all saved review sheets and release artifacts. Findings are enforced against both training and fresh pools, regardless of similarity threshold. A Boolean cannot substitute for this evidence.
- New runs use unique `full_run_v2_*` folders. Explicit resume checks frozen source/control/configuration/input hashes before writing reports. Changing extraction, seeds, thresholds, revisions or development inputs requires a new run.
- Each manifest includes `prompt_text`, the source solution, original row ID, prompt-specification version and template hash. `prompt_specification.json` defines the exact rendering, tokenizer revision, EOS supervision, masking and field mapping for Person 2.
- Token reconciliation requires nonempty, unique IDs, exactly the corrected R1000 manifest's coverage, a one-to-one merge and positive integer actual counts. Empty, incomplete, duplicate, unexpected or mismatching records fail. Budget B stays provisional until actual `labels != -100` counts match.
- Fresh-pool export requires checkpoint identity, actual parent manifest and matching seen groups. It checks stable IDs, exact problem hashes, current group mapping and bounded parent near-overlap, then requires the entire balanced target size. A template is never parent evidence.

Quality flags describe what is checked: `outside_math_word_problem_metadata_scope` tests the upstream question-type label; `missing_or_sentinel_final_answer` tests missing/`proof`/`notfound` values. Symbolic answers remain permitted. Quality/review flag counts are **nonexclusive**; row disposition counts are **mutually exclusive**.

## Rebuild and verification

Run the notebook in Colab against its configured repository SHAs for the final upstream rebuild. It creates draft manifests, nested subsets, fresh review sheets and reports before the final evidence gate. Complete the sheets for that run; rerun downstream cells with the same unchanged inputs to preserve decisions. Source/target findings marked quarantined or corrected must actually be absent from the selected outputs; correcting content requires a new run.

`rebuild_from_snapshot.py` is now the entrypoint for `streaming_rebuild.py`. By default it processes a **smoke subset of 500 rows per source**, not the million-row corpus. It reconstructs Numina eligibility conservatively from retained v1 validity flags, rechecks overlap against supplied problem texts, regenerates completions, retokenizes using the pinned tokenizer JSON, and exports small nested drafts. `--full` explicitly requests the full snapshot reconstruction. That additional blocker is intentional: v1 omitted some original Numina fields and truncated answer text. Possibly truncated answers are quarantined, and the final run must reload the pinned upstream records. Snapshot reconstruction cannot be a final approved handoff or replace independent mathematical review.

The source stage writes bounded, committed Parquet batches. Checkpoints bind source-file hashes, notebook/control/runner code and options, and validate every committed shard before reuse. Global compact grouping metadata lives in SQLite; full text is fetched only for selected candidate/review IDs. `--resume RUN_ID` reuses completed source rows and the committed catalog. Downstream bounded candidate processing is regenerated deterministically. A 768 MiB memory guard is the default; use `--max-memory-mb` to change the execution budget. Completed 1,000-row and 5,000-row smoke runs observed roughly 199 MiB and 214 MiB peak RSS respectively; full-scale memory use remains unverified.

Local commands, from the workspace root:

```powershell
& 'C:\softwares\pythonAnaconda\python.exe' -m unittest capstone_person1.test_pipeline_controls capstone_person1.test_handoff_gate capstone_person1.test_snapshot_stream capstone_person1.test_snapshot_catalog -v
& 'C:\softwares\pythonAnaconda\python.exe' -m capstone_person1.validate_notebook
& 'C:\softwares\pythonAnaconda\python.exe' capstone_person1\sync_notebook_controls.py
& 'C:\softwares\pythonAnaconda\python.exe' capstone_person1\prepare_tokenizer.py
& 'C:\softwares\pythonAnaconda\python.exe' -u capstone_person1\rebuild_from_snapshot.py --rows-per-source 500
& 'C:\softwares\pythonAnaconda\python.exe' -u capstone_person1\rebuild_from_snapshot.py --rows-per-source 2500
& 'C:\softwares\pythonAnaconda\python.exe' -u capstone_person1\rebuild_from_snapshot.py --resume full_run_v2_20261012T004834Z_c75c286d
& 'C:\softwares\pythonAnaconda\python.exe' -m capstone_person1.verify_smoke_run capstone_person1\runs\full_run_v2_20261012T004834Z_c75c286d
```

The local runner uses the same notebook cells and pinned tokenizer vocabulary/offsets. Windows maps run sequentially to avoid notebook multiprocessing/spawn problems. `vendor` holds the project-local pinned `datasketch==2.0.0`; `cache` is disposable download/computation cache. No model training is run by this notebook.

## Team evidence still needed

**Person 3:** `inputs_from_team/frozen_development.csv`, with `set_id,problem_id,problem`, plus `frozen_development_metadata.json` containing `manifest_sha256`, exact `set_counts`, `benchmark_families: ["gsm8k", "math"]`, `confirmed_by` and an ISO UTC `confirmed_utc`. Use distinct IDs for development and reserved tests; their normalized questions must be disjoint. Supply `equivalence_checker.py` with `mathematically_equal(left, right)` returning `True`, `False` or `None`, using the same implementation as evaluation. Its file hash is frozen in the run.

**Person 1 and independent reviewer:** complete every generated source/extraction/pair/missed-match/formatted-target sheet. Do not fill reviews as passed merely to unblock execution. Use `accepted`, `quarantined` or `corrected_in_new_run` for resolution. Pair classification is `duplicate_instance`, `parameter_or_word_variant`, `template_family` or `unrelated`; `manual_policy_action` explicitly records `keep` or `remove`. A `yes` same-instance decision must classify a duplicate instance. Duplicate instances and parameter/word variants under the current conservative policy require removal; protected removed pairs cannot leave the training row in either exported pool. Review completeness, extraction consistency and mathematical correctness independently. Save `reports/review_approval.json` using the example structure below and the actual review-file/artifact hashes and `run_fingerprint` from that run. Review findings require correction/quarantine before approval.

**Person 2:** `person2_token_verification.csv` with `row_id,actual_supervised_tokens`, covering **exactly corrected R1000**, and `person2_token_verification_metadata.json` containing that manifest's SHA-256, verification-file SHA-256, confirmer/date and `count_definition: "labels != -100 including supervised EOS"`. Use actual training preprocessing/collator output, not Person 1's precomputed counts. Then provide `parent_checkpoint_R1000.json`, `parent_actual_training_manifest_R1000.jsonl` and `parent_seen_groups_R1000.csv`; the actual manifest needs `row_id,problem_group_id,problem` for every training row. Parent metadata includes `checkpoint_id`, `checkpoint_sha256`, `actual_manifest_sha256`, `confirmed_by`, `confirmed_utc`. Its actual-manifest hash must match the corrected agreed R1000 export; a historical v1 parent is rejected.

`handoff_readiness.json` is the approval record for the exact manifest hashes. Exported rows carry their draft-at-export status permanently; they become authorized research inputs only when a matching readiness record says `research_ready: true`. Adding development questions or changing data/control policy requires a new run and new review evidence. Do not reuse v1's reported 333,328 token total after target corrections.

Example approval structure (placeholders are not approval):

```json
{
  "run_fingerprint": "<SHA-256 of this frozen run configuration>",
  "reviewer": "<independent reviewer>",
  "reviewed_utc": "<ISO timestamp with timezone>",
  "policy": "<copy the complete REVIEW_POLICY object for this run>",
  "review_file_sha256": {
    "orca_inspection_25.csv": "<actual file hash>",
    "numina_inspection_25.csv": "<actual file hash>",
    "orca_answer_extraction_review_25.csv": "<actual file hash>",
    "near_duplicate_review_sample.csv": "<actual file hash>",
    "evaluation_near_review_sample.csv": "<actual file hash>",
    "missed_match_review.csv": "<actual file hash>",
    "formatted_examples_review_20.csv": "<actual file hash>"
  },
  "artifact_sha256": {
    "prompt_specification.json": "<actual file hash>",
    "debug_32_full.jsonl": "<actual file hash>",
    "train_n1000_full.jsonl": "<actual file hash; include EVERY exported training size>",
    "fresh_candidate_pool_R1000_full.jsonl": "<actual file hash>"
  },
  "missed_match_audit": {
    "notes": "<independent search method, findings, limitations and resolved errors>"
  }
}
```

The pipeline does not certify dataset-wide mathematical accuracy or unknown pretraining contamination. A successful structural audit must not be reported as completed human review, verified training labels or actual training.

> Historical milestone: execution commands and paths describe that milestone. Use [the current package README](../../README.md) for today's layout and commands.

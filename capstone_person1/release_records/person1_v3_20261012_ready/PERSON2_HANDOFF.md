# Person 1 data handoff

Release `person1_v3_20261012_ready` — **DATA_READY_FOR_TRAINER**. All paths below are relative to this release. Checksums are in `metadata/release_manifest.json`. Source revisions and verified original files are in `metadata/source_revisions.json`. The complete original source fields are preserved per canonical row.

Use Qwen/Qwen2.5-0.5B base and tokenizer revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. Install `code/requirements.txt` into an isolated Python environment, then add `code/` to `PYTHONPATH`. The notebook calls the same named stages as the CLI; no notebook cell-number execution is required for this release.

Load JSON lines from `manifests/debug_32.jsonl`, then `train_n500`, `train_n1000`, `train_n2000`, and `train_n4000`. Import `encode_example` and `pad_encoded` from `capstone_person1.person1_data_contract`. Call `encode_example(tokenizer, row['problem'], row['solution'], row['final_answer'])`; use its input IDs, attention mask and labels directly. The exact prompt ends in `Solution:\n`. No chat template. Combined-text tokenization uses `add_special_tokens=False` and checked character offsets. Prompt labels are -100; completion and exactly one appended EOS (151643, `<|endoftext|>`) are supervised. Padding shares that ID: mask by position/attention, preserving the real EOS label. Full details are in `configs/formatting_contract.json`. Evaluation supplies only `render_prompt(problem)`.

| Manifest | Rows | Reference supervised tokens including EOS |
|---|---:|---:|
| train_bank | 40,000 | 12,766,973 |
| selection_bank | 10,000 | 3,191,599 |
| loss_dev | 1,000 | 317,357 |
| train_n500 | 500 | 160,220 |
| train_n1000 | 1,000 | 318,845 |
| train_n2000 | 2,000 | 638,802 |
| train_n4000 | 4,000 | 1,277,079 |
| train_n8000 | 8,000 | 2,554,989 |
| train_n16000 | 16,000 | 5,114,372 |
| train_n40000 | 40,000 | 12,766,973 |
| debug_32 | 32 | 10,461 |

These totals come from reference labels; the trainer has not yet supplied its actual-label verification. Verify **all** debug_32 IDs first, then all R1000 IDs. Export CSV `row_id,actual_supervised_tokens` and metadata with trainer/contract versions, exact manifest SHA-256, actual prompt/EOS/padding checks and confirmation identity/time. Run `python -m capstone_person1.integration_checks tokens --release RELEASE --scope debug_32 --actual ACTUAL.csv --metadata ACTUAL.json --output OUTSIDE_RELEASE.json`. Repeat for `train_n1000`. Empty, partial, duplicated, unexpected or mismatching rows fail. Example wrappers and input contracts are in `integration/`.

Person 2 next executes the 32-example QLoRA training/save/reload smoke test and records its training/config/checkpoint hashes. Every independent R500/R1000/R2000/R4000 learning-curve run starts from the original base model. `integration/planned_parent_seen_groups_R1000.csv` describes the planned parent, not completed training. After actual training, run the parent checker with the actual run/checkpoint identity, actual R1000 manifest and seen groups. It verifies the precise planned manifest hash and selection-bank freshness. Training outside this manifest requires a new coordinated data version.

`manifests/selection_bank.jsonl` is disjoint from the **entire** training bank and `loss_dev`. Its creation requires no trained parent. Actual parent binding remains pending; Person 4 owns student-fit scoring and selection arms. `loss_dev` is held-out training loss data, separate from Person 3's benchmark evaluation sets.

Protected question entries: 1,819. Development coverage: **pending**; Person 3 final-test role confirmation: **pending**. Person 3 supplies and freezes its own development/fast-development question manifests and evaluator contract. Known GSM8K test and MATH500 questions are protected now. When new evaluation questions arrive, import them and regenerate/recheck a new version before comparative runs. No final-test scores were used here.

See `reports/data_audit_v3.md`, `reports/review_decisions.json`, extraction/format reviews, whole-example encoding exclusions, exact/near pair decisions, complete row-status/group maps, and `reports/release_validation.json`. Similarity retrieves candidates; it never automatically certifies an instance. Changed-number template relatives remain distinct. All unresolved near-instance groups are quarantined. MinHash/LSH has approximate recall; detected-overlap removal does not prove zero contamination or rule out pretraining exposure. Mathematical correctness is **unknown** unless separately reviewed; source validity labels and structural consistency are not correctness proofs. Reviews identify Codex explicitly and do not claim independent human sign-off.

Compared with v1: corrected terminal targets, complete upstream Numina labels/answers, case-preserving global exact grouping, conservative reviewed instance policy, actual label-derived budgets, common length bins, three disjoint banks, independent readiness gates, and explicit downstream evidence contracts. Historical v1 manifests are not substituted for these outputs. Pilot scope: full pinned sources.

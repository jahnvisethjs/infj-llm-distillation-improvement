# Person 1 v4 training data handoff

**Release:** `person1_v4_20261016_decontaminated`

**Prepared:** 16 October 2026

**Original release:** `person1_v3_20261012_ready`

This release provides one ordered 40,000-row training bank with detected evaluation overlaps and internal duplicate questions removed. It preserves the original source balance, length bins, example schema, tokenization contract, and nested training sizes.

**302 rows were replaced; 39,698 original rows retain their contents and positions.** All replacements are existing unused examples from the original pinned source data. No new question or solution text was generated or edited.

## Download the release

[Download the dataset release and handoff files from Google Drive](https://drive.google.com/drive/folders/14dz8XfcRNcrwGs6CSpL3C-kolqvBtH7n?usp=sharing).

Download the complete release folder, or its ZIP archive, and preserve the folder structure. The expected local destination, relative to the repository root, is:

```text
capstone_person1/releases/person1_v4_20261016_decontaminated/
```

The validation results below describe the generated release. Verify the downloaded copy against its supplied `SHA256SUMS.txt` to confirm that it matches. The Drive link is the distribution location supplied for this handoff; its uploaded contents and sharing permissions have not been independently verified.

## What changed

| Change | Unique training rows replaced |
|---|---:|
| Protected evaluation-question overlaps, including reviewed similarity cases | 139 |
| Internal duplicate questions | 163 |
| **Total** | **302** |
| Original rows retained unchanged | **39,698** |

For protected matches, the exclusion covers the entire original `problem_group_id`. For internal duplicate questions, the earlier occurrence is retained and the later occurrence is replaced. The 302 removed rows belong to 302 distinct original training groups.

The original supplied checker report was reproduced, including all 139 flagged pairs and the two small-prefix exact duplicates at positions 97 and 311. Its original `EXACT` and `LIKELY_SAME` flags identify **86 distinct training rows and groups**. Reviewing the similarity cases and applying stronger formatting normalization increased the evaluation-related exclusions to 139 training rows.

The review includes paraphrases, inverse versions of the same problem, complete evaluation questions embedded in longer examples, formatting-only copies, and ambiguous extensions. Genuinely different problems sharing a template are retained when the review establishes different constraints and answers. Every decision is recorded in `reports/review_decisions.csv`.

### Replacements within each training size

| Training rows | Evaluation overlaps replaced | Internal duplicates replaced | Total replaced | Original rows unchanged | Added beyond previous size |
|---:|---:|---:|---:|---:|---:|
| 500 | 2 | 0 | **2** | 498 | 2 |
| 1,000 | 2 | 0 | **2** | 998 | 0 |
| 2,000 | 3 | 0 | **3** | 1,997 | 1 |
| 4,000 | 8 | 1 | **9** | 3,991 | 6 |
| 8,000 | 21 | 4 | **25** | 7,975 | 16 |
| 16,000 | 56 | 22 | **78** | 15,922 | 53 |
| 40,000 | 139 | 163 | **302** | 39,698 | 224 |

**These counts are cumulative.** The 1K file includes the first 500 rows, the 2K file includes the first 1K, and so on. A replacement at position 97 appears in every training size. Do not add the “Total replaced” column across sizes.

Positions refer to **1-based JSONL record numbers**:

- **500 and 1K:** 97 and 311.
- **2K:** 97, 311, and 1,736.
- **4K:** 97, 311, 1,736, 2,175, 2,393, 2,939, 3,757, 3,792, and 3,893.

Position 1,736 is an inverse version of a development question identified during similarity review. This explains the three replacements in 2K despite the original headline showing two exact duplicates. Position 3,792 duplicates position 152; position 152 is retained.

The complete list of all 302 positions, removed and replacement IDs, group IDs, sources, length bins, and token counts is in `reports/replacement_log.csv`.

## How replacements were selected

The original processing-provenance archive was unavailable. Its complete row-status audit and the exact pinned raw source files were available, so **all 44,961 original `checked_unused` representatives were reconstructed** using the original canonicalizers, final group mappings, and reference encoder. Forty original training examples were reproduced from raw sources as a schema and encoding check.

The sources remain `microsoft/orca-math-word-problems-200k` and `AI-MO/NuminaMath-1.5` at the revisions recorded in `metadata/source_revisions.json`. The reconstruction covers the original unused population; it does not claim to recover the missing full encoded archive byte-for-byte.

For each excluded position, in increasing order:

1. Use an unused candidate from the original approved universe.
2. Reject protected evaluation overlaps and excluded problem groups. Candidate overlap review rejected 178 reconstructed groups.
3. Match the removed example’s source (`orca` or `numina`) and its original length bin.
4. Minimize the absolute difference in supervised-token count; break ties by `row_id`.
5. Check occupied IDs, groups, and normalized questions, then place the replacement at the removed example’s position.

Development, final-test, selection-bank, and loss-dev examples were not used as replacements. All unaffected training rows remain in place. Each training size is then derived from the first N records of the same clean master bank.

## Protected evaluation datasets

The five files were obtained from the `person_3-evaluation` branch at commit `64da840623b956c5cea2237367176b6de057e5c1`, together with frozen role and split metadata.

| File | Rows | Purpose |
|---|---:|---|
| `dev_gsm.jsonl` | 500 | Full GSM8K development set |
| `dev_math.jsonl` | 500 | Full MATH development set |
| `dev_fast.jsonl` | 200 | 100 GSM + 100 MATH questions from the full development sets |
| `test_gsm.jsonl` | 1,319 | Final GSM8K test |
| `test_math500.jsonl` | 500 | Final MATH500 test |

There are **2,819 unique protected questions**. `dev_fast` is an exact subset of the full development sets and adds no new unique questions. Evaluation contents were preserved.

### Detected overlaps before and after

Each entry below is **before → after**, counting evaluation questions classified `EXACT` or `LIKELY_SAME` by the enhanced checks. Normalization includes Unicode punctuation, case, whitespace, harmless LaTeX, equivalent fraction commands, display delimiters, and Markdown links around terms.

| Training rows | GSM dev | MATH dev | Fast dev | GSM test | MATH500 test |
|---:|---:|---:|---:|---:|---:|
| 500 | 2 → 0 | 0 → 0 | 1 → 0 | 0 → 0 | 0 → 0 |
| 1,000 | 2 → 0 | 0 → 0 | 1 → 0 | 0 → 0 | 0 → 0 |
| 2,000 | 2 → 0 | 0 → 0 | 1 → 0 | 0 → 0 | 0 → 0 |
| 4,000 | 4 → 0 | 0 → 0 | 1 → 0 | 0 → 0 | 0 → 0 |
| 8,000 | 10 → 0 | 1 → 0 | 4 → 0 | 0 → 0 | 0 → 0 |
| 16,000 | 33 → 0 | 2 → 0 | 13 → 0 | 0 → 0 | 2 → 0 |
| 40,000 | 81 → 0 | 5 → 0 | 19 → 0 | 0 → 0 | 3 → 0 |

Overlap headlines count evaluation questions; replacement counts count training rows and also include reviewed similarity cases and internal duplicates. One evaluation question can match several training rows. The original checker’s reports are preserved separately alongside the enhanced results.

Two remaining similarity flags are retained as different problems:

- **Position 6,075:** a six-hour limousine hire becomes eight hours, and a new $150 tuxedo expense is added.
- **Position 14,876:** the spool length and number and lengths of candle wicks change, producing a different partition problem.

Their full explanations are in `reports/overlap_after/retained_similarity_reviews.json`. There are no unresolved flagged cases.

## Training files and token budgets

Use the required `train_nN.jsonl` for an experiment. Every JSONL has a corresponding `*_ids.txt` and `*_summary.json`. The summary records its manifest SHA-256, ordered-ID hash, token totals, source counts, and length-bin counts.

| Manifest | Rows | Supervised tokens including EOS |
|---|---:|---:|
| `train_bank.jsonl` | 40,000 | 12,766,972 |
| `train_n500.jsonl` | 500 | 160,220 |
| `train_n1000.jsonl` | 1,000 | 318,845 |
| `train_n2000.jsonl` | 2,000 | 638,802 |
| `train_n4000.jsonl` | 4,000 | 1,277,079 |
| `train_n8000.jsonl` | 8,000 | 2,554,988 |
| `train_n16000.jsonl` | 16,000 | 5,114,371 |
| `train_n40000.jsonl` | 40,000 | 12,766,972 |

`train_bank.jsonl` and `train_n40000.jsonl` are byte-identical. Prefixes reuse bank rows and are not additional disjoint data.

Additional manifests retain their existing roles: `debug_32.jsonl` is the 32-row integration prefix; `selection_bank.jsonl` contains 10,000 selection examples; `loss_dev.jsonl` contains 1,000 held-out loss examples. The selection and loss-dev JSONLs are copied byte-for-byte from the original release and remain disjoint from the new training bank.

301 replacements match the removed row’s supervised-token count exactly. At position 7,150 the count changes from 1,424 to 1,423. The full-bank total decreases by one token; the 500/1K/2K/4K totals remain unchanged.

### Preserved training contract

- **Source balance:** 50% Orca and 50% Numina in every training prefix; 20,000 rows per source at 40K.
- **Length bins:** original source-specific edges and counts; at 40K, 2,000 rows in each of ten bins for each source.
- **Ordering:** unaffected positions, original ordering, split seed 1729, and ordering seed 42 are preserved. Replacement selection uses the deterministic rule above.
- **Tokenizer:** `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- **Maximum input length:** 2,048 tokens, with no truncation of reasoning or answers.
- **Formatting and labels:** original plain-text prompt, combined prompt/completion encoding, prompt and padding labels `-100`, and exactly one supervised appended EOS. EOS/PAD ID remains `151643`.

## Release contents

All paths below are relative to `person1_v4_20261016_decontaminated/` in the download.

| Path | Contents |
|---|---|
| `manifests/` | Training bank, all prefixes, debug, selection and loss-dev JSONLs, IDs, and recalculated summaries |
| `reports/replacement_log.csv` | All 302 replacement positions and provenance |
| `reports/review_decisions.csv` | Individual overlap and duplicate decisions with reasons |
| `reports/overlap_before/` and `reports/overlap_after/` | Original and enhanced checker results and retained-similarity explanations |
| `reports/internal_duplicate_check.json` | Internal uniqueness and external-bank disjointness results |
| `reports/nesting_validation.json` | Exact record and byte-prefix checks |
| `reports/source_and_length_balance.json` | Source quotas and original length-bin counts |
| `reports/token_contract_validation.json` | Reference re-encoding of the complete training bank |
| `reports/automated_tests.json` and `.txt` | Automated test results and log |
| `reports/independent_release_validation.json` | Independent manifest-release validation and repeated replacement selection |
| `reports/input_integrity.json` | Original release and frozen evaluation input hashes |
| `inputs/`, `benchmarks/`, `configs/`, `metadata/` | Frozen inputs, protected questions, contracts, source pins, reconstruction records, and artifact metadata |
| `processed/encoded_candidates.parquet` | Intermediate reconstruction of the 44,961 original unused candidates, including candidates rejected during overlap review |
| `code/`, `rebuild.sh`, `README.md`, `SHA256SUMS.txt` | Implementation snapshot, recorded rebuild command, release explanation, and checksums |

**Use the JSONLs in `manifests/` for training.** The intermediate candidate Parquet file is not a training manifest. Keep IDs, summaries, configuration, metadata, and reports with the dataset when sharing it.

## Verification and use

The recorded results are:

- All **95 automated tests passed**, with no failures, errors, or skipped tests.
- All **40,000 training-bank rows were reference re-encoded** and checked against the original schema and token contract.
- All prefixes have their exact required row counts and are exact record and byte prefixes of one bank.
- No duplicate row ID, example ID, problem-group ID, or normalized question remains within the training bank.
- Original source quotas and source-by-length-bin counts are preserved; selection and loss-dev remain disjoint.
- Both the original checker and enhanced checks report zero remaining `EXACT`/`LIKELY_SAME` overlaps across their declared scopes. The enhanced scope includes all five evaluation files.
- The release validator passed in explicit `--manifests-only` mode and independently reproduced the replacement choices.
- Original release and evaluation contents were unchanged. No commit or push was performed during the rebuild.

After downloading, verify the files from inside the release folder. On Linux or Colab:

```bash
sha256sum -c SHA256SUMS.txt
```

On macOS:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

For a quick release-identity check, the new `train_n500.jsonl` SHA-256 is:

```text
86cb3168de93a1d2666fd6dc2532fe66d3981dbf8561858f26c82debd699472e
```

Record each training run’s **new manifest SHA-256** and supervised-token total. Complete the trainer’s actual-label and smoke-test checks using the shared encoder. Models trained on the old manifests require training with the new manifests to reflect these data changes.

### Rebuild and validator prerequisites

The implementation snapshot includes `capstone_person1/decontamination.py`, `capstone_person1/scripts/rebuild_decontaminated.py`, the updated release validator, and eight new regression tests. Use these with a matching Person 1 repository checkout and its pinned `requirements-person1.txt` environment.

The recorded `rebuild.sh` contains the original workspace paths. On another machine, adapt the input paths before rebuilding. A rebuild requires the original immutable v3 release and row-status audit, pinned raw sources, the pinned tokenizer snapshot, frozen Person 3 inputs, and the saved review inputs. The reconstructed Parquet file covers the original unused candidates; it is not the missing complete processing-provenance archive.

The downloaded package supports checksum verification on any machine. The full inherited-provenance validator additionally reads the original release and evaluation locations recorded in `metadata/release_manifest.json` and `reports/input_integrity.json`. Its recorded command, run from the original repository workspace, is:

```bash
python -m capstone_person1.scripts.validate_release \
  --release capstone_person1/releases/person1_v4_20261016_decontaminated \
  --manifests-only
```

The explicit mode validates this manifest derivative against inherited original provenance. Complete canonical/shard artifacts were not repackaged, and full-source canonicalization was not repeated. The saved validation covers all bank records; the independent validator did not separately re-encode every repeated prefix.

## Interpretation limits

Zero detected overlap describes the executed checkers, reviewed pairs, and documented normalization. Approximate similarity retrieval cannot guarantee exhaustive semantic recall. Source validity labels and structural answer consistency also do not independently certify every example’s mathematical correctness.

The release is a verified data handoff. Actual trainer-label reconciliation, model training, trained checkpoints, and downstream evaluation results remain separate work.

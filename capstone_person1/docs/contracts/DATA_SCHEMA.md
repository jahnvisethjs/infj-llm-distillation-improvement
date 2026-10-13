# Person 1 data schema and provenance

Load examples from a declared release manifest, not by filtering canonical `eligible` alone. Canonical eligibility is the initial structural/metadata check; exact conflicts, evaluation protection, near-match quarantine, encoding limits, review decisions and bank allocation follow it. `processed/row_status_audit.parquet` is the authoritative final disposition of every source row.

| Field | Meaning |
|---|---|
| `row_id`, `example_id` | Same stable SHA-256 ID: upstream dataset ID, pinned revision, train split and original global row position |
| `dataset_id`, `source` | Same short source label: `orca` or `numina`; full Hugging Face identities are in source metadata |
| `original_row_id` | Zero-based train row position in the recorded concatenation order of pinned Parquet files |
| `dataset_revision` | Immutable upstream dataset commit |
| `problem_raw`, `solution_raw` | Original upstream strings, preserved before cleanup |
| `problem`, `solution` | NFC normalization and whitespace cleanup; mathematical variable case/superscripts remain intact; no reasoning truncation |
| `upstream_fields_json` | Complete original upstream record, including provided answers, validity metadata and source taxonomy |
| `provided_answer_raw` | Original Numina answer; never truncated to 500 characters |
| `final_answer` | Supported comparison value consistent with terminal solution evidence; numeric percentages may be normalized as ratios. The formatted completion preserves terminal percentage notation. Not an independent mathematical correctness certificate |
| `answer_extraction_json`, evidence fields | Versioned extraction mode, candidate, confidence, ambiguity, basis and evidence spans; offsets refer to **canonical `solution`**, not raw source bytes |
| `quality_flags`, `review_flags` | Nonexclusive flags; their totals can exceed row totals |
| `correctness_status` | `unknown` unless separately established; source validity labels do not establish reasoning correctness |
| `solution_completeness_status` | `unknown`; suspected answer-only solutions are excluded conservatively |
| `problem_group_id` | Canonical records begin with exact groups. Release manifests and final mapping use the final instance-group ID |
| `group_representative_id` | Deterministic representative of a final checked group |
| `split`, `eligible` in canonical records | Initial canonical state (`candidate`/`quarantine`). Later exclusions appear in the final row-status map |
| `split`, `canonical_split` in released manifests | Actual bank membership (`train_bank`, `selection_bank`, `loss_dev`) and preserved initial canonical state; prefixes retain `train_bank` membership |
| `prompt_text`, `completion` | Exact shared plain-text serialization, without the literal EOS string; the encoder appends the EOS token ID |
| `prompt_tokens`, `completion_tokens`, `eos_tokens` | Separate reference counts; exactly one appended EOS |
| `supervised_tokens` | Actual reference `sum(label != -100)`; completion plus one EOS |
| `total_tokens` | Combined nonpadding input length, including EOS; at most 2,048 |
| `completion_start` | Token boundary verified from combined-text offsets |
| `length_bin` | Bin from shared source-specific edges on the declared checked encodable reference population |
| `tokenizer_revision` | Exact pinned 0.5B tokenizer revision; another model's counts require a compatibility check |

The full group map includes `exact_group_id`, final `problem_group_id`, `representative_id` and `final_status`. Group-level exclusions propagate to all original exact members. Outside-universe rows receive global exact checks, but are not claimed to have received the bounded near checks or reference encoding. All exported banks come from the same checked universe.

`train_n500/1000/2000/4000` and `debug_32` are ordered prefixes of `train_bank`, not additional disjoint data. `selection_bank` and `loss_dev` are disjoint from the entire train bank and from one another. The planned R1000 seen-group CSV is a plan, not actual training evidence.

Source-specific bin edges, ordered-ID/content hashes, source row/token shares and reference token totals accompany every manifest. Final statuses distinguish initial quality quarantine, exact conflicts, detected protected overlaps, duplicate alternatives, near-review quarantine, selected banks and checked unused/outside-universe examples.

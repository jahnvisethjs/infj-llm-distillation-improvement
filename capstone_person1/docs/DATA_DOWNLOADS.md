# Person 1 dataset archive inventory

The complete verified data bundle, pinned raw sources and tokenizer assets are packaged locally. They have **not yet been published** as GitHub Release downloads. This page records the intended archive layout and verified checksums so publication status is not confused with packaging status.

Planned release tag: `person1-data-v3-20261012`.

| Archive | Download size | Contents |
|---|---:|---|
| `person1-v3-01-training-and-handoff.zip` | 301.7 MiB | Training/selection/loss JSONL, all prefixes, row/group maps, code, reviews and handoff |
| `person1-v3-02-canonical-orca.zip` | 388.4 MiB | All Orca canonical rows, resumable shards and checkpoints |
| `person1-v3-03-canonical-numina.zip` | 1251.8 MiB | All Numina canonical rows in one Parquet file |
| `person1-v3-04-numina-canonical-shards.zip` | 1252.6 MiB | Numina canonical shards and resume checkpoints |
| `person1-v3-05-processing-provenance.zip` | 624.0 MiB | Exact-group catalog, checked candidates and reference-encoded candidates |
| `person1-v3-06-pinned-upstream-sources.zip` | 518.5 MiB | Original pinned upstream Parquet files, inventory and dataset cards |
| `person1-v3-07-pinned-tokenizer.zip` | 2.2 MiB | Pinned Qwen2.5-0.5B tokenizer assets; no model weights |

## Start training or selection

Clone the repository, then obtain and extract **01-training-and-handoff** into its root. All paths inside the ZIP begin with `capstone_person1/`. The training release appears at `capstone_person1/releases/person1_v3_20261012_ready/`.

Install `requirements-person1.txt` in an isolated environment. Obtain/extract **07-pinned-tokenizer** for offline reference encoding, or fetch that exact tokenizer revision from Hugging Face. The training/selection/loss data and handoff are usable without obtaining the entire original upstream corpus.

```python
import json
from pathlib import Path

release = Path("capstone_person1/releases/person1_v3_20261012_ready")
with (release / "manifests/train_bank.jsonl").open(encoding="utf-8") as stream:
    first_example = json.loads(next(stream))
print(first_example["row_id"], first_example["problem"])
```

Use `PERSON2_HANDOFF.md` and the shared encoder. Training data is 40,000 groups; selection is 10,000 fresh groups; held-out loss is 1,000 groups. Prefixes reuse training-bank rows and are not additional unique data.

## Obtain every dataset and processing record

Obtain **all seven ZIP files**, then extract each into the same repository root. Ordinary ZIP extraction recreates the original folder structure; the archives are independently readable and are not split pieces of one ZIP. Do not rename files inside the release.

| Folder after extraction | Use |
|---|---|
| `capstone_person1/releases/person1_v3_20261012_ready/manifests/` | Ready-to-load train/selection/loss JSONL, nested prefixes, IDs and token summaries |
| `.../processed/canonical_orca.parquet`, `canonical_numina.parquet` | Complete cleaned canonical corpus with original fields and provenance |
| `.../processed/canonical_orca/`, `canonical_numina/` | Resumable canonical shards/checkpoints |
| `.../processed/row_status_audit.parquet`, `problem_group_mapping.parquet` | Final disposition and group membership for every source row |
| `.../processed/global_exact_catalog.sqlite`, `checked_candidates.parquet`, `encoded_candidates.parquet` | Original exact/near/encoding stages used to build the released banks |
| `.../reports/`, `metadata/`, `configs/` | Reviews, audits, pins, configuration and complete release checksums |
| `.../integration/`, `PERSON2_HANDOFF.md` | Actual-label and actual-parent integration inputs and instructions |
| `capstone_person1/sources/` | Complete original pinned upstream files and source inventory |
| `capstone_person1/cache/huggingface/` | Exact pinned tokenizer assets |

Verify each ZIP against [SHA256SUMS.txt](../release_records/person1_v3_20261012_ready/SHA256SUMS.txt). [DATA_DOWNLOADS.json](../release_records/person1_v3_20261012_ready/DATA_DOWNLOADS.json) records every packaged file's original SHA-256 and size. Every archived file was decompressed and checked byte-for-byte after packaging. With all release archives extracted, independently validate the complete bundle:

```powershell
python -m capstone_person1.scripts.validate_release --release capstone_person1/releases/person1_v3_20261012_ready
```

The full release remains immutable and DATA_READY_FOR_TRAINER. Actual training-label checks, trained checkpoints and Person 3 evaluation freezes are later inputs. Data/source cards retain the upstream license declarations and provenance. Historical failed/draft releases and local dependency installations are not distributed as experiment inputs.

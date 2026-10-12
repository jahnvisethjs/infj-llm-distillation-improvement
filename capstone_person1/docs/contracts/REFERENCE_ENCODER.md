# Shared prompt and reference labels

Implemented for Qwen2.5-0.5B base at tokenizer revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. The actual cached `Qwen2TokenizerFast` is verified in [REFERENCE_CONTRACT_VERIFICATION.json](../../evidence/REFERENCE_CONTRACT_VERIFICATION.json); verified EOS/PAD values and asset hashes are in [configs/formatting_contract.json](../../configs/formatting_contract.json).

This helper loads tokenizer assets only. It does not load model weights or establish that the worked reasoning is mathematically correct. The corrected full data release remains pending.

## Person 2: reference encoding

Run from the repository root with the notebook's pinned dependencies installed:

```python
from capstone_person1.person1_data_contract import (
    load_tokenizer, encode_example, pad_encoded,
)

tokenizer = load_tokenizer(cache_dir="capstone_person1/cache/huggingface")
encoded = encode_example(
    tokenizer,
    problem=row["problem"],
    solution=row["solution"],
    verified_final_answer=row["final_answer"],
    max_total_tokens=2048,
)
batch_row = pad_encoded(encoded, length=2048)
input_ids = batch_row["input_ids"]
attention_mask = batch_row["attention_mask"]
labels = batch_row["labels"]
supervised_tokens = sum(label != -100 for label in labels)
```

The encoder tokenizes combined prompt/completion text with `add_special_tokens=False`, verifies offsets, masks prompt labels, and appends one supervised EOS. Padding uses its position and attention mask, preserving the real EOS label when PAD and EOS share ID 151643. Overlength examples raise `ContextLengthError`; no reasoning or answer is truncated. Unsupported or unproven terminal targets require correction/review rather than a guessed target.

Person 2 must compare its actual training collator's labels/counts to this reference and perform its own training/save/reload smoke test. No actual training-label reconciliation or trained checkpoint is claimed here.

## Person 3 and Person 4: identical serialization

```python
from capstone_person1.person1_data_contract import render_prompt

prompt = render_prompt(problem)
```

This returns the exact agreed prompt ending with `Solution:\n`. Evaluation receives only that prompt; the model generates the completion and EOS. Person 4 uses the same encoder for student-fit scoring. No chat template is applied.

## Manifest verification

`validate_manifest_record(record, tokenizer)` re-encodes a row and checks saved prompt/completion text and reference count fields. `validate_manifest(path, release_metadata, tokenizer)` additionally verifies the JSONL content hash, nonempty row count, unique row/group IDs and aggregate supervised-token budget. Metadata requires `tokenizer_id`, `tokenizer_revision`, `manifest_sha256`, `rows` and `supervised_tokens`.

Verification of diagnostic exports is not release approval. Person 1 still needs final source reload, policy/review evidence, three disjoint banks and release packaging. Load and verify the corresponding pinned tokenizer before reusing counts in any 1.5B experiment.

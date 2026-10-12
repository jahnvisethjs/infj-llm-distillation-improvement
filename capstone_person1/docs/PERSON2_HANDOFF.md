# Person 2: training integration

This is a diagnostic integration handoff. You can implement and smoke-test the loader/collator now. Main experiment data is pending; the fixture is not R1000 or a final training bank.

Read [the reference encoder contract](contracts/REFERENCE_ENCODER.md). Use the pinned Qwen2.5-0.5B fast tokenizer, combined prompt/completion encoding, completion-only labels, one supervised EOS and positional padding. EOS and PAD share ID 151643; preserve the real EOS label. Do not truncate overlength solutions or apply a chat template.

After the setup commands in [the package README](../README.md), encode an included fixture row:

```python
import json
from pathlib import Path
from capstone_person1.person1_data_contract import load_tokenizer, encode_example, pad_encoded

tokenizer = load_tokenizer(cache_dir="capstone_person1/cache/huggingface", local_files_only=True)
path = Path("capstone_person1/diagnostics/integration_smoke/debug_32_pilot.jsonl")
row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
encoded = encode_example(tokenizer, row["problem"], row["solution"], row["final_answer"])
batch_row = pad_encoded(encoded, length=2048)
supervised_tokens = sum(label != -100 for label in batch_row["labels"])
```

Compare your actual collator's input IDs, attention masks, labels and supervised-token counts against this encoder before running a training/save/reload smoke test. No training has been executed by this handoff.

For the later corrected release, return one actual label-derived count per training row, bound to the exact manifest/checksum. Templates are in `inputs_from_team/person2_token_verification_TEMPLATE.csv` and `person2_token_verification_metadata_TEMPLATE.json`. Parent evidence uses `parent_checkpoint_R1000_TEMPLATE.json` and `parent_seen_groups_R1000_TEMPLATE.csv`; provide the actual checkpoint identity, training-manifest hash and groups seen. Leave these templates unsigned until real evidence exists.

The verified diagnostic token counts do not finalize the R1000 continuation budget. Person 1 data readiness must be assessed separately from later trainer/parent integration and experiment readiness.

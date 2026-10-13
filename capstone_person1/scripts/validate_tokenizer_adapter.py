"""Check the pinned local tokenizer adapter against the preserved v1 token counts.

This verifies preprocessing agreement, not actual labels under Person 2's training code.
"""
import json
from pathlib import Path

from capstone_person1.pipeline_controls import PROMPT_TEMPLATE
from capstone_person1.tokenizer_adapter import SnapshotTokenizer

root = Path(__file__).resolve().parents[1]
manifest = root / 'runs/full_run_v1/outputs/manifests/train_n1000_full.jsonl'
records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
tokenizer = SnapshotTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987')
prompts = [PROMPT_TEMPLATE.format(problem=row['problem']) for row in records]
prompt_ids = tokenizer(prompts)['input_ids']
full_ids = tokenizer([prompt + row['completion'] for prompt, row in zip(prompts, records)])['input_ids']
for row, prompt, prefix, full in zip(records, prompts, prompt_ids, full_ids):
    assert full[:len(prefix)] == prefix, row['row_id']
    assert row['prompt_tokens'] == len(prefix), row['row_id']
    assert row['supervised_tokens'] == len(full) - len(prefix) + 1, row['row_id']
    assert row['total_tokens'] == len(full) + 1, row['row_id']
assert tokenizer.eos_token_id is not None
print(f'Pinned tokenizer adapter matches all {len(records)} preserved R1000 preprocessing counts ({sum(r["supervised_tokens"] for r in records):,} tokens).')

"""Download only tokenizer assets at the historical pinned revision into this workspace."""
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
revision = '060db6499f32faf8b98477b0a26969ef7d8b9987'
for filename in ['tokenizer.json', 'tokenizer_config.json']:
    path = hf_hub_download('Qwen/Qwen2.5-0.5B', filename, revision=revision,
                           cache_dir=root / 'cache' / 'huggingface')
    print(filename, path)

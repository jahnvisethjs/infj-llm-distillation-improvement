"""Lightweight adapter for the pinned tokenizer JSON; no model or training imports."""
import json
from pathlib import Path

from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parent


class SnapshotTokenizer:
    @classmethod
    def from_pretrained(cls, model_id, revision, **kwargs):
        if model_id != 'Qwen/Qwen2.5-0.5B':
            raise ValueError('Local adapter supports only the configured Qwen tokenizer')
        snapshot = PROJECT / 'cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots' / revision
        obj = cls()
        obj.backend = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
        config = json.loads((snapshot / 'tokenizer_config.json').read_text(encoding='utf-8'))
        eos = config['eos_token']
        pad = config['pad_token']
        obj.eos_token = eos['content'] if isinstance(eos, dict) else eos
        obj.pad_token = pad['content'] if isinstance(pad, dict) else pad
        obj.eos_token_id = obj.backend.token_to_id(obj.eos_token)
        obj.pad_token_id = obj.backend.token_to_id(obj.pad_token)
        obj.is_fast = True
        obj.name_or_path = model_id
        obj.revision = revision
        return obj

    def convert_tokens_to_ids(self, value):
        return self.backend.token_to_id(value)

    def __call__(self, values, add_special_tokens=False, return_offsets_mapping=False):
        scalar = isinstance(values, str)
        encoded = self.backend.encode_batch([values] if scalar else values, add_special_tokens=add_special_tokens)
        result = {'input_ids': encoded[0].ids if scalar else [e.ids for e in encoded]}
        if return_offsets_mapping:
            result['offset_mapping'] = encoded[0].offsets if scalar else [e.offsets for e in encoded]
        return result

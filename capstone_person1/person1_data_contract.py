"""Shared plain-text data contract for Person 2/3/4; never loads model weights.

Answer consistency is a structural check, not a proof of mathematical correctness.
The notebook embeds this same source after the controls source.
"""
import hashlib
import json
import re
from pathlib import Path
from capstone_person1.instance_matching import context_equivalent

if 'build_completion' not in globals():
    from capstone_person1.pipeline_controls import (
        PROMPT_TEMPLATE, build_completion, extract_answer_metadata, answers_equivalent, NUMBER_PATTERN,
    )

CONTRACT_VERSION = 'person1-plain-text-labels-v3'
TOKENIZER_ID = 'Qwen/Qwen2.5-0.5B'
PINNED_TOKENIZER_REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
EXPECTED_END_TOKEN = '<|endoftext|>'
EXPECTED_END_ID = 151643


class BoundaryError(ValueError):
    """A token straddles prompt and completion; do not guess a loss mask."""


class ContextLengthError(ValueError):
    """Exclude the whole row instead of truncating its reasoning or answer."""


def render_prompt(problem):
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError('A nonempty problem string is required')
    return PROMPT_TEMPLATE.format(problem=problem)


def render_completion(solution, verified_final_answer, *, problem=''):
    evidence = extract_answer_metadata('', solution)
    if evidence.mode.startswith('ambiguous'):
        raise ValueError('Ambiguous solution conclusion requires review: ' + evidence.mode)
    if not evidence.candidate:
        raise ValueError('No supported terminal answer evidence; quarantine for review')
    if context_equivalent(problem, evidence.candidate, verified_final_answer) is not True:
        raise ValueError('Target does not demonstrably agree with the terminal solution answer')
    if evidence.confidence == 'heuristic':
        raise ValueError('Heuristic answer requires an individual corrected/reviewed terminal block')
    # Numeric comparison keys normalize 90% to 9/10. Preserve the percentage
    # notation in the generated answer so a percentage question remains clear.
    percentages = re.findall(NUMBER_PATTERN.removesuffix('%?') + r'\s*\\?%', evidence.evidence_text)
    matching = [value.replace('\\%', '%').replace(' ', '') for value in percentages
                if answers_equivalent(value.replace('\\%', '%').replace(' ', ''), verified_final_answer) is True]
    display_answer = matching[0] if len(matching) == 1 else verified_final_answer
    return build_completion(solution, display_answer)


def verify_tokenizer(tokenizer):
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('The data contract requires a fast tokenizer with offsets')
    for field, expected in [('eos_token', EXPECTED_END_TOKEN), ('pad_token', EXPECTED_END_TOKEN),
                            ('eos_token_id', EXPECTED_END_ID), ('pad_token_id', EXPECTED_END_ID)]:
        if getattr(tokenizer, field, None) != expected:
            raise ValueError(f'Pinned tokenizer {field} differs from the agreed contract')
    if tokenizer.convert_tokens_to_ids(EXPECTED_END_TOKEN) != EXPECTED_END_ID:
        raise ValueError('EOS vocabulary mapping differs from the agreed contract')
    return {field: getattr(tokenizer, field) for field in
            ['eos_token', 'pad_token', 'eos_token_id', 'pad_token_id', 'is_fast']}


def load_tokenizer(*, cache_dir=None, local_files_only=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=PINNED_TOKENIZER_REVISION, use_fast=True,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    verify_tokenizer(tokenizer)
    return tokenizer


def encode_text_pair(tokenizer, prompt, completion, *, max_total_tokens=2048):
    """Encode combined text, append EOS, then derive every budget from actual labels."""
    verify_tokenizer(tokenizer)
    if not isinstance(prompt, str) or not prompt.endswith('Solution:\n'):
        raise ValueError('Prompt must end with the exact Solution newline')
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError('Completion must be nonempty')
    encoded = tokenizer(prompt + completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded['input_ids'])
    offsets = list(encoded['offset_mapping'])
    if len(ids) != len(offsets) or not ids:
        raise BoundaryError('Missing combined-text offsets')
    boundary = len(prompt)
    prompt_mask = []
    for left, right in offsets:
        if not (0 <= left < right <= len(prompt) + len(completion)):
            raise BoundaryError('Unsupported empty/invalid tokenizer offset')
        if left < boundary < right:
            raise BoundaryError('Token crosses the prompt/completion character boundary')
        prompt_mask.append(right <= boundary)
    prompt_tokens = sum(prompt_mask)
    if not 0 < prompt_tokens < len(ids) or prompt_mask != [True] * prompt_tokens + [False] * (len(ids) - prompt_tokens):
        raise BoundaryError('Combined-text offsets do not define a contiguous completion')
    if tokenizer.eos_token_id in ids:
        raise ValueError('Input text already contains an EOS control token')
    labels = [-100 if is_prompt else token for token, is_prompt in zip(ids, prompt_mask)]
    completion_tokens = sum(label != -100 for label in labels)
    ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    if max_total_tokens is not None and len(ids) > max_total_tokens:
        raise ContextLengthError(f'Whole example needs {len(ids)} tokens (limit {max_total_tokens})')
    # Separate prompt encoding is only a diagnostic; it never supplies labels or IDs.
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    return {
        'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels,
        'prompt_text': prompt, 'completion': completion,
        'completion_start': prompt_tokens, 'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens, 'eos_tokens': 1,
        'supervised_tokens': sum(label != -100 for label in labels),
        'total_tokens': len(ids), 'boundary_mismatch': ids[:len(prefix)] != list(prefix),
        'boundary_crossing_token': False,
    }


def encode_example(tokenizer, problem, solution, verified_final_answer, *, max_total_tokens=2048):
    evidence = extract_answer_metadata(problem, solution)
    if evidence.mode.startswith('ambiguous'):
        raise ValueError('Problem/solution requires review: ' + evidence.mode)
    return encode_text_pair(tokenizer, render_prompt(problem),
                            render_completion(solution, verified_final_answer, problem=problem),
                            max_total_tokens=max_total_tokens)


def pad_encoded(encoded, length, *, pad_token_id=EXPECTED_END_ID):
    """Right pad by position; the real EOS remains supervised even when PAD == EOS."""
    if length < len(encoded['input_ids']):
        raise ContextLengthError('Padding must never truncate a row')
    result = dict(encoded)
    count = length - len(encoded['input_ids'])
    result['input_ids'] = list(encoded['input_ids']) + [pad_token_id] * count
    result['attention_mask'] = list(encoded['attention_mask']) + [0] * count
    result['labels'] = list(encoded['labels']) + [-100] * count
    if sum(label != -100 for label in result['labels']) != encoded['supervised_tokens']:
        raise ValueError('Padding changed the supervised-token budget')
    return result


def validate_manifest_record(record, tokenizer, *, max_total_tokens=2048):
    for key in ['row_id', 'problem_group_id']:
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError('Manifest requires nonempty ' + key)
    encoded = encode_example(tokenizer, record['problem'], record['solution'], record['final_answer'],
                             max_total_tokens=max_total_tokens)
    for key in ['prompt_text', 'completion', 'prompt_tokens', 'completion_tokens', 'supervised_tokens', 'total_tokens']:
        if record.get(key) != encoded[key]:
            raise ValueError('Manifest differs from the shared encoder: ' + key)
    return encoded


def validate_manifest(path, release_metadata, tokenizer=None):
    """Stream a JSONL manifest, re-encode each row and verify IDs/groups/hash/counts."""
    if release_metadata.get('tokenizer_id') != TOKENIZER_ID or release_metadata.get('tokenizer_revision') != PINNED_TOKENIZER_REVISION:
        raise ValueError('Manifest release uses a different tokenizer identity/revision')
    path = Path(path)
    if release_metadata.get('manifest_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Manifest content hash differs from release metadata')
    if tokenizer is None:
        tokenizer = load_tokenizer(cache_dir=release_metadata.get('tokenizer_cache_dir'))
    ids, groups, supervised = set(), set(), 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            encoded = validate_manifest_record(record, tokenizer, max_total_tokens=release_metadata.get('max_total_tokens', 2048))
            if record['row_id'] in ids or record['problem_group_id'] in groups:
                raise ValueError('Duplicate row/group in manifest')
            ids.add(record['row_id'])
            groups.add(record['problem_group_id'])
            supervised += encoded['supervised_tokens']
    if not ids or len(ids) != release_metadata.get('rows'):
        raise ValueError('Manifest is empty or row count differs from release metadata')
    if supervised != release_metadata.get('supervised_tokens'):
        raise ValueError('Manifest label-derived budget differs from release metadata')
    return {'rows': len(ids), 'supervised_tokens': supervised, 'status': 'passed'}


def formatting_contract(tokenizer, *, asset_directory=None, max_total_tokens=2048):
    verified = verify_tokenizer(tokenizer)
    hashes = {}
    if asset_directory is not None:
        for name in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
            path = Path(asset_directory) / name
            if path.is_file(): hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not {'tokenizer.json', 'tokenizer_config.json'} <= hashes.keys():
            raise ValueError('Pinned tokenizer assets are missing')
    return {
        'contract_version': CONTRACT_VERSION, 'prompt_template': PROMPT_TEMPLATE,
        'prompt_template_sha256': hashlib.sha256(PROMPT_TEMPLATE.encode('utf-8')).hexdigest(),
        'tokenizer_id': TOKENIZER_ID, 'tokenizer_revision': PINNED_TOKENIZER_REVISION,
        'tokenizer_class': type(tokenizer).__name__, 'tokenizer_verified_at_load': True,
        **verified, 'tokenizer_asset_sha256': hashes,
        'add_special_tokens': False, 'apply_chat_template': False,
        'encoding': 'combined_prompt_and_completion_with_character_offsets',
        'completion_policy': 'preserve_correct_terminal_block; quarantine_unproven_consistency',
        'append_eos_count': 1, 'loss': 'completion_and_eos_only', 'ignore_label': -100,
        'padding': 'right_padding; attention_mask=0; labels=-100_by_position',
        'budget': 'sum(label != -100 for label in labels)', 'max_total_tokens': max_total_tokens,
        'evaluation_input': 'render_prompt(problem); no_reference_answer_or_eos',
        'other_model_tokenizers': 'load_corresponding_pinned_tokenizer_and_verify_before_reusing_counts',
    }

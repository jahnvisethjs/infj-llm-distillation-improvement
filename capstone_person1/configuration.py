"""Validate the data configuration and map it to notebook stage settings."""
import re


def notebook_settings(data):
    if data['tokenizer_id'] != 'Qwen/Qwen2.5-0.5B' or data['tokenizer_revision'] != '060db6499f32faf8b98477b0a26969ef7d8b9987':
        raise ValueError('This release requires the agreed pinned 0.5B tokenizer')
    for source in data['sources'].values():
        if source['split'] != 'train' or not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
            raise ValueError('Source split/revision must be an immutable train snapshot')
    for key in ['train_bank_per_source', 'selection_bank_per_source', 'loss_dev_per_source', 'max_total_tokens']:
        if type(data[key]) is not int or data[key] < 1: raise ValueError('Invalid positive count: ' + key)
    return {
        'ORCA_DATASET_ID': data['sources']['orca']['dataset_id'], 'NUMINA_DATASET_ID': data['sources']['numina']['dataset_id'],
        'ORCA_REVISION': data['sources']['orca']['revision'], 'NUMINA_REVISION': data['sources']['numina']['revision'],
        'TOKENIZER_ID': data['tokenizer_id'], 'TOKENIZER_REVISION': data['tokenizer_revision'],
        'GSM8K_REVISION': data['reserved_tests']['gsm8k']['revision'], 'MATH500_REVISION': data['reserved_tests']['math500']['revision'],
        'SPLIT_SEED': data['split_seed'], 'SELECTION_SEED': data['selection_seed'], 'MAX_TOTAL_TOKENS': data['max_total_tokens'],
        'TARGET_PER_SOURCE': data['train_bank_per_source'], 'TARGET_WORKING_POOL': data['train_bank_per_source'] * 2,
        'FRESH_POOL_TARGET': data['selection_bank_per_source'] * 2, 'LOSS_DEV_TARGET': data['loss_dev_per_source'] * 2,
        'MANIFEST_SIZES': data['required_training_prefixes'] + data['optional_training_prefixes'],
        'CANDIDATE_UNIVERSE_PER_SOURCE': data['candidate_universe_per_source'],
        'LSH_CANDIDATE_THRESHOLD': data['lsh_candidate_threshold'], 'NEAR_DUPLICATE_THRESHOLD': data['near_duplicate_threshold'],
        'DECONTAMINATION_THRESHOLD': data['decontamination_threshold'], 'MINHASH_PERMUTATIONS': data['minhash_permutations'],
        'EXCLUDE_SYNTHETIC_NUMINA': data['exclude_synthetic_numina'],
    }

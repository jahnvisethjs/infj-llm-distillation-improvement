"""Read back actual exports and verify nested order, hashes, counts and draft gating."""
import argparse
import json
from pathlib import Path

import pandas as pd

from capstone_person1.pipeline_controls import file_sha256, PROMPT_TEMPLATE
from capstone_person1.person1_data_contract import validate_manifest_record, TOKENIZER_ID, PINNED_TOKENIZER_REVISION
from capstone_person1.tokenizer_adapter import SnapshotTokenizer


def verify(run_root):
    run_root = Path(run_root)
    reports, manifests = run_root / 'outputs/reports', run_root / 'outputs/manifests'
    metrics = json.loads((reports / 'rebuild_metrics.json').read_text(encoding='utf-8'))
    ready = json.loads((reports / 'handoff_readiness.json').read_text(encoding='utf-8'))
    if metrics['status'] != 'draft_complete' or ready['research_ready'] is not False:
        raise ValueError('Smoke run must finish as a gated draft')
    for name, digest in ready['artifact_sha256'].items():
        path = (reports if name == 'prompt_specification.json' else manifests) / name
        if file_sha256(path) != digest: raise ValueError(f'Export hash mismatch: {name}')
    frames = {}
    tokenizer = SnapshotTokenizer.from_pretrained(TOKENIZER_ID, revision=PINNED_TOKENIZER_REVISION)
    for size in sorted(map(int, ready['manifest_sha256'])):
        path = manifests / f'train_n{size}_pilot.jsonl'
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        frame = pd.DataFrame(records)
        if len(frame) != size or not frame['row_id'].is_unique or not frame['problem_group_id'].is_unique:
            raise ValueError('Wrong manifest size or duplicate identities')
        if frame['total_tokens'].gt(2048).any(): raise ValueError('Overlength target selected')
        if frame['correctness_status'].ne('unknown').any(): raise ValueError('Unreviewed mathematical correctness claimed')
        for record in records:
            validate_manifest_record(record, tokenizer)
        frames[size] = frame
    largest = frames[max(frames)]
    for size, frame in frames.items():
        if frame['row_id'].tolist() != largest.head(size)['row_id'].tolist():
            raise ValueError('Nested manifest is not the exact ordered prefix')
    if largest['dataset_id'].value_counts().to_dict() != {'orca': 50, 'numina': 50}:
        raise ValueError('Smoke working-pool source balance changed')
    result = {'run': run_root.name, 'rows_processed': sum(stage['processed_rows'] for stage in metrics['source_stages'].values()),
              'elapsed_seconds': metrics['elapsed_seconds'], 'peak_rss_mib': metrics['peak_rss_mib'],
              'source_rows_reused': sum(stage['reused_rows'] for stage in metrics['source_stages'].values()),
              'nested_sizes': sorted(frames), 'research_ready': ready['research_ready'],
              'manifest_sha256': ready['manifest_sha256']}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root')
    verify(parser.parse_args().run_root)

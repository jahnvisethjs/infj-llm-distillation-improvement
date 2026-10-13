"""Offline verification using the actual pinned AutoTokenizer and local cached assets."""
import importlib.metadata
import json
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'capstone_person1'
sys.path.insert(0, str(ROOT))
# A compatible local Hub install avoids modifying the user's global environment.
if (PROJECT / 'runtime_deps').exists(): sys.path.insert(0, str(PROJECT / 'runtime_deps'))

from capstone_person1 import person1_data_contract as contract
from capstone_person1.tokenizer_adapter import SnapshotTokenizer
from capstone_person1.snapshot_stream import SnapshotCanonicalizer, SOURCE_COLUMNS


def inspect_snapshot_sample(tokenizer):
    """Inspect deterministic positions throughout snapshots; not a final upstream pilot."""
    import pyarrow.parquet as pq
    notebook = json.loads((ROOT / 'Person1_Full_Run_v2.ipynb').read_text(encoding='utf-8'))
    mapper = SnapshotCanonicalizer(notebook)
    counts, outcomes = {}, []
    for source in ['orca', 'numina']:
        path = PROJECT / f'runs/full_run_v1/outputs/processed/{source}_canonical_full.parquet'
        parquet = pq.ParquetFile(path)
        positions = set(random.Random(1729).sample(range(parquet.metadata.num_rows), 100))
        cursor, source_counts = 0, Counter()
        for batch in parquet.iter_batches(batch_size=256, columns=SOURCE_COLUMNS[source], use_threads=False):
            selected = [offset for offset in range(batch.num_rows) if cursor + offset in positions]
            if selected:
                rows = batch.to_pylist()
                for offset in selected:
                    row = mapper(source, rows[offset], cursor + offset)
                    record = {'source': source, 'original_row_id': row['original_row_id'], 'row_id': row['row_id'],
                              'quality_flags': row['quality_flags'], 'extraction': json.loads(row['answer_extraction_json'])}
                    try:
                        encoded = contract.encode_example(tokenizer, row['problem'], row['solution'], row['final_answer'])
                        source_counts['reference_encoding_passed'] += 1
                        record.update(reference_encoding='passed', supervised_tokens=encoded['supervised_tokens'])
                    except ValueError as error:
                        source_counts['reference_encoding_rejected'] += 1
                        record.update(reference_encoding='rejected', reason=str(error))
                    source_counts['initial_eligible'] += int(row['eligible'])
                    source_counts['rows_inspected'] += 1
                    outcomes.append(record)
            cursor += batch.num_rows
            if cursor > max(positions): break
        counts[source] = dict(source_counts)
        print('Sample inspection:', source, counts[source], flush=True)
    report = {'status': 'diagnostic_only', 'sampling': '100 fixed-seed random positions across each whole historical snapshot',
              'seed': 1729, 'sources': counts, 'cases': outcomes,
              'limitations': ['Historical Numina answers may be truncated and validity labels are inferred.',
                             'Encoding success alone does not mean eligible, mathematically correct or decontaminated.',
                             'This is not the required pilot from complete pinned upstream fields.']}
    (PROJECT / 'evidence/SNAPSHOT_CONTRACT_SAMPLE.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return counts


def main():
    cache = PROJECT / 'cache/huggingface'
    tokenizer = contract.load_tokenizer(cache_dir=cache, local_files_only=True)
    adapter = SnapshotTokenizer.from_pretrained(contract.TOKENIZER_ID, revision=contract.PINNED_TOKENIZER_REVISION)
    checked = []
    for problem, solution, answer in [
        ('What is 2 + 3?', 'Adding the quantities gives the total.\nFinal answer: 5', '5'),
        ('Find the value of x.', 'Solving the relation gives nine halves.\n\nFinal answer: 4.5\n', '9/2'),
        ('A café sold 25% of its stock. What fraction?', 'The percentage is one quarter of the whole.\nFinal answer: 1/4', '1/4'),
        ('Compute the negative quantity.', 'The steps give the negative ratio.\\boxed{\\frac{-9}{2}}', '-9/2'),
    ]:
        actual = contract.encode_example(tokenizer, problem, solution, answer)
        lightweight = contract.encode_example(adapter, problem, solution, answer)
        for key in ['input_ids', 'labels', 'attention_mask', 'completion_start', 'supervised_tokens']:
            assert actual[key] == lightweight[key], key
        padded = contract.pad_encoded(actual, actual['total_tokens'] + 4)
        assert padded['labels'][actual['total_tokens'] - 1:] == [151643] + [-100] * 4
        assert sum(label != -100 for label in padded['labels']) == actual['supervised_tokens']
        assert actual['input_ids'].count(tokenizer.eos_token_id) == 1
        assert actual['labels'][:actual['prompt_tokens']] == [-100] * actual['prompt_tokens']
        checked.append({'problem': problem, 'prompt_tokens': actual['prompt_tokens'],
                        'completion_tokens': actual['completion_tokens'], 'supervised_tokens': actual['supervised_tokens']})
    snapshot = cache / 'models--Qwen--Qwen2.5-0.5B/snapshots' / contract.PINNED_TOKENIZER_REVISION
    spec = contract.formatting_contract(tokenizer, asset_directory=snapshot)
    spec['reference_encoder_sha256'] = contract.hashlib.sha256((PROJECT / 'person1_data_contract.py').read_bytes()).hexdigest()
    (PROJECT / 'configs/formatting_contract.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    evidence = {'status': 'passed', 'model_weights_loaded': False, 'network_used': False,
                'tokenizer_class': type(tokenizer).__name__, 'revision': contract.PINNED_TOKENIZER_REVISION,
                'versions': {key: importlib.metadata.version(key) for key in ['transformers', 'huggingface-hub', 'tokenizers']},
                'checks': checked, 'corrected_corpus_ready': False}
    if '--inspect-snapshot-sample' in sys.argv:
        evidence['snapshot_sample'] = inspect_snapshot_sample(tokenizer)
    if '--run' in sys.argv:
        run_root = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
        if not run_root.is_relative_to((PROJECT / 'runs').resolve()):
            raise ValueError('Verification run must belong to the project runs folder')
        manifests = run_root / 'outputs/manifests'
        validations = {}
        for path in sorted(manifests.glob('train_n*_pilot.jsonl')):
            summary = json.loads(path.with_name(path.stem + '_summary.json').read_text(encoding='utf-8'))
            metadata = dict(tokenizer_id=contract.TOKENIZER_ID, tokenizer_revision=contract.PINNED_TOKENIZER_REVISION,
                            rows=summary['rows'], supervised_tokens=summary['total_supervised_tokens'],
                            manifest_sha256=summary['sha256'], max_total_tokens=2048)
            validations[path.name] = contract.validate_manifest(path, metadata, tokenizer)
        if not validations: raise ValueError('No diagnostic manifests found')
        evidence['actual_saved_manifest_validation'] = validations
        (run_root / 'outputs/reports/reference_label_validation.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    (PROJECT / 'evidence/REFERENCE_CONTRACT_VERIFICATION.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__': main()

"""Bounded Parquet canonicalization with independently committed batch checkpoints."""
import ast
import json
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from capstone_person1 import pipeline_controls as controls

SOURCE_COLUMNS = {
    'orca': ['original_row_id', 'dataset_revision', 'problem_raw', 'solution_raw', 'final_answer', 'answer_parse_status'],
    'numina': ['original_row_id', 'dataset_revision', 'problem_raw', 'solution_raw', 'final_answer',
               'original_source', 'question_type', 'problem_type', 'synthetic', 'quality_flags'],
}


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
    pending.replace(path)


def notebook_config(notebook):
    """Read literal configuration and execute only the notebook's canonical definitions."""
    values = {}
    for node in ast.parse(''.join(notebook['cells'][5]['source'])).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return values


class SnapshotCanonicalizer:
    def __init__(self, notebook):
        self.config = notebook_config(notebook)
        self.namespace = dict(vars(controls))
        self.namespace.update(self.config)
        self.namespace.update(orca_revision=self.config['ORCA_REVISION'], numina_revision=self.config['NUMINA_REVISION'])
        source = ''.join(notebook['cells'][14]['source']).split('def make_inspection_sheet', 1)[0]
        exec(compile(source, 'notebook_canonical_definitions', 'exec'), self.namespace)

    def __call__(self, source, row, position):
        revision = self.config[source.upper() + '_REVISION']
        if row['dataset_revision'] != revision:
            raise ValueError(f'{source} snapshot contains a different source revision')
        base = {'__original_index': int(row['original_row_id'])}
        if source == 'orca':
            base.update(question=row['problem_raw'], answer=row['solution_raw'])
        else:
            old_flags = set(row['quality_flags'] or [])
            base.update(problem=row['problem_raw'], solution=row['solution_raw'], answer=row['final_answer'],
                        source=row['original_source'], question_type=row['question_type'],
                        problem_type=row['problem_type'], synthetic=row['synthetic'],
                        problem_is_valid='no' if 'numina_problem_not_valid' in old_flags else 'yes',
                        solution_is_valid='no' if 'numina_solution_not_valid' in old_flags else 'yes')
        result = self.namespace['canonicalize_' + source](base, position)
        if source == 'numina' and len(row['final_answer']) >= 500:
            result['quality_flags'].append('possibly_truncated_v1_answer_requires_upstream_reload')
            result['eligible'], result['split'] = False, 'quarantine'
        return result


def record_schema(record):
    lists, booleans = {'quality_flags', 'review_flags'}, {'eligible', 'synthetic'}
    integers = {'problem_char_length', 'solution_char_length'}
    return pa.schema([(key, pa.list_(pa.string()) if key in lists else pa.bool_() if key in booleans
                       else pa.int64() if key in integers else pa.string()) for key in record])


def canonicalize_snapshot(source, input_path, stage_dir, mapper, fingerprint, *, limit=None,
                          batch_size=256, observe=lambda: None, progress=lambda message: None):
    """Resume committed batches only when inputs/controls/options and shard hashes match."""
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = stage_dir / 'checkpoint.json'
    parquet = pq.ParquetFile(input_path)
    rows = min(limit, parquet.metadata.num_rows) if limit is not None else parquet.metadata.num_rows
    binding = {'fingerprint': fingerprint, 'input_sha256': controls.file_sha256(input_path),
               'source': source, 'rows': rows, 'batch_size': batch_size}
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if state['binding'] != binding:
            raise ValueError('Checkpoint inputs/controls/options changed; start a new run')
        for shard in state['shards']:
            if controls.file_sha256(stage_dir / shard['name']) != shard['sha256']:
                raise ValueError('Committed canonical checkpoint shard changed')
            if 'changes_name' in shard and controls.file_sha256(stage_dir / shard['changes_name']) != shard['changes_sha256']:
                raise ValueError('Committed extraction ledger shard changed')
    else:
        state = {'binding': binding, 'processed_rows': 0, 'complete': False, 'shards': [],
                 'quality_counts': {}, 'review_counts': {}, 'changed_answers': 0}
        atomic_json(checkpoint_path, state)
    committed_before = state['processed_rows']
    if state['complete']:
        return dict(state, reused_rows=committed_before, elapsed_seconds=0.0)
    started, position = time.perf_counter(), 0
    quality, reviews = Counter(state['quality_counts']), Counter(state['review_counts'])
    for batch in parquet.iter_batches(batch_size=batch_size, columns=SOURCE_COLUMNS[source], use_threads=False):
        count = min(batch.num_rows, rows - position)
        if count <= 0: break
        if position + count <= state['processed_rows']:
            position += count
            continue
        if position != state['processed_rows']:
            raise ValueError('Checkpoint cursor is not on a committed batch boundary')
        original = batch.slice(0, count).to_pylist()
        records = [mapper(source, row, position + i) for i, row in enumerate(original)]
        for old, new in zip(original, records):
            quality.update(new['quality_flags'] or ['retained_without_flags'])
            reviews.update(new['review_flags'] or ['retained_without_flags'])
            state['changed_answers'] += int(source == 'orca' and old['final_answer'] != new['final_answer'])
        name = f'batch_{position:09d}.parquet'
        table = pa.Table.from_pylist(records, schema=record_schema(records[0]))
        pending = stage_dir / (name + '.pending')
        pq.write_table(table, pending, compression='zstd')
        pending.replace(stage_dir / name)
        shard = {'name': name, 'rows': count, 'sha256': controls.file_sha256(stage_dir / name)}
        if source == 'orca':
            changes = [dict(row_id=new['row_id'], original_row_id=new['original_row_id'],
                            final_answer_v1=old['final_answer'], final_answer_v2=new['final_answer'],
                            answer_parse_status_v1=old['answer_parse_status'], answer_parse_status_v2=new['answer_parse_status'],
                            eligible=new['eligible']) for old, new in zip(original, records)
                       if old['final_answer'] != new['final_answer'] or old['answer_parse_status'] != new['answer_parse_status']]
            change_path = stage_dir / f'changes_{position:09d}.jsonl'
            change_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in changes), encoding='utf-8')
            shard.update(changes_name=change_path.name, changes_sha256=controls.file_sha256(change_path))
        state['shards'].append(shard)
        state.update(processed_rows=position + count, quality_counts=dict(quality), review_counts=dict(reviews))
        atomic_json(checkpoint_path, state)
        position += count
        observe()
        if len(state['shards']) % 20 == 0: progress(f'{source}: {position:,}/{rows:,} rows committed')
    state['complete'] = state['processed_rows'] == rows
    atomic_json(checkpoint_path, state)
    return dict(state, reused_rows=committed_before, elapsed_seconds=time.perf_counter() - started)


def stage_batches(stage_dir, columns=None, batch_size=256):
    stage_dir = Path(stage_dir)
    state = json.loads((stage_dir / 'checkpoint.json').read_text(encoding='utf-8'))
    if not state['complete']:
        raise ValueError('Canonical stage is incomplete')
    for shard in state['shards']:
        yield from pq.ParquetFile(stage_dir / shard['name']).iter_batches(
            columns=columns, batch_size=batch_size, use_threads=False)


def selected_records(stage_dir, ids, columns=None):
    """Materialize text only for the bounded candidate/review IDs."""
    ids, records = set(ids), []
    if not ids: return records
    value_set = pa.array(list(ids))
    for batch in stage_batches(stage_dir, columns):
        mask = pc.is_in(batch.column('row_id'), value_set=value_set)
        records.extend(pa.Table.from_batches([batch]).filter(mask).to_pylist())
    if len(records) != len(ids):
        raise ValueError('Selected canonical IDs have missing or duplicate records')
    return records

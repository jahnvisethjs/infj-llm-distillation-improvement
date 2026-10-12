"""Checkpointed DRAFT rebuild. Use --rows-per-source 500, then 2500; --full is explicit."""
import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / 'capstone_person1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / 'vendor'))

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1 import pipeline_controls as controls
from capstone_person1.snapshot_catalog import open_catalog, prepare_candidates, ordered_source_ids, META_COLUMNS
from capstone_person1.snapshot_stream import SnapshotCanonicalizer, atomic_json, canonicalize_snapshot, selected_records
from capstone_person1.tokenizer_adapter import SnapshotTokenizer


class MemoryMeter:
    def __init__(self, max_mb):
        self.process = psutil.Process()
        self.maximum = max_mb * 1024 * 1024
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop.wait(0.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def observe(self):
        current = self.process.memory_info().rss
        self.peak = max(self.peak, current)
        if current > self.maximum:
            raise RuntimeError(f'Memory budget exceeded ({current / 2**20:.1f} MiB); committed batches can resume')


def source_reviews(conn, stages, states, ns):
    for source, stage in stages.items():
        positions = random.Random(ns['SPLIT_SEED']).sample(range(states[source]['processed_rows']), min(25, states[source]['processed_rows']))
        identities = [conn.execute('SELECT row_id FROM records WHERE dataset_id=? AND position=?', [source, index]).fetchone()[0]
                      for index in positions]
        lookup = {row['row_id']: row for row in selected_records(stage, identities)}
        inspection = pd.DataFrame([dict(
            dataset=source, original_index=int(lookup[row_id]['original_row_id']), problem=lookup[row_id]['problem_raw'],
            solution=lookup[row_id]['solution_raw'], provided_final_answer=lookup[row_id]['final_answer'] if source == 'numina' else '',
            manual_solution_complete_yes_no='', manual_final_answer_clear_yes_no='', manual_needs_missing_diagram_yes_no='',
            manual_keep_quarantine='', manual_notes='') for row_id in identities])
        controls.save_review_sheet(inspection, ns['REPORT_DIR'] / f'{source}_inspection_25.csv')
    ids = []
    for condition, seed, count in [
        ("answer_parse_status='terminal_single_number_review'", ns['SPLIT_SEED'], 20),
        ("answer_parse_status IN ('explicit_final_marker','gsm_marker','boxed','terminal_answer_clause')", ns['SPLIT_SEED'] + 1, 25),
    ]:
        values = ordered_source_ids(conn, 'orca', condition)
        order = np.random.default_rng(seed).permutation(len(values))[:min(count, 25 - len(ids), len(values))]
        ids.extend(values[int(index)] for index in order)
    lookup = {row['row_id']: row for row in selected_records(stages['orca'], ids)}
    cols = ['row_id','original_row_id','problem','solution','final_answer','answer_parse_status']
    extraction = pd.DataFrame([{col: lookup[value][col] for col in cols} for value in ids], columns=cols)
    extraction['solution_tail'] = extraction['solution'].str[-500:]
    for col in ['manual_extraction_correct_yes_no','manual_corrected_answer','manual_notes']: extraction[col] = ''
    controls.save_review_sheet(extraction, ns['REPORT_DIR'] / 'orca_answer_extraction_review_25.csv')
    for field, filename, label in [('quality_counts','quality_audit.csv','flag'), ('review_counts','review_flags_audit.csv','review_flag')]:
        frame = pd.DataFrame([{'dataset': source, label: key, 'count': value}
                              for source in stages for key, value in states[source][field].items()])
        frame.sort_values(['dataset','count'], ascending=[True,False]).to_csv(ns['REPORT_DIR'] / filename, index=False)
    fields = ['row_id','original_row_id','final_answer_v1','final_answer_v2','answer_parse_status_v1','answer_parse_status_v2','eligible']
    with open(ns['REPORT_DIR'] / 'orca_extraction_changes.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for shard in states['orca']['shards']:
            for line in (stages['orca'] / shard['changes_name']).read_text(encoding='utf-8').splitlines():
                writer.writerow(json.loads(line))


def status_audit(conn, ns, source, observe):
    exec(compile(source[:source.index('# Preserve a row-level disposition')], 'notebook_structural_assertions', 'exec'), ns)
    exec(compile(source[source.index('def assign_final_status'):source.index('row_audit =')], 'notebook_row_disposition', 'exec'), ns)
    cursor = conn.execute('SELECT ' + ','.join(META_COLUMNS) + ' FROM records ORDER BY dataset_id,position')
    path = ns['PROCESSED_DIR'] / f"row_status_audit_{ns['RUN_TAG']}.parquet"
    counts, writer = Counter(), None
    try:
        while rows := cursor.fetchmany(256):
            output = []
            for values in rows:
                row = dict(zip(META_COLUMNS, values))
                row['quality_flags'], row['review_flags'] = json.loads(row['quality_flags']), json.loads(row['review_flags'])
                status = ns['assign_final_status'](row)
                counts[status['final_status']] += 1
                output.append({**{col: row[col] for col in ['row_id','dataset_id','original_row_id','original_source','problem_hash','answer_parse_status','eligible']}, **status})
            table = pa.Table.from_pylist(output)
            if writer is None: writer = pq.ParquetWriter(path, table.schema, compression='zstd')
            writer.write_table(table)
            observe()
    finally:
        if writer is not None: writer.close()
    ns['final_status_counts'] = dict(counts)
    exec(compile(source[source.index('audit_summary ='):], 'notebook_audit_summary', 'exec'), ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--rows-per-source', type=int)
    size.add_argument('--full', action='store_true')
    parser.add_argument('--resume', metavar='RUN_ID')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-memory-mb', type=int, default=768)
    parser.add_argument('--sampling', choices=['prefix', 'random'], default=None,
                        help='Random fixed-seed positions across snapshots; full upstream pilot remains separate')
    args = parser.parse_args()
    if args.rows_per_source is not None and args.rows_per_source < 100: parser.error('Use at least 100 rows per source')
    if args.max_memory_mb < 128: parser.error('Memory budget must be at least 128 MiB')
    os.chdir(ROOT)
    notebook_path = ROOT / 'Person1_Full_Run_v2.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    mapper = SnapshotCanonicalizer(notebook)
    previous = PROJECT / 'runs/full_run_v1/outputs'
    inputs = {source: previous / f'processed/{source}_canonical_full.parquet' for source in ['orca','numina']}
    options = {'full': args.full, 'rows_per_source': args.rows_per_source or (None if args.full else 500),
               'batch_size': args.batch_size, 'sampling': args.sampling or 'random'}
    if args.resume:
        saved = json.loads((PROJECT / 'runs' / args.resume / 'snapshot_run_options.json').read_text(encoding='utf-8'))
        if args.rows_per_source is None and not args.full and args.sampling is None: options = saved
        elif options != saved: raise ValueError('Resume options differ from the original run')
        os.environ['CAPSTONE_RUN_ID'] = args.resume
    else: os.environ.pop('CAPSTONE_RUN_ID', None)
    ns = {'display': lambda *items: None, 'AutoTokenizer': SnapshotTokenizer}
    def run(index, code=None):
        print(f'Executing notebook cell {index+1}', flush=True)
        exec(compile(code if code is not None else ''.join(notebook['cells'][index]['source']), f'notebook_cell_{index+1}', 'exec'), ns)
    folder_code = ''.join(notebook['cells'][3]['source'])
    if args.resume: folder_code = folder_code.replace('RESUME_RUN = False', 'RESUME_RUN = True')
    run(3, folder_code)
    atomic_json(ns['RUN_ROOT'] / 'snapshot_run_options.json', options)
    run(5)
    if not options['full']:
        ns.update(RUN_MODE='pilot', REQUIRE_EVAL_MANIFEST=False, TARGET_PER_SOURCE=50,
                  TARGET_WORKING_POOL=100, CANDIDATE_UNIVERSE_PER_SOURCE=200, MANIFEST_SIZES=[32,64,100])
    ns['snapshot_binding'] = {
        'notebook_sha256': controls.file_sha256(notebook_path),
        'runner_sha256': controls.file_sha256(Path(__file__)),
        'stream_sha256': controls.file_sha256(PROJECT / 'snapshot_stream.py'),
        'catalog_sha256': controls.file_sha256(PROJECT / 'snapshot_catalog.py'),
        'contract_sha256': controls.file_sha256(PROJECT / 'person1_data_contract.py'),
        'tokenizer_adapter_sha256': controls.file_sha256(PROJECT / 'tokenizer_adapter.py'),
        'source_file_sha256': {source: controls.file_sha256(path) for source, path in inputs.items()}, 'options': options,
    }
    imports = ''.join(notebook['cells'][8]['source'])
    imports = imports.replace('from transformers import AutoTokenizer', '# Local pinned tokenizer JSON adapter')
    imports = imports.replace('from datasets import Dataset, concatenate_datasets, load_dataset', '# Source processing uses bounded Arrow batches')
    imports = imports.replace("'control_source_sha256': sha256_text(CONTROL_SOURCE),",
                              "'control_source_sha256': sha256_text(CONTROL_SOURCE), 'snapshot_binding': snapshot_binding,")
    run(8, imports)
    ns['tqdm'] = lambda iterable, **kwargs: iterable
    ns['RUN_TAG'] = 'full' if options['full'] else 'pilot'
    ns['INTEGRATION_BLOCKERS'].append('Offline snapshot reconstruction requires final reload of pinned upstream fields')
    if not options['full']: ns['INTEGRATION_BLOCKERS'].append('Smoke-size run is not a final research handoff')
    ns['revisions'] = {key: ns[key] for key in ['ORCA_REVISION','NUMINA_REVISION','TOKENIZER_REVISION']}
    for source in inputs: ns[source + '_revision'] = ns[source.upper() + '_REVISION']
    ns['tokenizer_revision'] = ns['TOKENIZER_REVISION']
    ns['full_counts'] = {source: pq.ParquetFile(path).metadata.num_rows for source, path in inputs.items()}
    checkpoint_root = ns['RUN_ROOT'] / 'checkpoints'
    stages = {source: checkpoint_root / source for source in inputs}
    meter = MemoryMeter(args.max_memory_mb)
    started, metrics = time.perf_counter(), {'status': 'running', 'source_stages': {}}
    conn = None
    try:
        states = {}
        for source, path in inputs.items():
            print('Streaming source:', source, flush=True)
            states[source] = canonicalize_snapshot(source, path, stages[source], mapper, ns['RUN_FINGERPRINT'],
                limit=options['rows_per_source'], batch_size=options['batch_size'], observe=meter.observe,
                sampling=options['sampling'], seed=ns['SPLIT_SEED'],
                progress=lambda message: print(message, flush=True))
            ns[source + '_raw'] = range(states[source]['processed_rows'])
            metrics['source_stages'][source] = {key: states[source][key] for key in ['processed_rows','reused_rows','elapsed_seconds','changed_answers']}
        atomic_json(ns['REPORT_DIR'] / 'snapshot_provenance.json', ns['snapshot_binding'])
        atomic_json(ns['REPORT_DIR'] / 'repository_revisions.json', ns['revisions'])
        inventory = json.loads((previous / 'reports/dataset_inventory.json').read_text(encoding='utf-8'))
        for item in inventory:
            item['license_claim_in_v1'] = item.pop('license')
            item['license_provenance'] = 'Historical claim only; pinned upstream card verification pending'
        atomic_json(ns['REPORT_DIR'] / 'dataset_inventory.json', inventory)
        conn = open_catalog(stages, checkpoint_root / 'catalog.sqlite', ns['RUN_FINGERPRINT'], meter.observe)
        source_reviews(conn, stages, states, ns)
        checker = None
        if ns['SHARED_EQUIVALENCE_PATH'].exists():
            spec = importlib.util.spec_from_file_location('team_equivalence', ns['SHARED_EQUIVALENCE_PATH'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            checker = module.mathematically_equal
        else: ns['INTEGRATION_BLOCKERS'].append('Shared evaluator equivalence checker missing')
        print('Preparing exact groups and bounded candidate text', flush=True)
        prepare_candidates(conn, stages, ns, checker, meter.observe)
        for index in [20,22,24,26,28]:
            run(index)
            meter.observe()
        status_audit(conn, ns, ''.join(notebook['cells'][30]['source']), meter.observe)
        run(32)
        run(34)
        try: run(36)
        except RuntimeError as error:
            if 'DRAFT exports created' not in str(error): raise
            print('Expected evidence gate:', error, flush=True)
        metrics['status'] = 'draft_complete'
    except BaseException as error:
        metrics.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        if conn is not None: conn.close()
        meter.stop.set()
        meter.thread.join(timeout=1)
        metrics.update(elapsed_seconds=time.perf_counter() - started, peak_rss_mib=meter.peak / 2**20,
                       memory_limit_mib=args.max_memory_mb)
        atomic_json(ns['REPORT_DIR'] / 'rebuild_metrics.json', metrics)
        print('Run metrics:', json.dumps(metrics), flush=True)
        print('Run directory:', ns['RUN_ROOT'], flush=True)


if __name__ == '__main__': main()

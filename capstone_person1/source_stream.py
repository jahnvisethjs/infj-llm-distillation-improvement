"""Pinned upstream canonicalization, resumable by independently hashed shards."""
import json
import random
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1.canonical_data import canonicalize_orca, canonicalize_numina
from capstone_person1.pipeline_controls import file_sha256, sha256_text
from capstone_person1.snapshot_stream import atomic_json, record_schema


def source_binding(project, source, inventory, config):
    spec = inventory[source]
    if any(spec[key] != config['sources'][source][key] for key in ['dataset_id','revision','split']):
        raise ValueError('Source inventory differs from pinned configuration')
    files = []
    for item in spec['files']:
        path = Path(project) / item['local_path']
        digest = file_sha256(path)
        if digest != item['sha256'] or digest != item['lfs_sha256']:
            raise ValueError('Pinned source file checksum mismatch: ' + str(path))
        files.append(dict(item, rows=pq.ParquetFile(path).metadata.num_rows))
    return dict(spec, files=files, rows=sum(item['rows'] for item in files))


def canonicalize_upstream(project, source, binding, config, stage_dir, *, limit=None, batch_size=4096):
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    module_dir = Path(__file__).parent
    signature = {'source': binding, 'config': {key:config[key] for key in ['sources','exclude_synthetic_numina','split_seed']}, 'limit':limit, 'batch_size':batch_size,
                 'code':{name:file_sha256(module_dir/name) for name in
                         ['source_stream.py','canonical_data.py','pipeline_controls.py','instance_matching.py']}}
    stamp = sha256_text(json.dumps(signature, sort_keys=True))
    checkpoint = stage_dir/'checkpoint.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['fingerprint'] != stamp: raise ValueError('Source checkpoint inputs or policy changed')
        for shard in state['shards']:
            if file_sha256(stage_dir/shard['name']) != shard['sha256']: raise ValueError('Canonical shard changed')
        if state['complete']:
            print(source + ': reusing ' + str(state['processed_rows']) + ' verified upstream rows', flush=True)
            return state
    else:
        state = {'fingerprint':stamp,'binding':signature,'processed_rows':0,'complete':False,'shards':[],
                 'eligible_rows':0,'quality_counts':{},'extraction_counts':{},'source_counts':{}}
        atomic_json(checkpoint,state)
    total = min(limit,binding['rows']) if limit is not None else binding['rows']
    positions = set(random.Random(config['split_seed']).sample(range(binding['rows']), total)) if limit else None
    cursor, sample_cursor, records = 0, 0, []
    counts = {key:Counter(state[key]) for key in ['quality_counts','extraction_counts','source_counts']}
    mapper = canonicalize_orca if source == 'orca' else canonicalize_numina
    last = time.monotonic()
    def commit():
        nonlocal records,last
        if not records: return
        name = f'batch_{state["processed_rows"]:09d}.parquet'
        pending = stage_dir/(name+'.pending')
        pq.write_table(pa.Table.from_pylist(records,schema=record_schema(records[0])),pending,compression='zstd')
        pending.replace(stage_dir/name)
        state['shards'].append({'name':name,'rows':len(records),'sha256':file_sha256(stage_dir/name)})
        state['processed_rows'] += len(records)
        state.update({key:dict(value) for key,value in counts.items()})
        atomic_json(checkpoint,state)
        records = []
        if time.monotonic()-last >= 15:
            print(f'{source}: {state["processed_rows"]:,}/{total:,} committed; {state["eligible_rows"]:,} initially eligible',flush=True)
            last = time.monotonic()
    for item in binding['files']:
        for batch in pq.ParquetFile(Path(project)/item['local_path']).iter_batches(batch_size=batch_size,use_threads=False):
            raw_rows = batch.to_pylist()
            for offset,raw in enumerate(raw_rows):
                original_index = cursor+offset
                if positions is not None and original_index not in positions: continue
                sample_cursor += 1
                if sample_cursor <= state['processed_rows']: continue
                canonical = mapper(raw,original_index,spec=config['sources'][source],config=config)
                canonical['upstream_fields_json'] = json.dumps(raw,ensure_ascii=False,sort_keys=True)
                records.append(canonical)
                state['eligible_rows'] += int(canonical['eligible'])
                counts['quality_counts'].update(canonical['quality_flags'] or ['retained_without_flags'])
                counts['extraction_counts'].update([canonical['answer_parse_status']])
                counts['source_counts'].update([canonical['original_source']])
                if len(records) == batch_size: commit()
            cursor += batch.num_rows
    commit()
    state['complete'] = state['processed_rows'] == total
    if not state['complete']: raise ValueError('Upstream row coverage incomplete')
    atomic_json(checkpoint,state)
    print(f'{source}: completed {total:,} upstream rows, {state["eligible_rows"]:,} initially eligible',flush=True)
    return state


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--release',required=True)
    parser.add_argument('--pilot',type=int)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    config = json.loads((project/'configs/person1_data_config.json').read_text(encoding='utf-8'))
    inventory = json.loads((project/'sources/source_inventory.json').read_text(encoding='utf-8'))
    release = Path(args.release).resolve()
    for source in config['sources']:
        binding = source_binding(project,source,inventory,config)
        canonicalize_upstream(project,source,binding,config,release/'processed'/('canonical_'+source),limit=args.pilot)


if __name__ == '__main__': main()

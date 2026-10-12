"""Global exact groups on disk; only checked/reviewed candidate text enters memory."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1.instance_matching import context_equivalent
from capstone_person1.pipeline_controls import file_sha256
from capstone_person1.snapshot_stream import atomic_json, stage_batches, selected_records


def build_catalog(stages, path):
    path = Path(path)
    checkpoint = path.with_suffix('.json')
    bindings = {key:file_sha256(Path(value)/'checkpoint.json') for key,value in stages.items()}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['bindings'] != bindings or state['sha256'] != file_sha256(path): raise ValueError('Catalog inputs/content changed')
        return sqlite3.connect(path)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('DROP TABLE IF EXISTS rows')
    conn.execute('CREATE TABLE rows(row_id TEXT PRIMARY KEY,dataset_id TEXT,problem_hash TEXT,answer_key TEXT,final_answer TEXT,problem TEXT,eligible INTEGER,quality_flags TEXT)')
    cols = ['row_id','dataset_id','problem_hash','answer_key','final_answer','problem','eligible','quality_flags']
    for source,stage in stages.items():
        for batch in stage_batches(stage,cols,batch_size=4096):
            values = batch.to_pylist()
            conn.executemany('INSERT INTO rows VALUES(?,?,?,?,?,?,?,?)',
                [tuple(json.dumps(row[key]) if key=='quality_flags' else row[key] for key in cols) for row in values])
        print(source + ': global metadata catalog loaded',flush=True)
    conn.execute('CREATE INDEX by_hash ON rows(problem_hash,eligible)')
    conn.execute('CREATE INDEX by_source ON rows(dataset_id,eligible)')
    conn.commit()
    atomic_json(checkpoint,dict(bindings=bindings,sha256=file_sha256(path)))
    return conn


def choose_candidates(conn, stages, protected_hashes, report_dir, *, maximum=45000, seed=42):
    conflicts, audits = set(), []
    for problem_hash,count in conn.execute('SELECT problem_hash,COUNT(DISTINCT answer_key) FROM rows WHERE eligible=1 GROUP BY problem_hash HAVING COUNT(DISTINCT answer_key)>1'):
        answers = conn.execute('SELECT row_id,problem,final_answer FROM rows WHERE problem_hash=? AND eligible=1 ORDER BY row_id',[problem_hash]).fetchall()
        comparisons = [context_equivalent(answers[0][1],answers[0][2],row[2]) for row in answers[1:]]
        state = 'equivalent' if all(value is True for value in comparisons) else 'proven_scalar_conflict' if any(value is False for value in comparisons) else 'unresolved_equivalence'
        if state!='equivalent': conflicts.add(problem_hash)
        audits.append(dict(problem_hash=problem_hash,state=state,row_ids=[row[0] for row in answers],answers=[row[2] for row in answers]))
    pd.DataFrame(audits,columns=['problem_hash','state','row_ids','answers']).to_csv(Path(report_dir)/'exact_answer_equivalence.csv',index=False)
    conn.execute('CREATE TEMP TABLE excluded_hash(problem_hash TEXT PRIMARY KEY,reason TEXT)')
    conn.executemany('INSERT INTO excluded_hash VALUES(?,?)',[(item,'conflicting_or_unresolved_exact_answers') for item in sorted(conflicts)])
    conn.executemany('INSERT OR IGNORE INTO excluded_hash VALUES(?,?)',[(item,'exact_protected_overlap') for item in sorted(protected_hashes)])
    conn.execute('CREATE TEMP TABLE representatives AS SELECT problem_hash,MIN(row_id) row_id FROM rows r WHERE eligible=1 AND NOT EXISTS(SELECT 1 FROM excluded_hash e WHERE e.problem_hash=r.problem_hash) GROUP BY problem_hash')
    conn.execute('CREATE INDEX reps_id ON representatives(row_id)')
    supply, chosen = {}, []
    for source,stage in stages.items():
        ids = [row[0] for row in conn.execute('SELECT r.row_id FROM rows r JOIN representatives p ON r.row_id=p.row_id WHERE r.dataset_id=? ORDER BY r.row_id',[source])]
        supply[source] = len(ids)
        ids = [ids[int(index)] for index in np.random.default_rng(seed).permutation(len(ids))[:maximum]]
        chosen.extend(selected_records(stage,ids))
    return pd.DataFrame(chosen).sort_values('row_id').reset_index(drop=True),dict(exact_conflict_groups=len(conflicts),unique_available_by_source=supply,
        checked_candidate_counts=pd.Series([row['dataset_id'] for row in chosen]).value_counts().to_dict(),global_exact_rows=conn.execute('SELECT COUNT(*) FROM rows').fetchone()[0],
        duplicate_exact_groups=conn.execute('SELECT COUNT(*) FROM (SELECT problem_hash FROM rows GROUP BY problem_hash HAVING COUNT(*)>1)').fetchone()[0],
        exact_protected_rows=conn.execute('SELECT COUNT(*) FROM rows WHERE problem_hash IN(SELECT problem_hash FROM excluded_hash WHERE reason="exact_protected_overlap")').fetchone()[0])


def export_row_status(conn, checked, banks, processed_dir):
    hashes = checked.set_index('problem_hash')[['problem_group_id','group_representative_id','group_excluded']].to_dict('index')
    chosen = {row.row_id:name for name,frame in banks.items() for row in frame.itertuples(index=False)}
    records, writer = [], None
    path = Path(processed_dir)/'row_status_audit.parquet'
    schema = pa.schema([('row_id',pa.string()),('dataset_id',pa.string()),('exact_group_id',pa.string()),('problem_group_id',pa.string()),
                        ('representative_id',pa.string()),('final_status',pa.string()),('quality_flags_json',pa.string())])
    query = 'SELECT r.row_id,r.dataset_id,r.problem_hash,r.eligible,r.quality_flags,p.row_id,e.reason FROM rows r LEFT JOIN representatives p ON p.problem_hash=r.problem_hash LEFT JOIN excluded_hash e ON e.problem_hash=r.problem_hash ORDER BY r.row_id'
    writer = pq.ParquetWriter(path,schema,compression='zstd')
    with writer:
        for row_id,source,problem_hash,eligible,flags,rep,excluded in conn.execute(query):
            group = hashes.get(problem_hash,{})
            exact_id = 'exact_'+problem_hash[:24]
            status = chosen.get(row_id,'checked_unused' if group else 'outside_checked_universe')
            if not eligible: status='quality_quarantine'
            elif excluded: status=excluded
            elif group.get('group_excluded'): status='near_review_quarantine_or_protected_overlap'
            elif rep and row_id!=rep: status='exact_duplicate_alternative'
            elif group and row_id!=group['group_representative_id']: status='near_duplicate_alternative'
            records.append(dict(row_id=row_id,dataset_id=source,exact_group_id=exact_id,problem_group_id=group.get('problem_group_id',exact_id),
                representative_id=group.get('group_representative_id',rep or ''),final_status=status,quality_flags_json=flags))
            if len(records)==8192:
                writer.write_table(pa.Table.from_pylist(records,schema=schema));records=[]
        if records: writer.write_table(pa.Table.from_pylist(records,schema=schema))
    # The status audit is the complete stable mapping, including non-candidates.
    table = pq.read_table(path,columns=['row_id','dataset_id','exact_group_id','problem_group_id','representative_id','final_status'])
    pq.write_table(table,Path(processed_dir)/'problem_group_mapping.parquet',compression='zstd')
    return table.column('final_status').to_pylist()

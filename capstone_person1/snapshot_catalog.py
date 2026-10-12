"""Disk-backed global grouping; only bounded candidate text is materialized."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from capstone_person1.pipeline_controls import answers_equivalent, file_sha256
from capstone_person1.snapshot_stream import atomic_json, stage_batches, selected_records

META_COLUMNS = ['row_id','dataset_id','original_source','original_row_id','problem_hash','problem_group_id',
                'answer_key','final_answer','solution_char_length','answer_parse_status','quality_flags','review_flags','eligible']


def open_catalog(stages, path, fingerprint, observe=lambda: None):
    path = Path(path)
    checkpoint = path.with_suffix('.checkpoint.json')
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['fingerprint'] != fingerprint or file_sha256(path) != state['sha256']:
            raise ValueError('Catalog checkpoint inputs or content changed')
        return sqlite3.connect(path)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA cache_size=-8192')
    conn.execute('PRAGMA temp_store=FILE')
    # This table belongs solely to an uncommitted stage; rerunning replaces partial rows.
    conn.execute('DROP TABLE IF EXISTS records')
    fields = ','.join(f'{col} ' + ('INTEGER' if col in {'eligible','solution_char_length'} else 'TEXT') for col in META_COLUMNS)
    conn.execute(f'CREATE TABLE records ({fields}, position INTEGER)')
    insert = 'INSERT INTO records VALUES (' + ','.join('?' for _ in range(len(META_COLUMNS) + 1)) + ')'
    for source, stage in stages.items():
        position = 0
        for batch in stage_batches(stage, META_COLUMNS):
            rows = []
            for row in batch.to_pylist():
                row['quality_flags'], row['review_flags'] = json.dumps(row['quality_flags']), json.dumps(row['review_flags'])
                rows.append(tuple(row[col] for col in META_COLUMNS) + (position,))
                position += 1
            conn.executemany(insert, rows)
            observe()
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS records_id ON records(row_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS records_hash ON records(problem_hash,eligible)')
    conn.execute('CREATE INDEX IF NOT EXISTS records_position ON records(dataset_id,position)')
    conn.commit()
    atomic_json(checkpoint, {'fingerprint': fingerprint, 'sha256': file_sha256(path)})
    return conn


def prepare_candidates(conn, stages, ns, shared_checker=None, observe=lambda: None):
    """Match the notebook's representative rule and seeded source shuffle using compact rows."""
    groups = conn.execute('SELECT problem_hash,COUNT(*),COUNT(DISTINCT answer_key) FROM records WHERE eligible=1 '
                          'GROUP BY problem_hash HAVING COUNT(*)>1').fetchall()
    report, audit, quarantined = [], [], []
    for problem_hash, count, answer_count in groups:
        rows = pd.read_sql_query('SELECT * FROM records WHERE eligible=1 AND problem_hash=?', conn, params=[problem_hash])
        state = 'same_normalized_answer'
        if answer_count > 1:
            answers = rows['final_answer'].unique()
            decisions = [answers_equivalent(answers[0], answer, shared_checker) for answer in answers[1:]]
            if all(value is True for value in decisions): answer_count, state = 1, 'equivalent'
            elif any(value is False for value in decisions): state = 'proven_scalar_or_checker_disagreement'
            else: state = 'unresolved_equivalence_quarantine'
            audit.append({'problem_hash': problem_hash, 'answers': json.dumps(answers.tolist()), 'equivalence_status': state})
        if answer_count > 1: quarantined.append(problem_hash)
        rows['duplicate_count'], rows['answer_count'] = count, answer_count
        rows['datasets'] = ','.join(sorted(set(rows['dataset_id'])))
        rows['conflicting_answers'] = answer_count > 1
        rows['equivalence_unresolved'] = state == 'unresolved_equivalence_quarantine'
        report.extend(rows.to_dict('records'))
    columns = META_COLUMNS + ['position','duplicate_count','answer_count','datasets','conflicting_answers','equivalence_unresolved']
    exact = pd.DataFrame(report, columns=columns).sort_values(['problem_hash','row_id'])
    exact.to_csv(ns['REPORT_DIR'] / 'exact_duplicates.csv', index=False)
    exact[exact['conflicting_answers'].astype(bool)].to_csv(ns['REPORT_DIR'] / 'conflicting_exact_answers.csv', index=False)
    pd.DataFrame(audit, columns=['problem_hash','answers','equivalence_status']).to_csv(
        ns['REPORT_DIR'] / 'exact_answer_equivalence_audit.csv', index=False)
    # TEMP tables leave the committed source catalog immutable for resume/hash checks.
    conn.execute('CREATE TEMP TABLE quarantined(problem_hash TEXT PRIMARY KEY)')
    conn.executemany('INSERT INTO quarantined VALUES (?)', [(value,) for value in quarantined])
    conn.execute('CREATE TEMP TABLE representatives AS SELECT problem_hash,MIN(row_id) AS row_id FROM records r '
                 'WHERE eligible=1 AND NOT EXISTS(SELECT 1 FROM quarantined q WHERE q.problem_hash=r.problem_hash) GROUP BY problem_hash')
    conn.execute('CREATE INDEX representatives_id ON representatives(row_id)')
    conn.execute('CREATE INDEX representatives_hash ON representatives(problem_hash)')
    ns['exact_report'] = exact
    ns['conflicting_row_ids'] = set(exact.loc[exact['conflicting_answers'].astype(bool), 'row_id'])
    ns['exact_alternative_ids'] = {row[0] for row in conn.execute(
        'SELECT r.row_id FROM records r JOIN representatives p ON p.problem_hash=r.problem_hash WHERE r.eligible=1 AND r.row_id!=p.row_id')}
    ns['eligible_all'] = range(conn.execute('SELECT COUNT(*) FROM records WHERE eligible=1').fetchone()[0])
    ids = {}
    for source in stages:
        values = [row[0] for row in conn.execute('SELECT r.row_id FROM records r JOIN representatives p ON p.row_id=r.row_id '
                                                'WHERE r.dataset_id=? ORDER BY position', [source])]
        n = min(ns['CANDIDATE_UNIVERSE_PER_SOURCE'], len(values))
        order = np.random.default_rng(ns['SELECTION_SEED']).permutation(len(values))[:n]
        ids[source] = [values[int(index)] for index in order]
        observe()
    ns['universe_per_source'] = min(map(len, ids.values()))
    candidates = []
    for source, stage in stages.items():
        ids[source] = ids[source][:ns['universe_per_source']]
        rows = selected_records(stage, ids[source])
        lookup = {row['row_id']: row for row in rows}
        candidates.extend(lookup[row_id] for row_id in ids[source])
    # Preserve provenance on the bounded candidate universe; the full corpus stays on disk.
    ns['candidate_df'] = pd.DataFrame(candidates)
    ns['candidate_universe_ids'] = set(ns['candidate_df']['row_id'])
    ns['exact_unique'] = range(conn.execute('SELECT COUNT(*) FROM representatives').fetchone()[0])
    observe()


def ordered_source_ids(conn, source, condition='1=1'):
    # Condition is an internal fixed SQL expression, never a user argument.
    return [row[0] for row in conn.execute(f'SELECT row_id FROM records WHERE dataset_id=? AND {condition} ORDER BY position', [source])]

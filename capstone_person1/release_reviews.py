"""Review queues are evidence, not automatic approvals. Preserve completed decisions."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from capstone_person1.pipeline_controls import sha256_text, file_sha256
from capstone_person1.snapshot_stream import atomic_json

REVIEW_VERSION = 'person1-release-review-v2'


def prior_rejections(release):
    """Prior inspected defects are mandatory inputs, even outside the new random sample."""
    path=Path(release)/'reports/prior_rejected_targets.json'
    if not path.exists():return []
    rows=json.loads(path.read_text(encoding='utf-8'))['records']
    seen=set()
    for row in rows:
        expected=sha256_text(json.dumps({key:row[key] for key in ['row_id','problem','solution','final_answer']},sort_keys=True,ensure_ascii=False))
        stamp=datetime.fromisoformat(row['reviewed_utc'].replace('Z','+00:00'))
        if (row['source_sha256']!=expected or row['decision'] not in {'quarantine','retain_quarantine'}
            or not row['reviewer'] or not row['notes'] or stamp.tzinfo is None or row['row_id'] in seen):
            raise ValueError('Invalid, duplicate or altered prior rejected target')
        seen.add(row['row_id'])
    return rows


def reviewed_rejected_groups(checked,conn,row_ids):
    hashes=set()
    row_ids=list(row_ids)
    for start in range(0,len(row_ids),900):
        part=row_ids[start:start+900]
        hashes.update(row[0] for row in conn.execute('SELECT problem_hash FROM rows WHERE row_id IN ('+','.join('?' for _ in part)+')',part))
    return set(checked.loc[checked['problem_hash'].isin(hashes),'problem_group_id'])


def sample_formatted(frame, count=100, seed=1729):
    frame=frame.copy()
    frame['_review_mode']=frame['answer_extraction_json'].map(lambda value:json.loads(value)['mode'])
    parts=[]
    for source,rows in frame.groupby('dataset_id'):
        selected=[]
        for mode,members in rows.groupby(['_review_mode','answer_type']):
            selected.extend(members.sample(n=min(3,len(members)),random_state=seed)['row_id'])
        remaining=rows[~rows['row_id'].isin(selected)]
        selected.extend(remaining.sample(n=min(max(0,count-len(selected)),len(remaining)),random_state=seed)['row_id'])
        parts.append(rows[rows['row_id'].isin(selected)].sort_values('row_id'))
    return pd.concat(parts,ignore_index=True).drop(columns='_review_mode')


def include_added_review_rows(sample,population,existing_ids):
    """Preserve explicitly inspected extra cases; bind their text to verified population."""
    if len(existing_ids)!=len(set(existing_ids)):raise ValueError('Duplicate added review IDs')
    required=set(sample['row_id'])
    extra=[row_id for row_id in existing_ids if row_id not in required]
    indexed=population.set_index('row_id',drop=False)
    if not set(extra).issubset(indexed.index):raise ValueError('Added review case is outside verified population')
    return pd.concat([sample,indexed.loc[extra].reset_index(drop=True)],ignore_index=True) if extra else sample


def make_review_queue(frame,path,kind):
    path=Path(path)
    records=[]
    for row in pa.Table.from_pandas(frame,preserve_index=False).to_pylist():
        records.append(dict(sample_id=kind+':'+row['row_id'],row_id=row['row_id'],dataset_id=row['dataset_id'],
            problem=row['problem'],solution=row['solution'],final_answer=row['final_answer'],completion=row.get('completion',''),
            extraction_mode=json.loads(row['answer_extraction_json'])['mode'],provided_answer_parse_status=row['answer_parse_status'],answer_type=row['answer_type'],quality_flags_json=json.dumps(row['quality_flags']),
            answer_evidence_text=row['answer_evidence_text'],supervised_tokens=row.get('supervised_tokens',0),
            source_sha256=sha256_text(json.dumps({key:row[key] for key in ['row_id','problem','solution','final_answer']},sort_keys=True,ensure_ascii=False)),
            reviewer='',reviewed_utc='',policy_version=REVIEW_VERSION,decision='',notes=''))
    current=pd.DataFrame(records)
    if path.exists():
        old=pd.read_csv(path,keep_default_na=False,dtype=str)
        if old[['sample_id','source_sha256']].to_dict('records')!=current[['sample_id','source_sha256']].to_dict('records'):
            raise ValueError('Review inputs changed; use a new release rather than overwriting completed reviews')
        return old
    current.to_csv(path,index=False,lineterminator='\n')
    return current


def validate_reviews(paths, *, required_formatted_per_source=100):
    pending,quarantined,evidence=[],set(),[]
    for path in paths:
        frame=pd.read_csv(path,keep_default_na=False,dtype=str)
        if frame.empty or frame['sample_id'].duplicated().any(): raise ValueError('Empty/duplicate mandatory review queue')
        if Path(path).name=='formatted_target_review.csv':
            if any(frame['dataset_id'].value_counts().get(source,0)<required_formatted_per_source for source in ['orca','numina']):
                pending.append('formatted_sample_coverage')
        for row in frame.to_dict('records'):
            expected=sha256_text(json.dumps({key:row[key] for key in ['row_id','problem','solution','final_answer']},sort_keys=True,ensure_ascii=False))
            if row['source_sha256']!=expected: raise ValueError('Stale or altered reviewed source text')
            if not row['reviewer'] or not row['reviewed_utc'] or not row['notes'] or row['decision'] not in {'accept_structural_target','retain_quarantine','quarantine'}:
                pending.append(row['sample_id']);continue
            if row['policy_version']!=REVIEW_VERSION: raise ValueError('Review policy differs')
            stamp=datetime.fromisoformat(row['reviewed_utc'].replace('Z','+00:00'))
            if stamp.tzinfo is None: raise ValueError('Review time must have timezone')
            if row['decision'] in {'quarantine','retain_quarantine'}: quarantined.add(row['row_id'])
            evidence.append(row)
    return dict(complete=not pending,pending=pending,quarantined_row_ids=sorted(quarantined),
                decisions=evidence,review_sha256={Path(path).name:file_sha256(path) for path in paths},
                policy_version=REVIEW_VERSION,mathematical_correctness='not_independently_certified',human_signoff=False)

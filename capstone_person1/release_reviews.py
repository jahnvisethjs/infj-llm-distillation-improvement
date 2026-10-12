"""Review queues are evidence, not automatic approvals. Preserve completed decisions."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from capstone_person1.pipeline_controls import sha256_text, file_sha256
from capstone_person1.snapshot_stream import atomic_json

REVIEW_VERSION = 'person1-release-review-v1'


def sample_formatted(frame, count=100, seed=1729):
    parts=[]
    for source,rows in frame.groupby('dataset_id'):
        selected=[]
        for mode,members in rows.groupby('answer_parse_status'):
            selected.extend(members.sample(n=min(3,len(members)),random_state=seed)['row_id'])
        remaining=rows[~rows['row_id'].isin(selected)]
        selected.extend(remaining.sample(n=min(max(0,count-len(selected)),len(remaining)),random_state=seed)['row_id'])
        parts.append(rows[rows['row_id'].isin(selected)].sort_values('row_id'))
    return pd.concat(parts,ignore_index=True)


def make_review_queue(frame,path,kind):
    path=Path(path)
    records=[]
    for row in pa.Table.from_pandas(frame,preserve_index=False).to_pylist():
        records.append(dict(sample_id=kind+':'+row['row_id'],row_id=row['row_id'],dataset_id=row['dataset_id'],
            problem=row['problem'],solution=row['solution'],final_answer=row['final_answer'],completion=row.get('completion',''),
            extraction_mode=row['answer_parse_status'],quality_flags_json=json.dumps(row['quality_flags']),
            answer_evidence_text=row['answer_evidence_text'],supervised_tokens=row.get('supervised_tokens',0),
            source_sha256=sha256_text(json.dumps({key:row[key] for key in ['row_id','problem','solution','final_answer']},sort_keys=True,ensure_ascii=False)),
            reviewer='',reviewed_utc='',policy_version=REVIEW_VERSION,decision='',notes=''))
    current=pd.DataFrame(records)
    if path.exists():
        old=pd.read_csv(path,keep_default_na=False)
        if old[['sample_id','source_sha256']].to_dict('records')!=current[['sample_id','source_sha256']].to_dict('records'):
            raise ValueError('Review inputs changed; use a new release rather than overwriting completed reviews')
        return old
    current.to_csv(path,index=False,lineterminator='\n')
    return current


def validate_reviews(paths, *, required_formatted_per_source=100):
    pending,quarantined,evidence=[],set(),[]
    for path in paths:
        frame=pd.read_csv(path,keep_default_na=False)
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

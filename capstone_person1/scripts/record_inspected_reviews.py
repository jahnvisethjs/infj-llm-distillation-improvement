"""Save explicit case-specific decisions after inspection; never infer approvals."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',required=True)
    parser.add_argument('--decisions',required=True)
    args=parser.parse_args()
    root=Path(args.release)/'reports'
    decisions=json.loads(Path(args.decisions).read_text(encoding='utf-8'))
    names={'formatted':'formatted_target_review.csv','risk':'answer_extraction_review.csv','prior':'prior_target_review.csv',
        'pairs':'duplicate_and_overlap_reports/pair_policy_review.csv'}
    for kind,records in decisions.items():
        if not records:continue
        path=root/names[kind]
        frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        for record in records:
            index=record['index']
            id_field='row_id_a' if kind=='pairs' else 'row_id'
            if not frame.loc[index,id_field].startswith(record['id_prefix']):
                raise ValueError('Inspected case does not match queue row')
            field='review_decision' if kind=='pairs' else 'decision'
            if frame.loc[index,field] and frame.loc[index,'notes']!=record['notes']:
                raise ValueError('Completed review cannot be overwritten')
            if frame.loc[index,field]:
                if frame.loc[index,field]!=record['decision']:raise ValueError('Completed decision cannot be changed')
                continue
            if not record['notes'].strip():raise ValueError('Case-specific evidence required')
            frame.loc[index,field]=record['decision']
            frame.loc[index,'notes']=record['notes']
            frame.loc[index,'reviewer']='Codex'
            frame.loc[index,'reviewed_utc']=datetime.now(timezone.utc).isoformat()
        frame.to_csv(path,index=False,lineterminator='\n')
        print(kind, 'saved',len(records),'explicit decisions;',sum(frame['reviewer'].ne('')),'reviewed of',len(frame))

if __name__=='__main__':main()

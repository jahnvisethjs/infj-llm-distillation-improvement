"""Reuse saved inspections only for byte-identical, hash-bound source targets."""
import argparse
import json
from pathlib import Path
import pandas as pd
from capstone_person1.pipeline_controls import sha256_text
from capstone_person1.release_reviews import REVIEW_VERSION

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True)
    args=parser.parse_args();root=Path(args.release)
    manifest=root/'metadata/release_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['immutable_after_finalization']:
        raise ValueError('Cannot modify finalized release')
    history=json.loads((root/'reports/reused_inspection_history.json').read_text(encoding='utf-8'))['records']
    prior=root/'reports/prior_rejected_targets.json'
    if prior.exists():history+=json.loads(prior.read_text(encoding='utf-8'))['records']
    inspected={}
    for row in history:
        if not row.get('reviewer') or not row.get('reviewed_utc') or not row.get('notes') or row.get('policy_version')!=REVIEW_VERSION:continue
        if row.get('decision') not in {'accept_structural_target','quarantine','retain_quarantine'}:continue
        signature=sha256_text(json.dumps({key:row[key] for key in ['row_id','problem','solution','final_answer']},sort_keys=True,ensure_ascii=False))
        if signature!=row['source_sha256']:raise ValueError('Altered inspection history')
        old=inspected.get(signature)
        if old and old['decision']!=row['decision']:raise ValueError('Conflicting prior inspections require explicit resolution')
        inspected[signature]=row
    for name in ['formatted_target_review.csv','answer_extraction_review.csv','prior_target_review.csv']:
        path=root/'reports'/name
        if not path.exists():continue
        frame=pd.read_csv(path,dtype=str,keep_default_na=False);reused=0
        for index,row in frame.iterrows():
            old=inspected.get(row['source_sha256'])
            if old is None or row['reviewer']:continue
            for key in ['reviewer','reviewed_utc','decision','notes']:
                frame.loc[index,key]=old[key]
            frame.loc[index,'review_origin']='Preserved Codex inspection of identical source target; history in reused_inspection_history.json or prior_rejected_targets.json'
            reused+=1
        frame.to_csv(path,index=False,lineterminator='\n')
        print(name, 'reused',reused,'actual identical-target inspections; unsigned cases remain unsigned')

if __name__=='__main__':main()

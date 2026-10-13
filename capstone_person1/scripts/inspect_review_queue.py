"""Print actual queued cases for inspection; never populates review decisions."""
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True)
    parser.add_argument('--queue',choices=['formatted','risk','pairs','prior'],default='formatted')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--compact',action='store_true',help='Print full source text without JSON escaping or repeated field names')
    args=parser.parse_args();root=Path(args.release)/'reports'
    name={'formatted':'formatted_target_review.csv','risk':'answer_extraction_review.csv','prior':'prior_target_review.csv','pairs':'duplicate_and_overlap_reports/pair_policy_review.csv'}[args.queue]
    frame=pd.read_csv(root/name,dtype=str,keep_default_na=False)
    print('Queue rows:',len(frame))
    for index,row in frame.iloc[args.start:args.start+args.count].iterrows():
        if args.compact and args.queue!='pairs':
            print(f"\nCASE {index} {row['row_id']} {row['dataset_id']} {row['extraction_mode']} {row['answer_type']} decision={row['decision'] or 'PENDING'}")
            print('QUESTION:',row['problem'])
            print('SOLUTION:',row['solution'])
            print('TARGET:',row['final_answer'],'FLAGS:',row['quality_flags_json'])
            print('COMPLETION END:',row['completion'][-100:])
            continue
        if args.queue=='pairs':
            keys=['row_id_a','row_id_b','kind','action','similarity','retrieval','problem_a','problem_b']
        else:keys=['row_id','dataset_id','extraction_mode','answer_type','problem','solution','final_answer','quality_flags_json']
        record={key:row[key] for key in keys}
        record['index']=index
        if args.queue!='pairs':record['completion_end']=row['completion'][-160:]
        print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':main()

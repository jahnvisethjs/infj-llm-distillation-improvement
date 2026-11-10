#!/usr/bin/env python3
"""Strict paired comparisons of real evaluator outputs. No placeholder scores."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import HERE,read,write_new,sha

COMPARE_FIELDS=('model','revision','precision','max_new_tokens','batch_size','stop_at_question',
                'question_text','prompt_sha256','add_special_tokens','decoding','eos_token_id',
                'transformers','peft','torch','gpu','cuda','parser_version','math_verify','eval_file_sha256',
                'answer_parser_sha256','evaluate_py_sha256')


def validate_pair(parent_path,B_path,require_full_frozen_set=True):
    parent_path,B_path=Path(parent_path),Path(B_path)
    def load(path):
        if not path.is_file():raise FileNotFoundError('Real evaluation predictions are missing: '+str(path))
        with path.open(encoding='utf-8') as file:
            rows=[json.loads(line) for line in file if line.strip()]
        config=read(path.with_suffix('.config.json'))
        ids=[r['problem_id'] for r in rows]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Empty or repeated prediction IDs')
        return rows,config,set(ids)
    a,ac,aid=load(parent_path);b,bc,bid=load(B_path)
    if aid!=bid:raise ValueError('Parent and B were not evaluated on the same question IDs')
    missing=[k for k in COMPARE_FIELDS if k not in ac or k not in bc]
    if missing:raise ValueError('Missing evaluation settings: '+', '.join(missing))
    mismatch=[k for k in COMPARE_FIELDS if ac[k]!=bc[k]]
    if mismatch:raise ValueError('Incompatible evaluation settings: '+', '.join(mismatch))
    if require_full_frozen_set:
        frozen=next((p for p in (HERE/'vendor/evaluation/data').glob('*.jsonl') if sha(p)==ac['eval_file_sha256']),None)
        if frozen is None:raise ValueError('Evaluation file hash is not a pinned question set')
        with frozen.open(encoding='utf-8') as file:
            expected={json.loads(line)['problem_id'] for line in file if line.strip()}
        if aid!=expected:raise ValueError('Predictions do not cover the complete frozen evaluation question set')
    def accuracy(rows):
        if any(type(r.get('correct')) is not bool for r in rows):raise ValueError('Missing/nonboolean correctness labels')
        return sum(r['correct'] for r in rows)/len(rows)
    pa,pb=accuracy(a),accuracy(b)
    return {'questions':len(a),'parent_accuracy':pa,'B_accuracy':pb,'change_percentage_points':100*(pb-pa),
            'benchmark':a[0].get('benchmark'),'settings_compatible':True,
            'complete_frozen_set_verified':require_full_frozen_set,'real_predictions_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--B',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--paired-statistics',action='store_true');a=p.parse_args()
    report=validate_pair(a.parent,a.B);write_new(a.out,report);print(json.dumps(report,indent=2))
    if a.paired_statistics:
        subprocess.run([sys.executable,str(HERE/'vendor/evaluation/metrics.py'),str(a.parent),str(a.B),'--compare'],check=True)


if __name__=='__main__':main()

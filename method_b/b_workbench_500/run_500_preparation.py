#!/usr/bin/env python3
"""Prepare the extra 0.5x budget using the completed SOL CPU environment/cache.

The original b_workbench and B_workbench_v1 outputs are never changed.
No package installation, model loading, training, or evaluation is launched.
"""
import argparse
import json
import subprocess
from pathlib import Path
from common import HERE, work_root


def run(root, offline=True):
    root=Path(root)
    python=root/'venv_b_training_py312/bin/python'
    if not python.is_file():
        raise FileNotFoundError('The previously prepared SOL Python environment is absent; complete the original CPU setup first')
    logs=work_root(root)/'add_500_logs';logs.mkdir(parents=True,exist_ok=True)
    progress={'kind':'R1000_half_budget_extension','training_started':False,
              'starting_parent_examples':1000,'budget_multipliers':[0.5,1,2,4],
              'original_outputs_preserved':True,'offline_tokenizers':offline,'stages':[]}
    def save():
        (logs/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    def stage(name,command):
        print('\n--- '+name+' ---',flush=True)
        item={'stage':name,'status':'running','command':command};progress['stages'].append(item);save()
        with (logs/(name+'.log')).open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,cwd=str(HERE),stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
        item.update(status='passed' if code==0 else 'failed',returncode=code);save()
        if code:raise RuntimeError(f'{name} failed; see '+str(logs/(name+'.log')))
    base=[str(python),str(HERE/'workbench.py'),'--root',str(root)]
    stage('extension_tests',[str(python),'-m','unittest','test_original_allocator','test_workbench','test_half_extension','-q'])
    stage('four_data_selections',base+['prepare'])
    stage('four_budget_tokenizer_verification',base+(['--offline'] if offline else [])+['verify'])
    stage('development_evaluation_plans',base+['plan-evaluations'])
    stage('readiness_report',base+['status'])
    progress['CPU_extension_complete']=True;save()
    print('\n500-BUDGET PREPARATION COMPLETE. No model training or evaluation started.',flush=True)
    return progress


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--allow-downloads',action='store_true',help='Permit tokenizer downloads if the previously completed cache was removed')
    args=parser.parse_args();run(args.root,offline=not args.allow_downloads)

#!/usr/bin/env python3
"""Run the CPU preparation batch without any trained parent adapter.

Installs into a dedicated project environment, verifies tokenizer/labels, and
optionally caches public base weights. No training or generation is launched.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import HERE,sha,write_new,work_root


def run(root,skip_prefetch=False,skip_install=False):
    root=Path(root);destination=work_root(root)/'preparation_logs';destination.mkdir(parents=True,exist_ok=True)
    progress={'kind':'weights_free_CPU_preparation','trained_parent_required':False,'training_started':False,'stages':[]}
    def stage(name,command,cwd=HERE):
        print('\n--- '+name+' ---',flush=True)
        item={'stage':name,'status':'running','command':command};progress['stages'].append(item)
        (destination/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        with (destination/(name+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        item['status']='passed' if code==0 else 'failed';item['returncode']=code
        (destination/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        if code:raise RuntimeError(f'{name} failed; see '+str(destination/(name+'.log')))
    python=root/'venv_b_training_py312/bin/python'
    if not skip_install:
        stage('software_environment',[sys.executable,str(HERE/'bootstrap_env.py'),'--root',str(root)])
    elif not python.is_file():
        raise FileNotFoundError('Requested --skip-install but the prepared Python environment is absent')
    # The pinned encoder imports pandas. Bootstrap before any encoder tests,
    # and use the prepared Python instead of the dependency-light Jupyter host.
    stage('offline_tests',[str(python),'-m','unittest','test_original_allocator','test_workbench','-q'])
    stage('real_data_selection',[str(python),str(HERE/'workbench.py'),'--root',str(root),'prepare'])
    stage('CPU_tensor_loss_checks',[str(python),'-m','unittest','test_tensor_loss','-q'])
    stage('shared_grader_tests',[str(python),'-m','pytest',str(HERE/'vendor/evaluation/test_answer_parser.py'),'-q'],HERE/'vendor/evaluation')
    stage('real_tokenizer_verification',[str(python),str(HERE/'workbench.py'),'--root',str(root),'verify'])
    if not skip_prefetch:
        stage('public_base_cache',[str(python),str(HERE/'workbench.py'),'--root',str(root),'prefetch'])
    stage('development_evaluation_plans',[str(python),str(HERE/'workbench.py'),'--root',str(root),'plan-evaluations'])
    stage('readiness_report',[str(python),str(HERE/'workbench.py'),'--root',str(root),'status'])
    progress['weights_free_preparation_complete']=True
    progress['public_base_prefetch_requested']=not skip_prefetch
    (destination/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print('\nWEIGHTS-FREE PREPARATION COMPLETE. Parent-dependent model smoke/training/evaluation have not run.',flush=True)
    return progress


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--skip-prefetch',action='store_true');p.add_argument('--skip-install',action='store_true');a=p.parse_args();run(a.root,a.skip_prefetch,a.skip_install)

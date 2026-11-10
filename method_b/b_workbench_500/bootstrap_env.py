#!/usr/bin/env python3
"""Create the isolated SOL training/evaluation environment on a CPU session.

Installs software only. It never starts a model, a training job, or a GPU job.
The notebook's current Python/kernel is unchanged.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PINS={'torch':'2.6.0+cu124','transformers':'4.57.1','peft':'0.17.1','accelerate':'1.10.1',
      'bitsandbytes':'0.48.1','safetensors':'0.8.0','numpy':'2.5.4','tokenizers':'0.22.2',
      'huggingface-hub':'0.36.2','math-verify':'0.9.0','pandas':'2.2.3'}


def setup(root,check_only=False):
    root=Path(root);environment=root/'venv_b_training_py312';python=environment/'bin/python'
    if not python.is_file():
        if check_only:return {'installed':False,'python':str(python)}
        choices=[Path('/packages/apps/jupyter/2025-03-24/bin/python3.12')]
        if sys.version_info[:2]==(3,12):choices.append(Path(sys.executable))
        found=shutil.which('python3.12')
        if found:choices.append(Path(found))
        base=next((p for p in choices if p.is_file()),None)
        if base is None:raise RuntimeError('Python 3.12 was not found; use the known SOL Python 3.12 module/executable')
        subprocess.run([str(base),'-m','venv',str(environment)],check=True)
    subprocess.run([str(python),'-c',"import sys; assert sys.version_info[:2]==(3,12), 'Wrong Python version in project environment'"],check=True)
    probe="""import json,sys
from importlib import metadata
pins=json.loads(sys.argv[1]);out={}
for name in pins:
 try:out[name]=metadata.version(name)
 except metadata.PackageNotFoundError:out[name]=None
print(json.dumps(out))
"""
    actual=json.loads(subprocess.check_output([str(python),'-c',probe,json.dumps(PINS)],text=True))
    ready=actual==PINS
    if not ready and not check_only:
        # Reproduce the working Person 2 CUDA build, in this isolated environment.
        subprocess.run([str(python),'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'],check=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(HERE/'requirements-training.txt')],check=True)
        subprocess.run([str(python),'-m','pip','check'],check=True)
        actual=json.loads(subprocess.check_output([str(python),'-c',probe,json.dumps(PINS)],text=True))
        ready=actual==PINS
        if not ready:raise RuntimeError('Installed versions do not match the recorded pins: '+json.dumps(actual))
    report={'installed':ready,'python':str(python),'packages':actual,'pins':PINS,'gpu_required_for_installation':False,
            'runtime_imports_passed':False,
            'gpu_model_execution_tested':False,'evaluation_note':'Use this same environment/GPU policy for fresh parent and B evaluations; historical prediction settings may differ.'}
    if not check_only:
        destination=root/'environment_reports';destination.mkdir(parents=True,exist_ok=True)
        (destination/'requirements.lock.txt').write_text(subprocess.check_output([str(python),'-m','pip','freeze'],text=True))
        result=subprocess.run([str(python),'-c',"from transformers import AutoTokenizer; import torch, peft, pandas; from common import load_person2; load_person2(); print('Imports and original encoder passed; CUDA available in this session:', torch.cuda.is_available())"],cwd=str(HERE))
        report['runtime_imports_passed']=result.returncode==0
        (destination/'environment.json').write_text(json.dumps(report,indent=2)+'\n')
        if result.returncode:raise RuntimeError('Package installation finished but runtime imports failed; inspect environment_reports/environment.json and the import error')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args();setup(a.root,a.check_only)

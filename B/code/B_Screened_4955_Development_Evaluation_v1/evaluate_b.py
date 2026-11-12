#!/usr/bin/env python3
"""Durable screened B training followed by frozen GSM/MATH development evaluation."""
import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
import zipfile

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
VERSION='screened-b-4955-development-evaluation-v1'
MAX_ATTEMPTS=3
BENCHMARKS={'dev_gsm':'3cb2c71007d23e01cb70b6370c1ad82ad5bcea3527c7f3258b4814cf2e3d1cc6',
            'dev_math':'641460fdc55beff591daad0cc7530a3f933eb263c3dfdf8a0534fcfaf8196416'}
GENERATION_FIELDS=('model','revision','precision','max_new_tokens','batch_size','stop_at_question',
    'question_text','prompt_sha256','add_special_tokens','decoding','eos_token_id','adapter_mode',
    'transformers','peft','torch','gpu','cuda')

class IntegrityError(ValueError):pass
class SelectionBlocked(IntegrityError):pass

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def package_check():
    for name,wanted in read(HERE/'PACKAGE_PINS.json').items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise IntegrityError('Invalid package path')
        check_hash(HERE/rel,wanted)
    return sha(HERE/'PACKAGE_PINS.json')

def expected():return read(HERE/'expected_evaluation.json')

def training(a):
    package_check()
    path=Path(a.work)/'B_Screened_4955_Controls_v1'
    check_hash(path/'PACKAGE_PINS.json',expected()['training_package_pins_sha256'])
    obj=load_module('_screened_B_training_unchanged',path/'screened_b.py');obj.package_check()
    return obj

def control_root(a):
    return Path(a.work).resolve()/'outputs/B_screened_4955_development_evaluation_v1'/digest({
        'package':package_check(),'work':str(Path(a.work).resolve()),'b_root':str(Path(a.b_root).resolve())})[:16]

def preflight(a):
    package_check();set_offline(a)
    if not 1<=a.eval_allocations<=12 or not 1800<=a.worker_seconds<=13800:
        raise IntegrityError('Use 1-12 evaluation allocations with 1800-13800 worker seconds')
    for command in ['sbatch','squeue','scontrol','scancel']:
        if shutil.which(command) is None:raise IntegrityError('Run on Sol; Slurm command unavailable: '+command)
    py=Path(a.b_root)/'venv_b_training_py312/bin/python'
    if Path(sys.prefix)!=py.parent.parent:raise IntegrityError('Use the existing B training Python: '+str(py))
    e=expected()
    for name in ['torch','transformers','peft']:
        if importlib.metadata.version(name)!=e['generation_contract'][name]:raise IntegrityError('Evaluation environment changed: '+name)
    if importlib.metadata.version('math-verify')!=e['generation_contract']['math_verify']:raise IntegrityError('Math checker version differs')
    t=training(a);t.original_pilot(a)
    root=control_root(a)
    for filename,wanted in e['evaluation_sources'].items():check_hash(runtime_root(a)/'vendor/evaluation'/filename,wanted)
    for bench,wanted in BENCHMARKS.items():
        source=runtime_root(a)/'vendor/evaluation/data'/(bench+'.jsonl');check_hash(source,wanted)
        save_bytes(root/'questions'/(bench+'.jsonl'),source.read_bytes());question_file(a,bench)
    pilot=e['screened_B_pilot'];verify_checkpoint(pilot)
    for bench in BENCHMARKS:
        path=Path(a.work)/'outputs/C0_screened_pilot_v1/0.5B_seed17/be46c67690fb33ba/development/pilot01/B'/(bench+'.jsonl')
        validate_predictions(a,pilot,bench,rows(path),read(path.with_suffix('.config.json')),allow_checker_errors=True)
    save(root/'PREFLIGHT.json',{'version':VERSION,'passed':True,'questions_sha256':BENCHMARKS,
        'required_prediction_files':24,'new_GPU_prediction_files':22,'pilot_files_reused':2,
        'training_package_pins_sha256':e['training_package_pins_sha256']})
    print('Development preflight passed: frozen 500 GSM + 500 MATH; 22 new evaluation files and 2 reused pilot files.',flush=True)

def make_plan(a):
    t=training(a);t.validate_final_marker(a)
    result=[]
    for row in t.collect(a):
        if not row['complete']:raise IntegrityError('Training checkpoint is incomplete')
        profile=f"{row['model_size']}_seed{row['parent_seed']}";entry=expected()['parents'][profile]
        receipt=row['receipt'];directory=Path(row['run_directory']);adapter=Path(receipt['checkpoint'])/'adapter'
        target={'key':f"B_screened_{row['model_size']}_p{row['parent_seed']}_t{row['continuation_seed']}",
            'method':'B_screened','profile':profile,'model_size':row['model_size'],'parent_seed':row['parent_seed'],
            'continuation_seed':row['continuation_seed'],'model':entry['model'],'revision':entry['revision'],
            'adapter':str(adapter),'checkpoint_sha256':receipt['adapter_sha256'],'manifest_sha256':t.B_MANIFEST_SHA,
            'bank_sha256':t.BANK_SHA,'parent_sha256':entry['expected_adapter_sha256'],
            'parent_adapter':str(Path(a.b_root)/entry['adapter_relative_path']),
            'adapter_config_sha256':sha(adapter/'adapter_config.json'),
            'checkpoint_record_sha256':sha(adapter.parent/'run_config.json'),
            'actual_supervised_tokens':t.ACTUAL_B_TOKENS,'reused_original_pilot':row['reused_pilot'],
            'training_receipt':str(directory/'completion.json'),'training_receipt_sha256':sha(directory/'completion.json')}
        verify_checkpoint(target);result.append(target)
    if len(result)!=12 or len({r['key'] for r in result})!=12:raise IntegrityError('Expected twelve distinct screened B checkpoints')
    return {'version':VERSION,'targets':result,'training_control_directory':str(t.control_root(a)),
        'training_complete_sha256':sha(t.control_root(a)/'TRAINING_COMPLETE.json')}

def plan(a):
    p=read(control_root(a)/'PLAN.json');freeze=read(control_root(a)/'FREEZE.json')
    if freeze!={'version':VERSION,'package_sha256':package_check(),'plan_sha256':digest(p),'questions_sha256':BENCHMARKS}:
        raise IntegrityError('Frozen checkpoint plan changed')
    check_hash(Path(p['training_control_directory'])/'TRAINING_COMPLETE.json',p['training_complete_sha256'])
    for target in p['targets']:check_hash(target['training_receipt'],target['training_receipt_sha256'])
    return p

def initialize(a):
    root=control_root(a)
    with locked(root/'PREPARE.lock'):
        current=make_plan(a)
        save(root/'PLAN.json',current)
        save(root/'FREEZE.json',{'version':VERSION,'package_sha256':package_check(),'plan_sha256':digest(current),'questions_sha256':BENCHMARKS})
        p=plan(a)
        for target in p['targets']:
            if target['reused_original_pilot']:
                for bench in BENCHMARKS:
                    source=Path(a.work)/'outputs/C0_screened_pilot_v1/0.5B_seed17/be46c67690fb33ba/development/pilot01/B'/(bench+'.jsonl')
                    finalize_result(a,target,bench,source,source.with_suffix('.config.json'),'reused_completed_screened_B_pilot')
    return p

def child(a,action,**values):
    cmd=[sys.executable,'-u','-B',str(HERE/'evaluate_b.py'),action,'--work',str(a.work),'--b-root',str(a.b_root)]
    for key,value in values.items():cmd+=['--'+key.replace('_','-'),str(value)]
    return cmd

def completed(a):
    records=[]
    for target in plan(a)['targets']:
        verify_checkpoint(target)
        for bench in BENCHMARKS:
            if (result_root(a,target,bench)/'completion.json').is_file():
                predictions,config,receipt=validate_result(a,target,bench)
                records.append({'key':target['key'],'model_size':target['model_size'],'parent_seed':target['parent_seed'],
                    'continuation_seed':target['continuation_seed'],'benchmark':bench,**metrics(predictions),
                    'checkpoint_sha256':target['checkpoint_sha256'],'manifest_sha256':target['manifest_sha256'],
                    'bank_sha256':target['bank_sha256'],'result_directory':str(result_root(a,target,bench)),
                    'reused_pilot':target['reused_original_pilot']})
    return records

def report(a,state=None,label=None):
    root=control_root(a)
    if (root/'EVALUATION_COMPLETE.json').is_file():return validate_final_marker(a)
    records=completed(a) if (root/'FREEZE.json').is_file() else []
    counts=Counter(r['key'] for r in records);done=sum(n==2 for n in counts.values())
    value={'version':VERSION,'status':label or ('evaluation_complete' if len(records)==24 else 'in_progress'),
        'B_screened_models_evaluated_of_12':done,'completed_evaluation_files':len(records),'required_evaluation_files':24,
        'completed_questions':500*len(records),'total_required_questions':12000,
        'pilot_evaluation_files_reused':sum(r['reused_pilot'] for r in records),
        'new_evaluation_files_complete_of_22':sum(not r['reused_pilot'] for r in records),
        'unit_state':state or {},'updated_utc':utc(),'reserved_final_tests_performed':False,
        'evaluation_benchmarks':['dev_gsm','dev_math'],'control_directory':str(root)}
    current_json(root/'STATUS.json',value)
    if records:
        buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        tmp=root/('RESULTS.csv.'+uuid.uuid4().hex+'.partial');tmp.write_text(buf.getvalue());os.replace(tmp,root/'RESULTS.csv')
    return value

def finish(a,state):
    root=control_root(a);r=report(a,state,'evaluation_complete')
    if r['completed_evaluation_files']!=24:raise IntegrityError('Not all screened B development results are complete')
    records=completed(a);summary=[]
    for target in plan(a)['targets']:
        rs={x['benchmark']:x for x in records if x['key']==target['key']}
        summary.append({'key':target['key'],'model_size':target['model_size'],'parent_seed':target['parent_seed'],
            'continuation_seed':target['continuation_seed'],'gsm_percent':100*rs['dev_gsm']['accuracy'],
            'math_percent':100*rs['dev_math']['accuracy'],
            'equal_weight_percent':50*(rs['dev_gsm']['accuracy']+rs['dev_math']['accuracy']),
            'bank_sha256':target['bank_sha256'],'manifest_sha256':target['manifest_sha256']})
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    save_bytes(root/'B_SCREENED_SUMMARY.csv',buf.getvalue().encode())
    members={}
    for name in ['PLAN.json','FREEZE.json','STATUS.json','RESULTS.csv','B_SCREENED_SUMMARY.csv']:
        members[name]=(root/name).read_bytes()
    for path in root.glob('results/*/*/complete/*'):
        if path.is_file():members[str(path.relative_to(root))]=path.read_bytes()
    for name in ['protocol.json','expected_evaluation.json','PACKAGE_PINS.json']:members['package/'+name]=(HERE/name).read_bytes()
    content=zip_bytes(members);filename='B_development_results_'+hashlib.sha256(content).hexdigest()[:16]+'.zip'
    save_bytes(root/filename,content)
    save(root/'EVALUATION_COMPLETE.json',{'version':VERSION,'package_sha256':package_check(),
        'evaluation_files':24,'models':12,'questions':12000,'new_GPU_evaluation_files':22,
        'results_zip':filename,'results_zip_sha256':sha(root/filename),'status_sha256':sha(root/'STATUS.json'),
        'results_csv_sha256':sha(root/'RESULTS.csv'),'summary_sha256':sha(root/'B_SCREENED_SUMMARY.csv'),
        'reserved_final_tests_performed':False})
    validate_final_marker(a);ensure_handoff(a)

def validate_final_marker(a):
    root=control_root(a);m=read(root/'EVALUATION_COMPLETE.json')
    if m.get('version')!=VERSION or m.get('package_sha256')!=package_check() or m.get('evaluation_files')!=24 or m.get('models')!=12:
        raise IntegrityError('Invalid evaluation-completion marker')
    for path,key in [(m['results_zip'],'results_zip_sha256'),('STATUS.json','status_sha256'),('RESULTS.csv','results_csv_sha256'),('B_SCREENED_SUMMARY.csv','summary_sha256')]:check_hash(root/path,m[key])
    if len(completed(a))!=24:raise IntegrityError('Completion requires all 24 validated development files')
    return read(root/'STATUS.json')

def ensure_handoff(a):
    root=control_root(a);marker=read(root/'EVALUATION_COMPLETE.json')
    t=training(a);tr=t.control_root(a)
    members={n:(root/n).read_bytes() for n in ['STATUS.json','EVALUATION_COMPLETE.json','RESULTS.csv','B_SCREENED_SUMMARY.csv',marker['results_zip']]}
    for n in ['TRAINING_COMPLETE.json','TRAINING_RUNS.csv','SCREENED_B_CHECKPOINTS.json']:
        members['training/'+n]=(tr/n).read_bytes()
    training_handoff=tr/('B_Screened_4955_Training_Handoff_'+tr.name+'.zip')
    if training_handoff.is_file():members['training/'+training_handoff.name]=training_handoff.read_bytes()
    dest=root/('B_Screened_4955_Training_and_Evaluation_Handoff_'+root.name+'.zip')
    save_bytes(dest,zip_bytes(members));return dest

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def check_hash(path,wanted):
    if sha(path)!=wanted: raise IntegrityError('Changed or incorrect file, preserved: '+str(path))
    return wanted

def save_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()==data:return
        raise IntegrityError('Different existing evidence preserved: '+str(path))
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(tmp,path)

def save(path,value):
    save_bytes(path,(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())

def save_rows(path,records):
    save_bytes(path,''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records).encode())

def current_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

@contextlib.contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise IntegrityError('Another submission/worker is active for this output directory')
        yield

def zip_bytes(members):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    return buffer.getvalue()

def queue_rows():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%r'],capture_output=True,text=True)
    if result.returncode:raise IntegrityError('Cannot establish active jobs; refusing duplicate submission: '+result.stderr)
    return {parts[0]:parts[1:] for line in result.stdout.splitlines() if len(parts:=line.split('|',2))==3}

def release_jobs(ledger,path,active):
    released=ledger.setdefault('released_job_ids',[])
    for job in ledger['jobs']:
        jid=job['job_id']
        if jid in released:continue
        details=active.get(jid)
        if details and details[0]=='PENDING' and details[1]=='JobHeldUser':
            subprocess.run(['scontrol','release',jid],check=True)
        elif details and details[0]=='PENDING' and 'Held' in details[1]:
            raise IntegrityError('Scheduler/admin hold requires review: '+jid+' '+details[1])
        elif not details:
            result=subprocess.run(['scontrol','show','job',jid,'-o'],capture_output=True,text=True)
            if result.returncode:
                raise IntegrityError('Cannot establish whether unreleased job is held: '+jid+'; preserve submission and check its scheduler state before retrying')
            if 'Reason=JobHeldUser' in result.stdout:
                subprocess.run(['scontrol','release',jid],check=True)
            elif any(term in result.stdout for term in ('Reason=JobHeldAdmin','Reason=JobHeld')):
                raise IntegrityError('Scheduler/admin hold requires review: '+jid)
        released.append(jid);current_json(path,ledger)

def terminate_process(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def cancel_future(submission,slot):
    active=queue_rows()
    ids=[j['job_id'] for j in submission['jobs'] if j['slot']>slot and j['job_id'] in active and active[j['job_id']][0]=='PENDING']
    if ids:subprocess.run(['scancel',*ids],check=True)
def runtime_root(a):
    return Path(a.b_root)/'outputs/B_method_B_document_v3/mixed/budget_1x/runtime'

def set_offline(a):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_CACHE=str(Path(a.b_root)/'hf_cache'),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')

def load_parser(a):
    path=runtime_root(a)/'vendor/evaluation/answer_parser.py'
    check_hash(path,expected()['evaluation_sources']['answer_parser.py'])
    spec=importlib.util.spec_from_file_location('_frozen_eval_answer_parser',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def key_for(row):
    return f"{row['method']}_{row['model_size']}_p{row['parent_seed']}_t{row['continuation_seed']}"

def verify_checkpoint(target):
    adapter=Path(target['adapter'])
    check_hash(adapter/'adapter_model.safetensors',target['checkpoint_sha256'])
    if target.get('adapter_config_sha256'):
        check_hash(adapter/'adapter_config.json',target['adapter_config_sha256'])
    if target.get('checkpoint_record_sha256'):
        check_hash(adapter.parent/'run_config.json',target['checkpoint_record_sha256'])
    adapter_config=read(adapter/'adapter_config.json')
    if (adapter_config.get('r')!=16 or adapter_config.get('lora_alpha')!=32
            or adapter_config.get('peft_type')!='LORA' or abs(adapter_config.get('lora_dropout',-1)-.05)>1e-12):
        raise IntegrityError('Saved LoRA configuration differs from the verified parent recipe')
    if target.get('parent_adapter'):
        parent_config=read(Path(target['parent_adapter'])/'adapter_config.json')
        for name in ('target_modules','task_type','bias','rank_pattern','alpha_pattern','use_rslora','use_dora',
                     'fan_in_fan_out','modules_to_save','target_parameters','layers_to_transform','layers_pattern'):
            left,right=adapter_config.get(name),parent_config.get(name)
            if isinstance(left,list) and isinstance(right,list):
                left,right=sorted(left),sorted(right)
            if left!=right:
                raise IntegrityError('Saved adapter targeting/scaling differs from original parent: '+name)
    config=read(adapter.parent/'run_config.json')
    required={'model':target['model'],'revision':target['revision'],'manifest_sha256':target['manifest_sha256']}
    if any(config.get(k)!=v for k,v in required.items()) or config.get('mode')=='debug':
        raise IntegrityError('Wrong model/manifest or a smoke checkpoint: '+str(adapter))
    if config.get('seed')!=target['continuation_seed']:
        raise IntegrityError('Wrong saved training seed: '+str(adapter))
    if target['method']!='parent':
        summary=read(adapter.parent.parent/'training_summary.json')
        if summary.get('complete') is not True or summary.get('unique_examples')!=1000:
            raise IntegrityError('Only completed 1000-example continuations can be evaluated')
    if target.get('approved_files'):
        for path,hash_value in target['approved_files'].items():
            check_hash(path,hash_value)
            if path.endswith('/run_config.json'):
                check_hash(adapter.parent/'run_config.json',hash_value)
    return config

def question_file(a,bench):
    if bench not in BENCHMARKS:
        raise IntegrityError('Only frozen development datasets are authorized')
    path=control_root(a)/'questions'/(bench+'.jsonl')
    check_hash(path,BENCHMARKS[bench])
    data=rows(path)
    wanted='gsm' if bench=='dev_gsm' else 'math'
    ids=[q['problem_id'] for q in data]
    if (len(data)!=500 or len(set(ids))!=500 or any(q.get('benchmark')!=wanted or q.get('split')=='test' for q in data)):
        raise IntegrityError('Unexpected question identities, benchmark, count, or final-test rows')
    return path,data

def expected_config(a,target,bench):
    e=expected()
    value={**e['generation_contract'],'model':target['model'],'revision':target['revision'],
        'eval_file_sha256':BENCHMARKS[bench],'checkpoint_sha256':target['checkpoint_sha256'],
        'train_manifest_sha256':target['manifest_sha256'],
        'train_rows':1000,'train_seed':target['continuation_seed'],
        'evaluate_py_sha256':e['evaluation_sources']['evaluate.py'],
        'answer_parser_sha256':e['evaluation_sources']['answer_parser.py']}
    if target['method'] in ('C0','C1','B_screened'):
        value['train_arm']='B' if target['method']=='B_screened' else target['method']
    if target['method']!='parent':
        value['train_mode']='branch'
    return value

def validate_config(a,target,bench,config):
    for key,value in expected_config(a,target,bench).items():
        if config.get(key)!=value:
            raise IntegrityError(f'Evaluation setting {key} differs for {target["key"]}/{bench}')
    reconstructed=hashlib.sha256(json.dumps({k:config[k] for k in GENERATION_FIELDS},sort_keys=True).encode()).hexdigest()[:12]
    if reconstructed!=config.get('eval_config_hash'):
        raise IntegrityError('Native generation-configuration hash does not reproduce')

def normalized_gold(parser,bench,value):
    value=str(value)
    if bench=='dev_gsm':
        return parser.gsm_gold(value)
    if '\\boxed' in value:
        value=parser.math_gold({'solution':value})
    return parser._plain(parser._strip_thousands(str(value)))

def validate_predictions(a,target,bench,predictions,config,partial=False,allow_checker_errors=False):
    validate_config(a,target,bench,config)
    _,questions=question_file(a,bench)
    wanted={q['problem_id']:q for q in questions}
    ids=[p['problem_id'] for p in predictions]
    if len(ids)!=len(set(ids)) or set(ids)-set(wanted) or (not partial and set(ids)!=set(wanted)):
        raise IntegrityError('Missing, duplicate, or unknown evaluation IDs')
    parser=load_parser(a)
    allowed={'ok','no_answer','unparseable_number','unparseable_math','ambiguous'}
    if allow_checker_errors:
        allowed.add('verify_error')
    for p in predictions:
        q=wanted[p['problem_id']]
        expected_row={'checkpoint_sha256':target['checkpoint_sha256'],'model_revision':target['model']+'@'+target['revision'],
            'train_manifest_sha256':target['manifest_sha256'],'parser_version':config['parser_version'],
            'eval_config_hash':config['eval_config_hash'],'benchmark':q['benchmark'],'run_id':config['run_id']}
        if any(p.get(k)!=v for k,v in expected_row.items()) or type(p.get('correct')) is not bool:
            raise IntegrityError('Mixed prediction checkpoint, seed, parser, question, or generation identity')
        if p.get('parse_status') not in allowed:
            raise IntegrityError('Checker error or unknown parse outcome: '+str(p.get('parse_status')))
        if p['correct'] and p['parse_status']!='ok':
            raise IntegrityError('A valid correct answer requires a successful parse')
        if normalized_gold(parser,bench,p.get('gold_answer',''))!=normalized_gold(parser,bench,q['gold_answer']):
            raise IntegrityError('Gold answer differs from frozen question file')
        if not isinstance(p.get('raw_output'),str):
            raise IntegrityError('Raw generation text missing')
        if type(p.get('generated_tokens')) is not int or not 0<=p['generated_tokens']<=1024:
            raise IntegrityError('Invalid generated-token count')
        if not isinstance(p.get('latency'),(int,float)) or not math.isfinite(p['latency']) or p['latency']<0:
            raise IntegrityError('Invalid prediction latency')
        if not isinstance(p.get('repetition_score'),(int,float)) or not 0<=p['repetition_score']<=1:
            raise IntegrityError('Invalid repetition score')
        if p.get('finish_reason') not in ('eos','length','stop_string','unknown'):
            raise IntegrityError('Invalid stopping reason')
    return {p['problem_id']:p['correct'] for p in predictions}

def result_root(a,target,bench):
    return control_root(a)/'results'/target['key']/bench/'complete'

def metrics(predictions):
    n=len(predictions)
    if n!=500:
        raise IntegrityError('Metrics require the complete frozen 500-question dataset')
    tokens=sorted(p['generated_tokens'] for p in predictions)
    return {'total':n,'correct':sum(p['correct'] for p in predictions),
        'accuracy':sum(p['correct'] for p in predictions)/n,
        'valid_answer_rate':sum(p['parse_status']=='ok' for p in predictions)/n,
        'length_limit_rate':sum(p['finish_reason']=='length' for p in predictions)/n,
        'generation_tokens_mean':statistics.mean(tokens),'generation_tokens_median':statistics.median(tokens),
        'generation_tokens_p95':tokens[min(n-1,int(.95*n))],
        'repetition_rate_full_output_threshold_0p20':sum(p['repetition_score']>=.20 for p in predictions)/n,
        'summed_per_question_latency_seconds':sum(p['latency'] for p in predictions),
        'parse_status_counts':dict(Counter(p['parse_status'] for p in predictions))}

def repair_checker_errors(a,target,bench,predictions):
    parser=load_parser(a)
    _,questions=question_file(a,bench)
    gold={q['problem_id']:q['gold_answer'] for q in questions}
    repaired=[]
    result=[]
    for p in predictions:
        if p['parse_status']!='verify_error':
            result.append(p)
            continue
        for attempt in range(1,4):
            value=gold[p['problem_id']]
            if bench=='dev_math' and '\\boxed' in str(value):
                value=parser.math_gold({'solution':str(value)})
            graded=(parser.score_gsm if bench=='dev_gsm' else parser.score_math)(p['raw_output'],str(value))
            if graded['parse_status']!='verify_error':
                break
        if graded['parse_status']=='verify_error':
            raise SelectionBlocked('Unresolved checker execution error for '+p['problem_id']+'; not counted as wrong')
        fixed={**p,**graded}
        result.append(fixed)
        repaired.append({'problem_id':p['problem_id'],'before':p,'after':fixed,'same_parser_attempts':attempt})
    return result,repaired

def finalize_result(a,target,bench,source,source_config,origin):
    dest=result_root(a,target,bench)
    if dest.exists():
        return validate_result(a,target,bench)
    config=read(source_config)
    original=rows(source)
    validate_predictions(a,target,bench,original,config,allow_checker_errors=True)
    predictions,repaired=repair_checker_errors(a,target,bench,original)
    validate_predictions(a,target,bench,predictions,config)
    staging=dest.parent/('publish_'+uuid.uuid4().hex+'.partial')
    if repaired:
        save_rows(staging/'predictions.jsonl',predictions)
        save_rows(staging/'checker_retries.jsonl',repaired)
    else:
        save_bytes(staging/'predictions.jsonl',Path(source).read_bytes())
    save_bytes(staging/'predictions.config.json',Path(source_config).read_bytes())
    summary=metrics(predictions)
    save(staging/'metrics.json',summary)
    receipt={'version':VERSION,'target_key':target['key'],'benchmark':bench,'complete':True,
        'target_sha256':digest(target),'predictions_sha256':sha(staging/'predictions.jsonl'),
        'config_sha256':sha(staging/'predictions.config.json'),'metrics_sha256':sha(staging/'metrics.json'),
        'source_predictions':str(Path(source).resolve()),'source_predictions_sha256':sha(source),
        'source_config_sha256':sha(source_config),'origin':origin,'checker_retry_rows':len(repaired),
        'checkpoint_sha256':target['checkpoint_sha256'],'eval_file_sha256':BENCHMARKS[bench],
        'scope':'Exposure-limited development evaluation; no reserved final tests',
        'training_performed':False,'reserved_final_tests_performed':False}
    if repaired:
        receipt['checker_retries_sha256']=sha(staging/'checker_retries.jsonl')
    save(staging/'completion.json',receipt)
    try:
        staging.rename(dest)
    except FileExistsError:
        return validate_result(a,target,bench)
    return validate_result(a,target,bench)

def validate_result(a,target,bench):
    dest=result_root(a,target,bench)
    receipt=read(dest/'completion.json')
    required={'version':VERSION,'target_key':target['key'],'benchmark':bench,'complete':True,
        'target_sha256':digest(target),'checkpoint_sha256':target['checkpoint_sha256'],
        'eval_file_sha256':BENCHMARKS[bench],'training_performed':False,'reserved_final_tests_performed':False}
    if any(receipt.get(k)!=v for k,v in required.items()):
        raise IntegrityError('Result completion belongs to a different experiment')
    for name,key in [('predictions.jsonl','predictions_sha256'),('predictions.config.json','config_sha256'),
                     ('metrics.json','metrics_sha256')]:
        check_hash(dest/name,receipt[key])
    predictions=rows(dest/'predictions.jsonl')
    config=read(dest/'predictions.config.json')
    validate_predictions(a,target,bench,predictions,config)
    if read(dest/'metrics.json')!=metrics(predictions):
        raise IntegrityError('Stored metrics differ from saved predictions')
    if receipt['checker_retry_rows']:
        check_hash(dest/'checker_retries.jsonl',receipt['checker_retries_sha256'])
    return predictions,config,receipt

def native_runtime_check():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise IntegrityError('This evaluation needs one allocated A100 GPU; submit through the notebook')
    contract=expected()['generation_contract']
    observed={'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
              **{name:importlib.metadata.version(name) for name in ('torch','transformers','peft')}}
    if any(observed[k]!=contract[k] for k in observed):
        raise IntegrityError('GPU/CUDA/library runtime differs from frozen B/pilot evaluation contract: '+json.dumps(observed))

def run_evaluation(a,target,bench):
    verify_checkpoint(target)
    dest=result_root(a,target,bench)
    if (dest/'completion.json').is_file():
        return validate_result(a,target,bench)
    native_runtime_check()
    evaluator=runtime_root(a)/'vendor/evaluation/evaluate.py'
    check_hash(evaluator,expected()['evaluation_sources']['evaluate.py'])
    source,_=question_file(a,bench)
    raw=None
    for attempt in range(1,MAX_ATTEMPTS+1):
        candidate=dest.parent/f'attempt{attempt:02d}'/'raw.jsonl'
        config_path=candidate.with_suffix('.config.json')
        if not candidate.exists():
            if candidate.parent.exists():
                continue
            raw=candidate
            break
        try:
            prior=rows(candidate)
            config=read(config_path)
        except (FileNotFoundError,json.JSONDecodeError):
            continue  # Preserve malformed interrupted attempts; regenerate into a new one.
        validate_predictions(a,target,bench,prior,config,partial=True,allow_checker_errors=True)
        if len(prior)==500:
            return finalize_result(a,target,bench,candidate,config_path,'new_GPU_evaluation')
        if prior and not candidate.read_bytes().endswith(b'\n'):
            continue  # Never append another JSON object to an unterminated partial line.
        raw=candidate
        break
    if raw is None:
        raise SelectionBlocked('Three interrupted evaluation attempts preserved; automatic attempts stopped')
    raw.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-B',str(evaluator),'--eval_file',str(source),'--adapter',target['adapter'],
        '--expect_train_manifest_sha256',target['manifest_sha256'],'--run_id',VERSION+'_'+target['key']+'_'+bench,
        '--out',str(raw),'--precision','fp32','--batch_size','16','--max_new_tokens','1024','--question_text','p1_clean']
    subprocess.run(command,check=True)
    verify_checkpoint(target)
    return finalize_result(a,target,bench,raw,raw.with_suffix('.config.json'),'new_GPU_evaluation')

def submit(a):
    preflight(a);t=training(a);root=control_root(a)
    with locked(root/'SUBMIT.lock'):
        active=queue_rows()
        for path in sorted((root/'submissions').glob('submission_*.json'),reverse=True):
            ledger=read(path)
            if ledger.get('pending_sbatch_slot') is not None:
                raise IntegrityError('Prior evaluation sbatch outcome uncertain; inspect '+str(path)+' and your queue before retrying')
            if any(j['job_id'] in active for j in ledger['jobs']):
                release_jobs(ledger,path,active)
                print(json.dumps({'evaluation_submission_reused':True,'jobs':ledger['jobs'],
                    'all_jobs_released':len(ledger['released_job_ids'])==len(ledger['jobs'])},indent=2));return
        if (root/'EVALUATION_COMPLETE.json').is_file():
            validate_final_marker(a);ensure_handoff(a);print('Training and development evaluation already complete.');return
        # Validate all evaluation resources before any new training submission.
        index=len(list((root/'submissions').glob('submission_*.json')))+1
        path=root/'submissions'/f'submission_{index:03d}.json'
        logs=root/'slurm_logs';logs.mkdir(parents=True,exist_ok=True)
        base=['sbatch','--parsable','--hold','--account',a.account,'--partition',a.partition,'--qos',a.qos,
            '--nodes','1','--ntasks','1','--cpus-per-task','4','--mem','64G','--gpus','a100:1',
            '--time','04:00:00','--signal','B:TERM@180','--chdir',str(Path(a.work)),
            '--job-name','b4955ev-'+root.name[:7]+'-'+str(index),
            '--output',str(logs/'%j.out'),'--error',str(logs/'%j.err')]
        script=[str(HERE/'gpu_worker.sbatch'),str(HERE),str(Path(a.work)),str(Path(a.b_root))]
        check=subprocess.run([*base,'--test-only',*script,str(path),'0',str(a.worker_seconds)],capture_output=True,text=True)
        if check.returncode:raise IntegrityError('Slurm rejected evaluation resources before training submission: '+check.stderr+check.stdout)
        command=[sys.executable,'-u','-B',str(Path(a.work)/'B_Screened_4955_Controls_v1/screened_b.py'),
            'submit','--work',str(a.work),'--b-root',str(a.b_root),
            '--allocations',str(a.training_allocations),'--worker-seconds',str(a.worker_seconds),
            '--account',a.account,'--partition',a.partition,'--qos',a.qos]
        subprocess.run(command,check=True)
        tr=t.control_root(a);dependencies=[]
        if (tr/'TRAINING_COMPLETE.json').is_file():t.validate_final_marker(a)
        else:
            location=read(Path(a.work)/'B_Screened_4955_Controls_v1/RUN_LOCATION.json')
            if location['control_directory']!=str(tr):raise IntegrityError('Training submission belongs to another experiment')
            active=queue_rows();dependencies=[j['job_id'] for j in location['jobs'] if j['job_id'] in active]
            if not dependencies:raise IntegrityError('Training is incomplete without an active recorded job; no evaluation submitted')
        ledger={'version':VERSION,'package_sha256':package_check(),'control_directory':str(root),
            'created_utc':utc(),'jobs':[],'released_job_ids':[],'submission_complete':False,
            'training_control_directory':str(tr),'training_dependencies':dependencies}
        current_json(path,ledger)
        for slot in range(a.eval_allocations):
            command=list(base)
            dep=dependencies if slot==0 else [ledger['jobs'][-1]['job_id']]
            if dep:command+=['--dependency','afterany:'+':'.join(dep)]
            command+=script+[str(path),str(slot),str(a.worker_seconds)]
            ledger['pending_sbatch_slot']=slot;current_json(path,ledger)
            result=subprocess.run(command,capture_output=True,text=True)
            jid=result.stdout.strip().split(';')[0]
            if result.returncode or not jid.isdigit():
                ledger['submission_error']=result.stdout+result.stderr;current_json(path,ledger)
                raise IntegrityError('Evaluation sbatch outcome uncertain; jobs remain held. Training jobs continue. Inspect '+str(path))
            ledger['jobs'].append({'slot':slot,'job_id':jid,'stdout':str(logs/(jid+'.out')),'stderr':str(logs/(jid+'.err'))})
            ledger.pop('pending_sbatch_slot',None);current_json(path,ledger)
        ledger['submission_complete']=True;current_json(path,ledger)
        current_json(HERE/'RUN_LOCATION.json',{'control_directory':str(root),'submission_file':str(path),'jobs':ledger['jobs']})
        if not (root/'STATUS.json').exists():report(a,label='waiting_for_training' if dependencies else 'ready_for_evaluation')
        release_jobs(ledger,path,queue_rows())
        print(json.dumps({'status':'training_then_development_evaluation_submitted',
            'evaluation_jobs':ledger['jobs'],'training_dependencies':dependencies,
            'all_jobs_released':len(ledger['released_job_ids'])==len(ledger['jobs']),
            'training_control_directory':str(tr),'evaluation_control_directory':str(root),
            'evaluation_files':24,'new_GPU_evaluation_files':22,'reused_pilot_evaluation_files':2,
            'benchmarks':{'dev_gsm':500,'dev_math':500},'reserved_final_tests':False,
            'note':'You may disconnect after this cell returns with job IDs and all_jobs_released true.'},indent=2),flush=True)

def worker(a):
    set_offline(a);root=control_root(a);ledger=read(a.submission)
    if ledger.get('package_sha256')!=package_check() or ledger.get('control_directory')!=str(root):raise IntegrityError('Wrong evaluation submission')
    jid=os.environ.get('SLURM_JOB_ID','')
    if not any(j['job_id']==jid and j['slot']==a.slot for j in ledger['jobs']):raise IntegrityError('Worker is not its recorded Slurm job')
    stop={'requested':False,'process':None}
    def interrupted(signum,frame):
        stop['requested']=True
        if stop['process'] is not None:terminate_process(stop['process'])
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with locked(root/'WORKER.lock'):
        t=training(a)
        if not (t.control_root(a)/'TRAINING_COMPLETE.json').is_file():
            report(a,label='blocked_training_incomplete');cancel_future(ledger,a.slot)
            print('Training is incomplete. No evaluation was performed. Inspect training status, then rerun submission.',flush=True);return 2
        if (root/'EVALUATION_COMPLETE.json').is_file():
            validate_final_marker(a);ensure_handoff(a);cancel_future(ledger,a.slot);return 0
        p=initialize(a);state=read(root/'UNIT_STATE.json') if (root/'UNIT_STATE.json').exists() else {}
        deadline=time.monotonic()+a.worker_seconds
        for target in p['targets']:
            for bench in BENCHMARKS:
                key=target['key']+'__'+bench
                if (result_root(a,target,bench)/'completion.json').is_file():
                    validate_result(a,target,bench);state[key]={'status':'complete'};continue
                if stop['requested'] or deadline-time.monotonic()<1200:break
                if state.get(key,{}).get('status')=='blocked':continue
                log=root/'stage_logs'/(key+'_'+utc().replace(':','-')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                state[key]={'status':'running','job_id':jid,'log':str(log)}
                current_json(root/'UNIT_STATE.json',state);report(a,state)
                with log.open('x') as stream:
                    process=subprocess.Popen(child(a,'evaluate-one',target=target['key'],benchmark=bench),
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    stop['process']=process
                    try:code=process.wait(timeout=max(1,deadline-time.monotonic()-120))
                    except subprocess.TimeoutExpired:terminate_process(process);code=-signal.SIGTERM
                    finally:stop['process']=None
                if code==0:
                    validate_result(a,target,bench);state[key]={'status':'complete','log':str(log),'job_id':jid}
                elif code in (65,66):state[key]={'status':'blocked','log':str(log),'returncode':code}
                else:state[key]={'status':'retry_pending','log':str(log),'returncode':code}
                current_json(root/'UNIT_STATE.json',state);report(a,state)
                if stop['requested'] or code<0:break
            if stop['requested'] or deadline-time.monotonic()<1200:break
        r=report(a,state)
        if r['completed_evaluation_files']==24:
            finish(a,state);cancel_future(ledger,a.slot)
            print('SCREENED B TRAINING AND DEVELOPMENT EVALUATION COMPLETE. No reserved final tests.',flush=True);return 0
        pending=[t['key']+'__'+b for t in p['targets'] for b in BENCHMARKS if state.get(t['key']+'__'+b,{}).get('status') not in ('complete','blocked')]
        label='incomplete_blocked' if not pending else ('incomplete_allocation_chain_exhausted' if a.slot==max(j['slot'] for j in ledger['jobs']) else 'continuing_next_allocation')
        report(a,state,label)
        if not pending:cancel_future(ledger,a.slot)
        print('Evaluation status: '+label,flush=True);return 2 if not pending else 0

def status(a):
    t=training(a);t.status(a);root=control_root(a)
    location=HERE/'RUN_LOCATION.json'
    if location.is_file():
        loc=read(location)
        if loc['control_directory']!=str(root):raise IntegrityError('Evaluation location belongs to another package')
        ids=[j['job_id'] for j in loc['jobs']]
        if ids:
            for cmd in [['squeue','-j',','.join(ids),'-o','%.12i %.12T %.12M %.45R'],['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,Timelimit,ExitCode']]:
                if shutil.which(cmd[0]):
                    result=subprocess.run(cmd,capture_output=True,text=True);print(result.stdout or result.stderr)
    if (root/'EVALUATION_COMPLETE.json').exists():value=validate_final_marker(a)
    elif (root/'STATUS.json').exists():value=read(root/'STATUS.json')
    else:print('Evaluation has not yet been submitted.');return
    print('DEVELOPMENT EVALUATION:',json.dumps({k:v for k,v in value.items() if k!='unit_state'},indent=2))
    for key,state in value.get('unit_state',{}).items():
        if state.get('status') in ('running','blocked','retry_pending'):
            print(key,json.dumps(state))
            if state.get('log'):
                path=Path(state['log'])
                if path.is_file() and path.resolve().is_relative_to(root.resolve()):print('\n'.join(path.read_text(errors='replace').splitlines()[-20:]))
    print('Evaluation output:',root)
    print('Completion marker:',root/'EVALUATION_COMPLETE.json')
    print('Summary CSV:',root/'B_SCREENED_SUMMARY.csv')
    if (root/'EVALUATION_COMPLETE.json').exists():print('Combined handoff:',ensure_handoff(a))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','status','preflight','worker','evaluate-one'])
    p.add_argument('--work',default='/scratch/sshind38/capstone_c0_c1_work')
    p.add_argument('--b-root',default='/scratch/sshind38/capstone_approach_b')
    p.add_argument('--account',default='class_fse57081354fall2026');p.add_argument('--partition',default='public');p.add_argument('--qos',default='class')
    p.add_argument('--training-allocations',type=int,default=2);p.add_argument('--eval-allocations',type=int,default=3)
    p.add_argument('--worker-seconds',type=int,default=12600)
    p.add_argument('--submission');p.add_argument('--slot',type=int,default=0)
    p.add_argument('--target');p.add_argument('--benchmark',choices=list(BENCHMARKS))
    return p

def main(argv=None):
    a=parser().parse_args(argv);set_offline(a)
    if a.action=='submit':submit(a)
    elif a.action=='preflight':preflight(a)
    elif a.action=='status':status(a)
    elif a.action=='worker':return worker(a)
    else:
        targets=[r for r in plan(a)['targets'] if r['key']==a.target]
        if len(targets)!=1 or a.benchmark not in BENCHMARKS:raise IntegrityError('Unknown frozen evaluation target/benchmark')
        run_evaluation(a,targets[0],a.benchmark)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except SelectionBlocked as e:print('BLOCKED: '+str(e),file=sys.stderr);raise SystemExit(66)
    except (IntegrityError,ValueError,KeyError,TypeError,FileNotFoundError) as e:
        print('INTEGRITY CHECK FAILED: '+str(e),file=sys.stderr);raise SystemExit(65)

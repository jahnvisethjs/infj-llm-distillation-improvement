#!/usr/bin/env python3
"""Frozen development evaluation of completed C0/C1 adapters on Sol; no training or final tests."""
import argparse
from collections import Counter, defaultdict
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
VERSION = 'c0-c1-frozen-development-evaluation-v1'
PROFILES = (('0.5B',17),('0.5B',123),('1.5B',17),('1.5B',123))
SEEDS = (42,137,2026)
BENCHMARKS = {'dev_gsm':'3cb2c71007d23e01cb70b6370c1ad82ad5bcea3527c7f3258b4814cf2e3d1cc6',
              'dev_math':'641460fdc55beff591daad0cc7530a3f933eb263c3dfdf8a0534fcfaf8196416'}
GENERATION_FIELDS = ('model','revision','precision','max_new_tokens','batch_size','stop_at_question',
    'question_text','prompt_sha256','add_special_tokens','decoding','eos_token_id','adapter_mode',
    'transformers','peft','torch','gpu','cuda')
COMPARE_FIELDS = GENERATION_FIELDS + ('parser_version','math_verify','eval_file_sha256',
                                     'evaluate_py_sha256','answer_parser_sha256','cut_at_next_question')
ACTIVE_ROOT = None
MAX_ATTEMPTS = 3


class IntegrityError(ValueError):
    pass


class SelectionBlocked(IntegrityError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


# Atomic I/O, process supervision and the previously tested Slurm controller are inserted by the builder.
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]

def check_hash(path, expected):
    if sha(path) != expected:
        raise IntegrityError(f"Hash mismatch; preserved existing file: {path}")
    return expected

def save_bytes(path, content):
    """Create immutable artifacts; atomic hard-link prevents overwriting other attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise IntegrityError(f"Different existing evidence preserved: {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)

def save(path, value):
    save_bytes(path, (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode())

def save_rows(path, records):
    save_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                             for r in records).encode())

def current_json(path, value):
    """Only controller status/index files are replaceable, never experimental artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

@contextlib.contextmanager
def locked(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def package_check():
    for name, expected in read(HERE/'PACKAGE_PINS.json').items():
        check_hash(HERE/name,expected)
    return sha(HERE/'PACKAGE_PINS.json')


def expected():
    return read(HERE/'expected_training.json')


def control_root(a):
    return Path(a.work).resolve()/'outputs/C0_C1_development_evaluation_v1'/digest({
        'package':package_check(),'training':sha(HERE/'expected_training.json'),
        'work':str(Path(a.work).resolve()),'b_root':str(Path(a.b_root).resolve())})[:16]


def activate(a):
    global ACTIVE_ROOT
    ACTIVE_ROOT=control_root(a)
    return ACTIVE_ROOT


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


def source_candidates(a,target,bench):
    b_root=Path(a.b_root)
    work=Path(a.work)
    if target.get('reused_original_pilot') or target['method']=='B_screened' or (target['method']=='parent' and target['profile']=='0.5B_seed17'):
        label='C0' if target['method']=='C0' else 'B' if target['method']=='B_screened' else 'parent'
        yield work/f'outputs/C0_screened_pilot_v1/0.5B_seed17/be46c67690fb33ba/development/pilot01/{label}/{bench}.jsonl'
    if target['method']=='parent':
        base=b_root/'outputs/B_method_B_document_v3'
        yield from sorted((base/'parent_evaluation_cache').glob(target['profile']+'_'+bench+'_*.jsonl'))
        for file in sorted((base/'mixed/budget_1x/evaluation').glob('*/parent_'+bench+'.jsonl')):
            yield file
    if target['method']=='B_legacy':
        yield Path(target['legacy_evaluation_directory'])/('B_'+bench+'.jsonl')


def import_existing(a,target,bench):
    dest=result_root(a,target,bench)
    if (dest/'completion.json').is_file():
        validate_result(a,target,bench)
        return True,[]
    rejected=[]
    for path in source_candidates(a,target,bench):
        if not path.is_file() or not path.with_suffix('.config.json').is_file():
            continue
        try:
            validate_predictions(a,target,bench,rows(path),read(path.with_suffix('.config.json')),allow_checker_errors=True)
        except (ValueError,KeyError,TypeError) as error:
            rejected.append({'path':str(path),'reason':str(error)})
            continue
        finalize_result(a,target,bench,path,path.with_suffix('.config.json'),'reused_compatible_predictions')
        return True,rejected
    return False,rejected


def build_legacy_target(a,profile,seed):
    entry=expected()['parents'][profile]
    condition=Path(a.b_root)/'outputs/B_method_B_document_v3/mixed/budget_1x'
    index=read(condition/'run_index.json') if (condition/'run_index.json').is_file() else {}
    logical=f"B_{entry['model_size']}_p{entry['parent_training_seed']}_t{seed}_1x"
    actual=index.get(logical,logical)
    if Path(actual).name!=actual:
        raise IntegrityError('Legacy B run index has an unexpected path')
    run=condition/'runs'/actual
    if not (run/'B_completion_audit.json').is_file() or read(run/'B_completion_audit.json').get('passed') is not True:
        return None,'Original B completion record unavailable: '+str(run)
    config=read(run/'run_config.json')
    if (config.get('parent_profile')!=profile or config.get('continuation_training_seed')!=seed
            or config.get('manifest_sha256')!=expected()['legacy_B_manifest_sha256']
            or config.get('shared_universe_sha256')!=expected()['legacy_bank_sha256']
            or config.get('parent_adapter_sha256')!=entry['expected_adapter_sha256']):
        return None,'Original B parent, seed or training-bank lineage differs: '+str(run)
    target={'key':f'B_legacy_{profile}_t{seed}','profile':profile,'model_size':entry['model_size'],
        'parent_seed':entry['parent_training_seed'],'continuation_seed':seed,'method':'B_legacy',
        'model':entry['model'],'revision':entry['revision'],'adapter':str(run/'final/adapter'),
        'checkpoint_sha256':sha(run/'final/adapter/adapter_model.safetensors'),
        'manifest_sha256':config['manifest_sha256'],'bank_sha256':expected()['legacy_bank_sha256'],
        'parent_sha256':entry['expected_adapter_sha256'],'required':False,
        'legacy_evaluation_directory':str(condition/'evaluation'/actual),
        'interpretation':'Historical context; different candidate bank; excluded from matched C-versus-B effects'}
    summary=read(run/'training_summary.json')
    target['actual_supervised_tokens']=summary.get('processed_tokens')
    target['recorded_training_loop_seconds']=summary.get('training_seconds')
    verify_checkpoint(target)
    return target,None


def make_plan(a):
    e=expected()
    targets=[]
    for profile,entry in e['parents'].items():
        targets.append({'key':'parent_'+profile,'profile':profile,'model_size':entry['model_size'],
            'parent_seed':entry['parent_training_seed'],'continuation_seed':entry['parent_training_seed'],
            'method':'parent','model':entry['model'],'revision':entry['revision'],
            'adapter':str(Path(a.b_root)/entry['adapter_relative_path']),
            'checkpoint_sha256':entry['expected_adapter_sha256'],'manifest_sha256':entry['parent_manifest_sha256'],
            'bank_sha256':None,'parent_sha256':entry['expected_adapter_sha256'],'required':True})
    targets.append(e['screened_B_pilot'])
    targets.extend(e['C_runs'])
    unavailable=[]
    for profile in e['parents']:
        for seed in SEEDS:
            try:
                target,error=build_legacy_target(a,profile,seed)
            except (OSError,ValueError,KeyError) as problem:
                target,error=None,str(problem)
            if target:
                targets.append(target)
            else:
                unavailable.append({'profile':profile,'seed':seed,'reason':error})
    parent_adapters={t['profile']:t['adapter'] for t in targets if t['method']=='parent'}
    for target in targets:
        target['parent_adapter']=parent_adapters[target['profile']]
        verify_checkpoint(target)
        target['adapter_config_sha256']=sha(Path(target['adapter'])/'adapter_config.json')
        target['checkpoint_record_sha256']=sha(Path(target['adapter']).parent/'run_config.json')
    if len([t for t in targets if t['required']])!=29 or len({t['key'] for t in targets})!=len(targets):
        raise IntegrityError('Incorrect required checkpoint matrix')
    return {'version':VERSION,'targets':targets,'legacy_B_unavailable':unavailable,
        'required_benchmark_files':58,'C0_C1_benchmark_files':48,
        'matched_B_missing':[{'model_size':m,'parent_seed':p,'continuation_seed':s,'status':'training_and_evaluation_required'}
            for m,p in PROFILES for s in SEEDS if (m,p,s)!=('0.5B',17,42)],
        'development_only':True,'training_scheduled':False,'reserved_final_tests_scheduled':False,
        'limitations':e['limitations']}


def frozen_plan(a):
    plan=read(control_root(a)/'PLAN.json')
    freeze=read(control_root(a)/'FREEZE.json')
    if (freeze['package_sha256']!=package_check() or freeze['plan_sha256']!=digest(plan)
            or freeze['expected_training_sha256']!=sha(HERE/'expected_training.json')):
        raise IntegrityError('Frozen evaluation identity changed')
    return plan


def initialize(a):
    root=activate(a)
    root.mkdir(parents=True,exist_ok=True)
    e=expected()
    with locked(root/'PREPARE.lock'):
        for relative,wanted in e['approved_training_files'].items():
            check_hash(Path(a.work)/relative,wanted)
        for filename,wanted in e['evaluation_sources'].items():
            check_hash(runtime_root(a)/'vendor/evaluation'/filename,wanted)
        for bench,wanted in BENCHMARKS.items():
            source=runtime_root(a)/'vendor/evaluation/data'/(bench+'.jsonl')
            check_hash(source,wanted)
            save_bytes(root/'questions'/(bench+'.jsonl'),source.read_bytes())
            question_file(a,bench)
        if (root/'PLAN.json').exists():
            if not (root/'FREEZE.json').exists():
                # Recover a disconnect between the two immutable preparation writes.
                recovered=read(root/'PLAN.json')
                if recovered.get('version')!=VERSION or len([t for t in recovered['targets'] if t['required']])!=29:
                    raise IntegrityError('Incomplete preparation has an invalid checkpoint plan')
                save(root/'FREEZE.json',{'version':VERSION,'plan_sha256':digest(recovered),
                    'package_sha256':package_check(),'expected_training_sha256':sha(HERE/'expected_training.json'),
                    'questions_sha256':BENCHMARKS,'training_started':False,'reserved_final_tests_used':False})
            plan=frozen_plan(a)
        else:
            plan=make_plan(a)
            save(root/'PLAN.json',plan)
            save(root/'FREEZE.json',{'version':VERSION,'plan_sha256':digest(plan),
                'package_sha256':package_check(),'expected_training_sha256':sha(HERE/'expected_training.json'),
                'questions_sha256':BENCHMARKS,'training_started':False,'reserved_final_tests_used':False})
        for target in plan['targets']:
            if target['required']:
                verify_checkpoint(target)
        if (root/'EVALUATION_COMPLETE.json').exists():
            validate_final_marker(a)
            return plan
        rejected=[]
        for target in plan['targets']:
            for bench in BENCHMARKS:
                try:
                    imported,issues=import_existing(a,target,bench)
                except (IntegrityError,OSError,KeyError,TypeError) as problem:
                    if target['required']:
                        raise
                    imported,issues=False,[{'target':target['key'],'benchmark':bench,
                                            'reason':'Optional historical import unavailable: '+str(problem)}]
                rejected.extend(issues)
                if imported and target['required']:
                    complete_unit(a,unit_for(target,bench))
        current_json(root/'IMPORT_STATUS.json',{'rejected_sources':rejected,
            'legacy_B_unavailable':plan['legacy_B_unavailable'],
            'legacy_B_prediction_files_imported':sum((result_root(a,t,b)/'completion.json').exists()
                for t in plan['targets'] if t['method']=='B_legacy' for b in BENCHMARKS)})
    return plan


def unit_for(target,bench):
    return {'id':target['key']+'__'+bench,'profile':target['profile'],'stage':'evaluate','requires':[],
            'target_key':target['key'],'benchmark':bench}


def units():
    if ACTIVE_ROOT is None:
        raise IntegrityError('Evaluation plan has not been activated')
    return [unit_for(t,b) for t in read(ACTIVE_ROOT/'PLAN.json')['targets'] if t['required'] for b in BENCHMARKS]


def unit_path(a,unit):
    return control_root(a)/'unit_receipts'/(unit['id']+'.json')


def target_for(a,unit):
    return next(t for t in frozen_plan(a)['targets'] if t['key']==unit['target_key'])


def complete_unit(a,unit):
    target=target_for(a,unit)
    validate_result(a,target,unit['benchmark'])
    dest=result_root(a,target,unit['benchmark'])
    save(unit_path(a,unit),{'version':VERSION,'unit':unit,'complete':True,
        'files':{str(dest/name):sha(dest/name) for name in ('completion.json','metrics.json','predictions.jsonl','predictions.config.json')},
        'training_performed':False,'reserved_final_tests_performed':False})


def execute_unit(a,unit):
    set_offline(a)
    target=target_for(a,unit)
    run_evaluation(a,target,unit['benchmark'])
    complete_unit(a,unit)


def validate_unit_receipts(a):
    frozen_plan(a)
    done=set()
    for unit in units():
        path=unit_path(a,unit)
        if not path.exists():
            continue
        receipt=read(path)
        if receipt.get('unit')!=unit or receipt.get('complete') is not True or receipt.get('version')!=VERSION:
            raise IntegrityError('Incompatible evaluation work receipt')
        for file,wanted in receipt['files'].items():
            check_hash(file,wanted)
        done.add(unit['id'])
    return done


def write_csv(path,records):
    import csv
    out=io.StringIO(newline='')
    if records:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_bytes(path,out.getvalue().encode())


def paired_difference(later,earlier,resamples=2000,seed=0):
    if not later or set(later)!=set(earlier) or any(type(x) is not bool for x in [*later.values(),*earlier.values()]):
        raise IntegrityError('Paired comparisons need the same complete IDs and boolean correctness')
    ids=sorted(later)
    differences=[int(later[i])-int(earlier[i]) for i in ids]
    rng=random.Random(seed)
    estimates=sorted(100*sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(resamples))
    counts=Counter((earlier[i],later[i]) for i in ids)
    return {'difference_pp':100*sum(differences)/len(ids),
        'ci95_low_pp':estimates[int(.025*(resamples-1))],'ci95_high_pp':estimates[int(.975*(resamples-1))],
        'wrong_to_correct':counts[False,True],'correct_to_wrong':counts[True,False],
        'questions':len(ids),'bootstrap_resamples':resamples,'bootstrap_seed':seed,
        'uncertainty_scope':'Paired question bootstrap for fixed checkpoints; not training-seed uncertainty'}


def controls_match(left,right):
    if left['profile']!=right['profile'] or left['parent_sha256']!=right['parent_sha256']:
        return False
    if 'parent' in (left['method'],right['method']):
        return True
    return left['bank_sha256']==right['bank_sha256'] and left['continuation_seed']==right['continuation_seed']


def summarize(a):
    plan=frozen_plan(a)
    targets=plan['targets']
    results=[]
    loaded={}
    for t in targets:
        for bench in BENCHMARKS:
            if not (result_root(a,t,bench)/'completion.json').exists():
                if t['required']:
                    raise IntegrityError('Cannot summarize incomplete required evaluation')
                continue
            predictions,config,receipt=validate_result(a,t,bench)
            loaded[t['key'],bench]=(predictions,config,receipt)
            m=metrics(predictions)
            results.append({'model_size':t['model_size'],'parent_seed':t['parent_seed'],
                'continuation_seed':t['continuation_seed'],'method':t['method'],'benchmark':bench,
                'correct':m['correct'],'total':500,'accuracy_percent':100*m['accuracy'],
                'valid_answer_percent':100*m['valid_answer_rate'],'length_limit_percent':100*m['length_limit_rate'],
                'generated_tokens_mean':m['generation_tokens_mean'],
                'inference_latency_sum_seconds':m['summed_per_question_latency_seconds'],
                'actual_supervised_training_tokens':t.get('actual_supervised_tokens'),
                'recorded_training_loop_seconds':t.get('recorded_training_loop_seconds'),
                'repetition_percent':100*m['repetition_rate_full_output_threshold_0p20'],
                'checkpoint_sha256':t['checkpoint_sha256'],'bank_sha256':t['bank_sha256'],
                'origin':receipt['origin'],'checker_retry_rows':receipt['checker_retry_rows'],
                'predictions':str(result_root(a,t,bench)/'predictions.jsonl'),
                'interpretation':'historical context; different training bank' if t['method']=='B_legacy' else 'current screened-study development result'})
    comparisons=[]
    by_profile=defaultdict(list)
    for t in targets:
        by_profile[t['profile']].append(t)
    for profile,items in by_profile.items():
        parent=next(t for t in items if t['method']=='parent')
        pairs=[]
        for seed in SEEDS:
            c0=next(t for t in items if t['method']=='C0' and t['continuation_seed']==seed)
            c1=next(t for t in items if t['method']=='C1' and t['continuation_seed']==seed)
            pairs.extend([(c1,c0,'C1_minus_C0'),(c0,parent,'C0_minus_parent'),(c1,parent,'C1_minus_parent')])
            matched=[t for t in items if t['method']=='B_screened' and t['continuation_seed']==seed]
            for b in matched:
                pairs.extend([(c0,b,'C0_minus_B_screened'),(c1,b,'C1_minus_B_screened'),(b,parent,'B_screened_minus_parent')])
        for later,earlier,label in pairs:
            if not controls_match(later,earlier):
                raise IntegrityError('Attempted a matched contrast with incompatible training controls')
            for bench in BENCHMARKS:
                lp,lc,_=loaded[later['key'],bench]
                ep,ec,_=loaded[earlier['key'],bench]
                if any(lc.get(k)!=ec.get(k) for k in COMPARE_FIELDS):
                    raise IntegrityError('Comparison uses incompatible questions/generation/parser/platform settings')
                stats=paired_difference({p['problem_id']:p['correct'] for p in lp},{p['problem_id']:p['correct'] for p in ep})
                comparisons.append({'model_size':later['model_size'],'parent_seed':later['parent_seed'],
                    'continuation_seed':later['continuation_seed'],'comparison':label,'benchmark':bench,
                    'earlier_accuracy_percent':100*sum(p['correct'] for p in ep)/500,
                    'later_accuracy_percent':100*sum(p['correct'] for p in lp)/500,**stats})
    per_seed=[]
    for t in targets:
        cells=[r for r in results if r['method']==t['method'] and r['model_size']==t['model_size']
               and r['parent_seed']==t['parent_seed'] and r['continuation_seed']==t['continuation_seed']]
        if len(cells)!=2:
            continue
        per_seed.append({'model_size':t['model_size'],'parent_seed':t['parent_seed'],'method':t['method'],
            'continuation_seed':t['continuation_seed'],'gsm_accuracy_percent':next(r['accuracy_percent'] for r in cells if r['benchmark']=='dev_gsm'),
            'math_accuracy_percent':next(r['accuracy_percent'] for r in cells if r['benchmark']=='dev_math'),
            'equal_weight_mean_percent':statistics.mean(r['accuracy_percent'] for r in cells),
            'bank_sha256':t['bank_sha256'],'interpretation':cells[0]['interpretation']})
    groups=defaultdict(list)
    for row in per_seed:
        if row['method']!='parent':
            groups[row['model_size'],row['parent_seed'],row['method']].append(row)
    seed_summary=[]
    for (model,parent_seed,method),values in sorted(groups.items()):
        means=[r['equal_weight_mean_percent'] for r in values]
        seed_summary.append({'model_size':model,'parent_seed':parent_seed,'method':method,'seeds_completed':len(values),
            'gsm_mean_percent':statistics.mean(r['gsm_accuracy_percent'] for r in values),
            'math_mean_percent':statistics.mean(r['math_accuracy_percent'] for r in values),
            'equal_weight_mean_percent':statistics.mean(means),
            'training_seed_SD_pp':statistics.stdev(means) if len(means)>1 else None,
            'interpretation':values[0]['interpretation']})
    report={'version':VERSION,'evaluation_complete':True,'required_targets':29,
        'C0_C1_targets':24,'C0_C1_benchmark_files':48,
        'results':results,'paired_comparisons':comparisons,'per_seed_equal_weight_results':per_seed,
        'seed_summary':seed_summary,'matched_B_missing':plan['matched_B_missing'],
        'full_matched_B_C0_C1_comparison_complete':not plan['matched_B_missing'],
        'legacy_B_import_status':read(control_root(a)/'IMPORT_STATUS.json'),
        'scope':'Development evaluation only; inherited exposure disclosed; no final-test or improvement certification',
        'statistics':'GSM and MATH separately; equal-weight aggregate descriptive; fixed-checkpoint item-bootstrap CIs; no formal significance claims',
        'limitations':plan['limitations']}
    root=control_root(a)/'reports'
    save(root/'comparison_report.json',report)
    write_csv(root/'DEVELOPMENT_RESULTS.csv',results)
    write_csv(root/'PAIRED_COMPARISONS.csv',comparisons)
    write_csv(root/'PER_SEED_AGGREGATE.csv',per_seed)
    write_csv(root/'SEED_SUMMARY.csv',seed_summary)
    write_csv(root/'MISSING_MATCHED_B.csv',plan['matched_B_missing'])
    write_csv(root/'HISTORICAL_B_CONTEXT.csv',[r for r in results if r['method']=='B_legacy'])
    text=['# C0/C1 development evaluation','',
        'All 24 C0/C1 models have two complete frozen development evaluations. Reserved final tests were not run.',
        '','| Model | Parent | Method | Seeds | GSM mean (%) | MATH mean (%) | Equal-weight mean (%) | Seed SD (pp) |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in seed_summary:
        sd='n/a' if r['training_seed_SD_pp'] is None else f"{r['training_seed_SD_pp']:.2f}"
        text.append(f"| {r['model_size']} | {r['parent_seed']} | {r['method']} | {r['seeds_completed']} | {r['gsm_mean_percent']:.2f} | {r['math_mean_percent']:.2f} | {r['equal_weight_mean_percent']:.2f} | {sd} |")
    text.extend(['','B_screened is matched only for 0.5B / parent17 / continuation42. Eleven further screened B training/evaluation controls are missing.',
        'B_legacy uses the original 6409-row bank and is historical context. It is excluded from matched C-minus-B effects.',
        '','PAIRED_COMPARISONS.csv contains all 12 C1-minus-C0 comparisons on each benchmark, parent contrasts and the supported screened B pilot contrasts.',
        'Intervals describe question variation for fixed checkpoints; three-seed means and SD are descriptive. No best-seed selection or pooled unsupported significance claim is made.',
        '','These development measurements retain the disclosed inherited parent exposure and conservative-screen limitations.'])
    save_bytes(root/'comparison.md',('\n'.join(text)+'\n').encode())
    return report


def report(a,state,status=None):
    plan=frozen_plan(a)
    done={u['id'] for u in units() if unit_path(a,u).is_file()}
    targets=[t for t in plan['targets'] if t['required']]
    counts={}
    for method in ('C0','C1','parent','B_screened'):
        counts[method]=sum(all(unit_for(t,b)['id'] in done for b in BENCHMARKS) for t in targets if t['method']==method)
    total_questions=sum(500 for u in units() if u['id'] in done)
    value={'version':VERSION,'updated_utc':utc(),'status':status or 'in_progress',
        'all_requested_evaluation_complete':len(done)==58,'C0_models_evaluated_of_12':counts['C0'],
        'C1_models_evaluated_of_12':counts['C1'],'parents_evaluated_of_4':counts['parent'],
        'screened_B_pilot_evaluated_of_1':counts['B_screened'],
        'completed_evaluation_files':len(done),'required_evaluation_files':58,
        'completed_questions_in_required_files':total_questions,'unit_state':state,
        'control_directory':str(control_root(a)),'training_performed':False,'reserved_final_tests_performed':False,
        'missing_matched_B_controls':11,'full_matched_three_method_comparison_complete':False}
    current_json(control_root(a)/'STATUS.json',value)
    return value


def build_bundle(a,final):
    root=control_root(a)
    target=root/'C0_C1_Development_Evaluation_Results.zip'
    temporary=root/('results_'+uuid.uuid4().hex+'.zip.partial')
    wanted=[root/'STATUS.json',root/'PLAN.json',root/'FREEZE.json',root/'IMPORT_STATUS.json',HERE/'expected_training.json',
            HERE/'PACKAGE_PINS.json',HERE/'protocol.json']
    for folder in ('reports','unit_receipts','questions','submissions'):
        wanted.extend(p for p in (root/folder).rglob('*') if p.is_file() and not p.name.endswith('.partial'))
    for target_info in frozen_plan(a)['targets']:
        for bench in BENCHMARKS:
            dest=result_root(a,target_info,bench)
            if (dest/'completion.json').exists():
                wanted.extend(dest/name for name in ('predictions.jsonl','predictions.config.json','metrics.json','completion.json'))
                if (dest/'checker_retries.jsonl').exists():
                    wanted.append(dest/'checker_retries.jsonl')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(wanted)):
            name=str(file.relative_to(root)) if file.is_relative_to(root) else 'package/'+file.name
            archive.write(file,name)
    os.replace(temporary,target)
    if final['all_requested_evaluation_complete'] and final['status']=='evaluation_complete':
        save(root/'EVALUATION_COMPLETE.json',{'version':VERSION,'C0_models':12,'C1_models':12,'required_files':58,
            'status_sha256':sha(root/'STATUS.json'),'report_sha256':sha(root/'reports/comparison_report.json'),
            'results_zip_sha256':sha(target),'reserved_final_tests_performed':False,
            'full_matched_B_C0_C1_comparison_complete':False,'missing_matched_B_controls':11})


def validate_final_marker(a):
    root=control_root(a)
    marker=read(root/'EVALUATION_COMPLETE.json')
    for filename,key in [('STATUS.json','status_sha256'),('reports/comparison_report.json','report_sha256'),
                         ('C0_C1_Development_Evaluation_Results.zip','results_zip_sha256')]:
        check_hash(root/filename,marker[key])
    if marker.get('C0_models')!=12 or marker.get('C1_models')!=12 or marker.get('required_files')!=58:
        raise IntegrityError('Final evaluation marker has the wrong checkpoint matrix')
    return read(root/'STATUS.json')


def child(a,action,**parameters):
    command=[sys.executable,'-B',str(HERE/'evaluate_remaining.py'),action,'--work',str(a.work),'--b-root',str(a.b_root),
             '--model',a.model,'--parent-seed',str(a.parent_seed)]
    for key,value in parameters.items():
        command.extend(['--'+key.replace('_','-'),str(value)])
    return command


def preflight(a):
    if not 1<=a.allocations<=12 or not 900<=a.worker_seconds<=13800:
        raise IntegrityError('Use 1-12 serial allocations and a worker budget leaving at least ten minutes of each four-hour job')
    for command in ('sbatch','scontrol','squeue','scancel'):
        if shutil.which(command) is None:
            raise IntegrityError('Run this notebook on Sol: missing '+command)
    python=Path(a.b_root)/'venv_b_training_py312/bin/python'
    if str(Path(sys.prefix))!=str(python.parent.parent):
        raise IntegrityError('Use the existing B training/evaluation Python: '+str(python))
    set_offline(a)
    e=expected()
    for name in ('torch','transformers','peft'):
        if importlib.metadata.version(name)!=e['generation_contract'][name]:
            raise IntegrityError('Frozen evaluator package version differs: '+name)
    if importlib.metadata.version('math-verify')!=e['generation_contract']['math_verify']:
        raise IntegrityError('Frozen mathematical checker version differs')
    from huggingface_hub import snapshot_download
    for model,revision in sorted({(t['model'],t['revision']) for t in e['C_runs']}):
        snapshot=Path(snapshot_download(model,revision=revision,local_files_only=True))
        if not list(snapshot.glob('*.safetensors')) or not (snapshot/'tokenizer.json').is_file():
            raise IntegrityError('Pinned offline model/tokenizer cache incomplete')
    plan=initialize(a)
    done=validate_unit_receipts(a)
    state=read(control_root(a)/'UNIT_STATE.json') if (control_root(a)/'UNIT_STATE.json').exists() else {}
    for key in done:
        state[key]={'status':'complete'}
    if not (control_root(a)/'EVALUATION_COMPLETE.json').exists():
        current_json(control_root(a)/'UNIT_STATE.json',state)
        report(a,state)
    value={'passed':True,'C0_C1_models':24,'required_prediction_files':58,'reused_complete_files':len(done),
        'GPU_evaluation_files_remaining':58-len(done),'historical_B_imported_files':read(control_root(a)/'IMPORT_STATUS.json')['legacy_B_prediction_files_imported'],
        'missing_screened_B_controls':11,'training_scheduled':False,'reserved_final_tests_scheduled':False,
        'control_directory':str(control_root(a))}
    current_json(control_root(a)/'PREFLIGHT.json',value)
    print(json.dumps(value,indent=2),flush=True)
    return value


def terminate_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)

def cancel_future(a, submission, slot):
    """Only this submission's later queued allocations are cancelled after terminal status."""
    ids = [j["job_id"] for j in submission["jobs"] if j["slot"] > slot]
    if ids:
        result = subprocess.run(["scancel", *ids], text=True, capture_output=True)
        current_json(control_root(a) / "cancellation.json", {"job_ids": ids, "returncode": result.returncode,
                                                               "reason": "No runnable work remains in this exact submission"})

def worker(a):
    root = control_root(a)
    submission = read(Path(a.submission))
    if submission["package_sha256"] != package_check() or submission["control_directory"] != str(root):
        raise IntegrityError("Slurm submission belongs to another package/output directory")
    current_job = os.environ.get("SLURM_JOB_ID")
    if not any(j["slot"] == a.slot and j["job_id"] == current_job for j in submission["jobs"]):
        raise IntegrityError("Worker must be launched by its recorded Slurm job")
    stop = {"requested": False, "process": None}
    def interrupted(signum, frame):
        stop["requested"] = True
        if stop["process"] is not None:
            terminate_process(stop["process"])
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with locked(root / "WORKER.lock"):
        done = validate_unit_receipts(a)
        frozen_plan(a)
        if (root / "EVALUATION_COMPLETE.json").is_file():
            validate_final_marker(a)
            cancel_future(a, submission, a.slot)
            print("All requested evaluation was already completed and verified", flush=True)
            return 0
        state_path = root / "UNIT_STATE.json"
        state = read(state_path) if state_path.exists() else {}
        deadline = time.monotonic() + a.worker_seconds
        budget_min = {"evaluate": 2400}
        terminal_failure = False
        for unit in units():
            key = unit["id"]
            if key in done:
                state[key] = {"status": "complete"}
                continue
            if stop["requested"]:
                break
            prior = state.get(key, {})
            if prior.get("status") in ("blocked", "failed_permanently"):
                terminal_failure = True
                continue
            if any(state.get(d, {}).get("status") in ("blocked", "failed_permanently") for d in unit["requires"]):
                state[key] = {"status": "blocked", "reason": "Required preparation/selection did not pass"}
                terminal_failure = True
                continue
            if any(d not in done for d in unit["requires"]):
                continue
            if deadline - time.monotonic() < budget_min[unit["stage"]]:
                print("Passing unfinished work to the next pre-submitted allocation", flush=True)
                break
            failed = int(prior.get("failures", 0))
            params = {k: unit[k] for k in ("method", "seed", "chunk") if k in unit}
            size, seed_text = unit["profile"].split("_seed")
            profile_args = SimpleNamespace(**{**vars(a), "model": size, "parent_seed": int(seed_text)})
            command = child(profile_args, "unit", unit_id=key)
            log_dir = root / "stage_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = log_dir / (key + "_" + utc().replace(":", "-") + ".log")
            state[key] = {"status": "running", "job_id": current_job, "log": str(log), "failures": failed}
            current_json(state_path, state)
            report(a, state)
            print(f"START {key}; log: {log}", flush=True)
            with log.open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                stop["process"] = process
                try:
                    code = process.wait(timeout=max(1, deadline - time.monotonic() - 120))
                except subprocess.TimeoutExpired:
                    terminate_process(process)
                    code = -signal.SIGTERM
                finally:
                    stop["process"] = None
            if code == 0:
                if not unit_path(a, unit).is_file():
                    raise IntegrityError("Successful stage did not produce its completion receipt")
                done.add(key)
                state[key] = {"status": "complete", "log": str(log), "job_id": current_job}
            elif code in (65, 66):
                state[key] = {"status": "blocked", "returncode": code, "log": str(log),
                              "reason": "Evaluation identity or checker integrity failed; no automatic relaxation"}
                terminal_failure = True
            elif code < 0 or stop["requested"]:
                state[key] = {"status": "interrupted", "log": str(log), "failures": failed}
                current_json(state_path, state)
                break
            else:
                failed += 1
                state[key] = {"status": "retry_pending" if failed < MAX_ATTEMPTS else "failed_permanently",
                              "log": str(log), "failures": failed, "returncode": code}
                if failed >= MAX_ATTEMPTS:
                    terminal_failure = True
            current_json(state_path, state)
            report(a, state)
            print(f"END {key}: {state[key]['status']}", flush=True)
        current_json(state_path, state)
        # Fresh hashes at the allocation boundary prevent a corrupt receipt being counted complete.
        validate_unit_receipts(a)
        frozen_plan(a)
        final = report(a, state)
        remaining = [u for u in units() if u["id"] not in done and state.get(u["id"], {}).get("status")
                     not in ("blocked", "failed_permanently")]
        if final["all_requested_evaluation_complete"]:
            summarize(a)
            final = report(a, state, "evaluation_complete")
            build_bundle(a, final)
            cancel_future(a, submission, a.slot)
        elif not remaining:
            final = report(a, state, "incomplete_blocked")
            build_bundle(a, final)
            cancel_future(a, submission, a.slot)
        elif a.slot == max(j["slot"] for j in submission["jobs"]):
            final = report(a, state, "incomplete_allocation_chain_exhausted")
            build_bundle(a, final)
        else:
            final = report(a, state, "continuing_in_next_allocation")
        print(json.dumps({k: final[k] for k in ("status", "C0_models_evaluated_of_12", "C1_models_evaluated_of_12",
                         "completed_evaluation_files", "control_directory")}, indent=2), flush=True)
        return 0 if final["all_requested_evaluation_complete"] or remaining else 2

def submit(a):
    preflight(a)
    root = control_root(a)
    with locked(root / "SUBMIT.lock"):
        for path in sorted((root / "submissions").glob("*.json"), reverse=True):
            previous = read(path)
            ids = [j["job_id"] for j in previous["jobs"]]
            if ids:
                check = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T %r"], text=True, capture_output=True)
                if check.returncode:
                    proof=subprocess.run(['sacct','-n','-X','-j',','.join(ids),
                        '--format=JobIDRaw,State%30','--parsable2'],text=True,capture_output=True)
                    states={}
                    for line in proof.stdout.splitlines():
                        fields=line.strip().split('|')
                        if len(fields)>=2 and fields[0] in ids:
                            states[fields[0]]=fields[1].strip().split()[0].rstrip('+') if fields[1].strip() else ''
                    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY',
                              'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
                    if proof.returncode or set(states)!=set(ids) or not all(s in terminal for s in states.values()):
                        raise IntegrityError('Unable to establish prior Slurm status in queue/accounting; no duplicate submission')
                    check=SimpleNamespace(returncode=0,stdout='',stderr='')
                if check.stdout.strip():
                    released = previous.setdefault("released_job_ids", [])
                    queue = {line.split()[0]: line.split(maxsplit=2)[1:]
                             for line in check.stdout.splitlines() if len(line.split()) >= 2}
                    for job in previous["jobs"]:
                        if job["job_id"] in released:
                            continue
                        details = queue.get(job["job_id"], [])
                        if len(details) == 2 and details[1] == "JobHeldUser":
                            subprocess.run(["scontrol", "release", job["job_id"]], check=True)
                        # Running, completed and previously released dependencies need no release.
                        released.append(job["job_id"])
                        current_json(path, previous)
                    previous["submission_complete"] = True
                    current_json(path, previous)
                    current_json(HERE / "RUN_LOCATION.json", {"control_directory": str(root),
                        "submission_file": str(path), "jobs": previous["jobs"]})
                    print(json.dumps({"reused_active_submission": True, "submission_file": str(path),
                                      "jobs": previous["jobs"], "queue": check.stdout}, indent=2))
                    return
        validate_unit_receipts(a)
        if (root / "EVALUATION_COMPLETE.json").is_file():
            validate_final_marker(a)
            print("All requested evaluation is already complete: " + str(root / "EVALUATION_COMPLETE.json"))
            return
        index = len(list((root / "submissions").glob("*.json"))) + 1
        path = root / "submissions" / f"submission_{index:03d}.json"
        log_root = root / "slurm_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        base = ["sbatch", "--parsable", "--hold", "--account", a.account, "--partition", a.partition,
            "--qos", a.qos, "--nodes", "1", "--ntasks", "1", "--cpus-per-task", "4", "--mem", "32G",
            "--gpus", "a100:1", "--time", "04:00:00", "--signal", "B:TERM@180", "--chdir", str(Path(a.work)),
            "--job-name", "c0c1-eval-" + root.name[:8], "--output", str(log_root / "%j.out"),
            "--error", str(log_root / "%j.err")]
        script_args = [str(HERE / "gpu_worker.sbatch"), str(HERE), str(Path(a.work)), str(Path(a.b_root))]
        test = subprocess.run([*base, "--test-only", *script_args, str(path), "0", str(a.worker_seconds)],
                              text=True, capture_output=True)
        if test.returncode:
            raise IntegrityError("Sol rejected the requested resources before submission:\n" + test.stderr + test.stdout)
        ledger = {"version": VERSION, "package_sha256": package_check(), "control_directory": str(root),
            "created_utc": utc(), "allocations_requested": a.allocations, "submission_complete": False, "jobs": []}
        for slot in range(a.allocations):
            command = list(base)
            if ledger["jobs"]:
                command += ["--dependency", "afterany:" + ledger["jobs"][-1]["job_id"]]
            command += script_args + [str(path), str(slot), str(a.worker_seconds)]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                ledger["submission_error"] = result.stderr + result.stdout
                current_json(path, ledger)
                if not ledger["jobs"]:
                    raise IntegrityError("Slurm submission failed: " + ledger["submission_error"])
                print("Slurm accepted a shorter allocation chain; queued work remains resumable:\n" + ledger["submission_error"])
                break
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise IntegrityError("Unrecognized Slurm job ID: " + result.stdout)
            ledger["jobs"].append({"slot": slot, "job_id": job_id,
                "stdout": str(log_root / f"{job_id}.out"), "stderr": str(log_root / f"{job_id}.err")})
            current_json(path, ledger)
        ledger["submission_complete"] = True
        current_json(path, ledger)
        current_json(HERE / "RUN_LOCATION.json", {"control_directory": str(root), "submission_file": str(path),
                                                    "jobs": ledger["jobs"]})
        ledger["released_job_ids"] = []
        for job in ledger["jobs"]:
            subprocess.run(["scontrol", "release", job["job_id"]], check=True)
            ledger["released_job_ids"].append(job["job_id"])
            current_json(path, ledger)
        print(json.dumps({"status": "submitted_evaluation_not_yet_verified", "C0_C1_models": 24, "required_evaluation_files": 58,
            "development_evaluation": True, "reserved_final_tests": False, "jobs": ledger["jobs"], "control_directory": str(root),
            "completion_report": str(root / "STATUS.json"), "completion_marker": str(root / "EVALUATION_COMPLETE.json"),
            "note": "Notebook/laptop can disconnect now. Slurm queues the allocations serially; completion is established by the report."}, indent=2))


def status(a):
    root=activate(a)
    location=HERE/'RUN_LOCATION.json'
    if location.exists():
        info=read(location)
        ids=[j['job_id'] for j in info['jobs']]
        for cmd in (['squeue','-j',','.join(ids),'-o','%.12i %.10T %.12M %.45R'],
                    ['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,Timelimit,ExitCode']):
            r=subprocess.run(cmd,text=True,capture_output=True)
            print(r.stdout or r.stderr)
    if not (root/'STATUS.json').exists():
        print('Evaluation not started yet. Run the launch cell.');return
    s=read(root/'STATUS.json')
    print(json.dumps({k:s[k] for k in ('status','C0_models_evaluated_of_12','C1_models_evaluated_of_12',
        'completed_evaluation_files','required_evaluation_files','completed_questions_in_required_files',
        'training_performed','reserved_final_tests_performed','missing_matched_B_controls')},indent=2))
    for name,value in s['unit_state'].items():
        if value['status']=='running':
            print('\nCURRENT:',name,'Slurm job',value.get('job_id'))
            log=Path(value['log'])
            if log.exists():
                print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
                print('Log modified UTC:',datetime.fromtimestamp(log.stat().st_mtime,timezone.utc).isoformat())
        elif value['status'] not in ('complete',):
            print(name,json.dumps(value))
    print('Output:',root)
    print('Completion marker:',root/'EVALUATION_COMPLETE.json')
    print('Results ZIP:',root/'C0_C1_Development_Evaluation_Results.zip')


def collect(a):
    root=activate(a)
    validate_unit_receipts(a)
    validate_final_marker(a)
    target=Path.home()/('C0_C1_Evaluation_Handoff_'+root.name+'.zip')
    files=('C0_C1_Development_Evaluation_Results.zip','EVALUATION_COMPLETE.json','STATUS.json')
    if not target.exists():
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(root/name,name)
    else:
        with zipfile.ZipFile(target) as archive:
            if any(hashlib.sha256(archive.read(name)).hexdigest()!=sha(root/name) for name in files):
                raise IntegrityError('Different existing handoff ZIP preserved: '+str(target))
    print('Download this ZIP from your Sol HOME folder and upload it to this conversation:')
    print(target)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','submit','worker','unit','status','collect'))
    p.add_argument('--work',default='/scratch/sshind38/capstone_c0_c1_work')
    p.add_argument('--b-root',default='/scratch/sshind38/capstone_approach_b')
    p.add_argument('--model',choices=('0.5B','1.5B'),default='0.5B')
    p.add_argument('--parent-seed',choices=(17,123),type=int,default=17)
    p.add_argument('--unit-id')
    p.add_argument('--submission')
    p.add_argument('--slot',type=int,default=0)
    p.add_argument('--worker-seconds',type=int,default=13800)
    p.add_argument('--allocations',type=int,default=6)
    p.add_argument('--account',default='class_fse57081354fall2026')
    p.add_argument('--partition',default='public')
    p.add_argument('--qos',default='class')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    set_offline(a)
    activate(a)
    if a.action=='prepare':
        preflight(a)
    elif a.action=='submit':
        submit(a)
    elif a.action=='worker':
        return worker(a)
    elif a.action=='unit':
        unit=next(u for u in units() if u['id']==a.unit_id)
        if unit['profile']!=f'{a.model}_seed{a.parent_seed}':
            raise IntegrityError('Wrong model/parent work unit')
        execute_unit(a,unit)
    elif a.action=='status':
        status(a)
    elif a.action=='collect':
        collect(a)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except SelectionBlocked as error:
        print('BLOCKED: '+str(error),file=sys.stderr,flush=True);raise SystemExit(66)
    except (IntegrityError,ValueError) as error:
        print('INTEGRITY CHECK FAILED: '+str(error),file=sys.stderr,flush=True);raise SystemExit(65)
    except subprocess.CalledProcessError as error:
        print('Child evaluation failed: '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(error.returncode if error.returncode in (65,66) else 1)

#!/usr/bin/env python3
"""B commands: prepare, status, import-parents, verify, prefetch, train, evaluate, compare.

Import and prepare use the standard library. No GPU jobs launch implicitly.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from common import HERE, sha, read, write_new, config, registry, data_paths, work_root, budget_name, profile, load_rows, sources_valid, load_person2
import allocator


def prepare(root, cfg):
    sources_valid()
    paths = data_paths(root)
    for name, key in [('parent', 'parent_manifest_sha256'), ('universe', 'common_universe_sha256'), ('contract', 'formatting_contract_sha256')]:
        if sha(paths[name]) != cfg[key]:
            raise ValueError(f'Input hash mismatch: {name}')
    parent, universe = load_rows(paths['parent']), load_rows(paths['universe'])
    contract = read(paths['contract'])
    allocator.validate_rows(parent, contract['max_total_tokens'])
    allocator.validate_rows(universe, contract['max_total_tokens'])
    if len(parent) != 1000 or len(universe) != 5000:
        raise ValueError('Unexpected parent/common universe size')
    normalize = lambda text: ' '.join(unicodedata.normalize('NFKC', text).split())
    protected = {normalize(r['problem']) for r in csv.DictReader(paths['protected'].open(newline='', encoding='utf-8'))}
    if {normalize(r['problem']) for r in universe} & protected:
        raise ValueError('Exact protected overlap in the shared universe')
    for key in ['row_id', 'problem_group_id']:
        if {r[key] for r in parent} & {r[key] for r in universe}:
            raise ValueError(f'Parent/universe overlap: {key}')
    weights = Counter(allocator.cell(r) for r in parent)
    base_target = sum(r['supervised_tokens'] for r in parent)
    priority = allocator.random_priorities(universe, cfg['selection_seed'])
    destination = work_root(root)
    write_new(destination/'common_candidate_universe.jsonl', paths['universe'].read_bytes())
    results = []
    for mult in cfg['budget_multipliers']:
        label = budget_name(mult)
        selected, report = allocator.allocate(
            universe, priority, weights, base_target*mult,
            cfg['relative_token_tolerance'], cfg['quota_seed'], cfg['selection_seed'],
            cfg['row_count_adjustment_fraction'], cfg['swap_window'], cfg['max_priority_swaps'])
        ordered = sorted(selected, key=lambda r: (allocator.stable_key('training-order-v1', cfg['training_order_seed'], r['row_id']), r['row_id']))
        content = allocator.jsonl_bytes(ordered)
        selected_ids = {r['row_id'] for r in selected}
        priorities = []
        ranks = {}
        for h in sorted(weights):
            group = sorted([r for r in universe if allocator.cell(r) == h], key=lambda r: (priority[r['row_id']], r['row_id']))
            for i, row in enumerate(group, 1):
                ranks[row['row_id']] = i
        for i, row in enumerate(ordered):
            priorities.append({'training_order': i, 'row_id': row['row_id'], 'problem_group_id': row['problem_group_id'],
                               'dataset_id': row['dataset_id'], 'length_bin': row['length_bin'],
                               'supervised_tokens': row['supervised_tokens'], 'prompt_tokens': row['prompt_tokens'],
                               'total_tokens': row['total_tokens'], 'random_priority_hex': f"{priority[row['row_id']]:064x}",
                               'within_stratum_priority_rank': ranks[row['row_id']]})
        counts = Counter(allocator.cell(r) for r in selected)
        quotas = allocator.apportion(weights, len(selected), Counter(allocator.cell(r) for r in universe), cfg['quota_seed'])
        coverage = [{'dataset_id': h[0], 'length_bin': h[1], 'parent_reference_rows': weights[h],
                     'required_rows': quotas[h], 'actual_rows': counts[h], 'deviation_rows': counts[h]-quotas[h],
                     'supervised_tokens': sum(r['supervised_tokens'] for r in selected if allocator.cell(r)==h)} for h in sorted(weights)]
        if any(r['deviation_rows'] for r in coverage):
            raise AssertionError('Quota deviation')
        record = {'budget_multiplier': mult, 'target_supervised_tokens': base_target*mult,
                  'selected_rows': len(ordered), 'selected_supervised_tokens': report['actual_tokens'],
                  'difference_tokens': report['actual_tokens']-base_target*mult,
                  'difference_percent': report['relative_token_difference']*100,
                  'selected_manifest_sha256': hashlib.sha256(content).hexdigest(),
                  'common_universe_sha256': cfg['common_universe_sha256'],
                  'parent_manifest_sha256': cfg['parent_manifest_sha256'],
                  'parent_row_id_overlap': 0, 'parent_recorded_group_overlap': 0,
                  'source_rows': dict(Counter(r['dataset_id'] for r in selected)),
                  'source_supervised_tokens': {s: sum(r['supervised_tokens'] for r in selected if r['dataset_id']==s) for s in ['orca','numina']},
                  'quota_deviations': 0, 'tolerance_relaxed': False,
                  'accounting': 'Stored release tokens; real model-specific encoder verification is a separate stage.',
                  'training_started': False, 'actual_training_tokens_consumed': None,
                  'independent_continuation_from_parent': True,
                  'allocator_version': allocator.VERSION, 'allocator_code_sha256': sha(HERE/'allocator.py'),
                  'selection_seed': cfg['selection_seed'], 'training_order_seed': cfg['training_order_seed']}
        folder = destination/label
        write_new(folder/'train_B.jsonl', content)
        write_new(folder/'selection_manifest.csv', allocator.csv_bytes(priorities))
        write_new(folder/'coverage_report.csv', allocator.csv_bytes(coverage))
        write_new(folder/'selected_ids.txt', ('\n'.join(sorted(selected_ids))+'\n').encode())
        write_new(folder/'budget_report.json', record)
        write_new(folder/'allocation_log.json', report)
        results.append(record)
        print(f"{label}: {len(selected)} examples, {report['actual_tokens']} supervised tokens, {record['difference_percent']:+.6f}%", flush=True)
    capacity = {'pilot_universe_rows': len(universe), 'pilot_universe_stored_tokens': sum(r['supervised_tokens'] for r in universe),
                'screened_bank_rows_previously_verified': 9909, 'screened_bank_stored_tokens_previously_verified': 3169918,
                'current_scope': [1,2,4], '8x': 'Requires a newly shared larger universe; do not expand only B.',
                '16x_and_40x': 'Impossible with the current screened bank: required unique-token budgets exceed all its available stored supervised tokens.',
                'new_pool_requirement': 'More screened fresh candidates, common to all fresh selectors, before extending the universe.'}
    write_new(destination/'selection_summary.json', {'budgets':results,'capacity':capacity,'model_training_run':False})
    return results


def parent_check(root, key, entry):
    adapter = Path(root)/entry['adapter_relative_path']
    weights = adapter/'adapter_model.safetensors'
    acfg_path = adapter/'adapter_config.json'
    cfg_path = adapter.parent/'run_config.json'
    result = {'profile':key,'adapter':str(adapter),'ready':False,'issues':[]}
    for path in [weights, acfg_path, cfg_path]:
        if not path.is_file():result['issues'].append('MISSING: '+str(path))
    if result['issues']:return result
    if sha(weights) != entry['expected_adapter_sha256']:
        result['issues'].append('Adapter weight SHA-256 differs from the recorded parent')
    acfg, run_cfg = read(acfg_path), read(cfg_path)
    for name, expected in [('model',entry['model']),('revision',entry['revision']),('manifest_sha256',entry['parent_manifest_sha256']),('seed',entry['parent_training_seed']),('rows',1000)]:
        if run_cfg.get(name)!=expected:result['issues'].append('Parent record mismatch: '+name)
    if acfg.get('r') != 16 or acfg.get('lora_alpha') != 32 or abs(acfg.get('lora_dropout',-1)-0.05)>1e-12 or acfg.get('peft_type')!='LORA':
        result['issues'].append('Adapter LoRA configuration differs from the recorded recipe')
    try:
        header = safetensors_header(weights.read_bytes())
        tensor_names = [name for name in header if name!='__metadata__']
        if not tensor_names or not all('lora_' in name for name in tensor_names):
            result['issues'].append('Unexpected/non-LoRA tensor names')
    except (ValueError,json.JSONDecodeError) as error:result['issues'].append(str(error))
    result['ready'] = not result['issues']
    result['adapter_sha256'] = sha(weights)
    return result


def safetensors_header(content):
    if len(content)<10:raise ValueError('Truncated safetensors file')
    length=struct.unpack('<Q',content[:8])[0]
    if not 2<=length<=16*1024*1024 or 8+length>len(content):raise ValueError('Invalid safetensors header length')
    header=json.loads(content[8:8+length])
    for name, value in header.items():
        if name=='__metadata__':continue
        a,b=value['data_offsets']
        if not 0<=a<=b<=len(content)-8-length:raise ValueError('Invalid tensor data offsets')
    return header


def import_parents(root, archive_path):
    """Match incoming weights by known hashes; never load pickle/optimizer state."""
    known={entry['expected_adapter_sha256']:(key,entry) for key,entry in registry().items()}
    imported=[]
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe archive member: '+name)
        for name in archive.namelist():
            if PurePosixPath(name).name!='adapter_model.safetensors':continue
            content=archive.read(name);weight_hash=hashlib.sha256(content).hexdigest()
            if weight_hash not in known:raise ValueError('Unrecognized parent weight SHA-256: '+name)
            key,entry=known[weight_hash];safetensors_header(content)
            config_name=str(PurePosixPath(name).parent/'adapter_config.json')
            acfg=archive.read(config_name)
            destination=Path(root)/entry['adapter_relative_path']
            write_new(destination/'adapter_model.safetensors',content)
            write_new(destination/'adapter_config.json',acfg)
            # Use the matching original records already supplied with this project.
            records=HERE/entry['records_relative_path']
            write_new(destination.parent/'run_config.json',(records/'run_config.json').read_bytes())
            write_new(destination.parent.parent/'training_summary.json',(records/'training_summary.json').read_bytes())
            result=parent_check(root,key,entry)
            if not result['ready']:raise ValueError('Parent import failed: '+json.dumps(result))
            imported.append(result)
    if not imported:raise ValueError('Archive contains no adapter_model.safetensors files')
    write_new(work_root(root)/('parent_import_'+sha(archive_path)[:12]+'.json'),{'imported':imported,'archive_sha256':sha(archive_path)})
    print(json.dumps(imported,indent=2))
    return imported


def status(root,cfg):
    try:
        paths=data_paths(root)
        data_ready=sha(paths['parent'])==cfg['parent_manifest_sha256'] and sha(paths['universe'])==cfg['common_universe_sha256']
        data_issue=None
    except (FileNotFoundError,ValueError) as error:
        data_ready=False;data_issue=str(error)
    parents=[parent_check(root,key,entry) for key,entry in registry().items()]
    checks=[]
    for size in cfg['models']:
        p=work_root(root)/'label_checks'/size/'verification.json'
        verified=read(p) if p.is_file() else {}
        okay=verified.get('passed') is True and verified.get('real_tokenizer_execution') is True and verified.get('universe_sha256')==cfg['common_universe_sha256'] and verified.get('verification_code_sha256')==sha(HERE/'workbench.py')
        checks.append({'model_size':size,'passed':okay,'report':str(p)})
    environment_path=Path(root)/'venv_b_training_py312/bin/python'
    environment_report_path=Path(root)/'environment_reports/environment.json'
    environment_report=read(environment_report_path) if environment_report_path.is_file() else {}
    inputs_ready=data_ready and all(p['ready'] for p in parents) and all(c['passed'] for c in checks) and cfg['protocol_status']=='frozen'
    result={'data_ready':data_ready,'data_issue':data_issue,'parent_adapters':parents,'model_specific_label_checks':checks,
            'protocol_status':cfg['protocol_status'],'gpu_environment_path':str(environment_path),
            'gpu_environment_installed':environment_path.is_file() and environment_report.get('runtime_imports_passed') is True,
            'continuation_inputs_ready':inputs_ready,
            'continuation_ready':inputs_ready and environment_path.is_file() and environment_report.get('runtime_imports_passed') is True,
            'gpu_smoke_still_required_per_parent':True,
            'independent_work_available':['prepare','bootstrap environment','prefetch public base assets','verify tokenizer/labels','offline tests','documentation'],
            'blocked_work':['Real parent/model smoke check and continuation until a matching adapter is present','Controlled main comparisons until the shared protocol is frozen'],
            'no_model_work_executed_by_status':True}
    print(json.dumps(result,indent=2))
    return result


def model_environment(root,offline=False):
    os.environ['HF_HUB_CACHE']=str(Path(root)/'hf_cache')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if offline:os.environ['HF_HUB_OFFLINE']='1'


def prefetch(root,cfg,offline=False):
    model_environment(root,offline)
    from huggingface_hub import snapshot_download
    done=set()
    for entry in registry().values():
        if entry['model'] in done:continue
        snapshot_download(entry['model'],revision=entry['revision'],cache_dir=str(Path(root)/'hf_cache'),
                          allow_patterns=['*.json','*.safetensors','*.txt','*.model'],max_workers=2,local_files_only=offline,token=False)
        done.add(entry['model']);print('Cached public base assets:',entry['model'],flush=True)
    print('No model weights were loaded into a model and no training ran.')


def verify(root,cfg,offline=False):
    """Use the exact recovered Person 1/2 encoder on the shared universe/parent."""
    model_environment(root,offline)
    os.environ['USE_TORCH']='0';os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from functools import lru_cache
    p2=load_person2();paths=data_paths(root);contract=read(paths['contract'])
    if sha(paths['parent'])!=cfg['parent_manifest_sha256'] or sha(paths['universe'])!=cfg['common_universe_sha256']:
        raise ValueError('Verification inputs changed')
    @lru_cache(maxsize=2)
    def tokenizer(model,revision):
        tok=AutoTokenizer.from_pretrained(model,revision=revision,cache_dir=str(Path(root)/'hf_cache'),use_fast=True,trust_remote_code=False,local_files_only=offline,token=False)
        p2.contract.verify_tokenizer(tok)
        return tok
    p2.model_tokenizer=tokenizer
    for name,expected in contract['tokenizer_asset_sha256'].items():
        path=hf_hub_download(contract['tokenizer_id'],filename=name,revision=contract['tokenizer_revision'],cache_dir=str(Path(root)/'hf_cache'),local_files_only=offline,token=False)
        if sha(path)!=expected:raise ValueError('Reference tokenizer asset mismatch: '+name)
    parent=load_rows(paths['parent']);universe=load_rows(paths['universe']);rows=parent+universe
    reports=[]
    for size in cfg['models']:
        p2.select_model(size);tok=p2.get_tokenizer();counts={};records=[]
        for i,row in enumerate(rows,1):
            try:
                e=p2.encode(row,tok,contract['prompt_template'],contract['max_total_tokens'])
                counts[row['row_id']]=e['counts']['supervised_tokens']
                records.append({'row_id':row['row_id'],'stored_supervised_tokens':row['supervised_tokens'],'actual_supervised_tokens':e['counts']['supervised_tokens'],'passed':True,'error':''})
            except (ValueError,KeyError,TypeError) as error:
                records.append({'row_id':row['row_id'],'stored_supervised_tokens':row['supervised_tokens'],'actual_supervised_tokens':None,'passed':False,'error':str(error)})
            if i%250==0:print(f'{size}: {i}/{len(rows)} rows checked',flush=True)
        budget_reports=[]
        parent_tokens=sum(counts[r['row_id']] for r in parent) if all(r['row_id'] in counts for r in parent) else None
        for mult in cfg['budget_multipliers']:
            path=work_root(root)/budget_name(mult)/'train_B.jsonl';selected=load_rows(path)
            actual=sum(counts[r['row_id']] for r in selected) if all(r['row_id'] in counts for r in selected) else None
            target=parent_tokens*mult if parent_tokens is not None else None
            okay=actual is not None and target is not None and abs(actual-target)*200<=target
            budget_reports.append({'multiplier':mult,'manifest_sha256':sha(path),'rows':len(selected),'actual_supervised_tokens':actual,'target_supervised_tokens':target,'within_0_5_percent':okay})
        report={'model_size':size,'passed':all(r['passed'] for r in records) and all(b['within_0_5_percent'] for b in budget_reports),
                'real_tokenizer_execution':True,'rows_checked':len(rows),'mismatches':sum(not r['passed'] for r in records),
                'parent_manifest_sha256':cfg['parent_manifest_sha256'],'universe_sha256':cfg['common_universe_sha256'],
                'actual_parent_supervised_tokens':parent_tokens,'budgets':budget_reports,
                'person1_executable_encoder_hashes_match_parent_records':True,
                'person1_package_init_note':read(HERE/'source_provenance.json')['person1_init_difference'],
                'model_specific_token_ids_labels_attention_compared_to_0_5B':size=='1.5B',
                'verification_code_sha256':sha(HERE/'workbench.py'),'parent_weights_loaded':False,'training_started':False}
        dest=work_root(root)/'label_checks'/size
        write_new(dest/'rows.csv',allocator.csv_bytes(records));write_new(dest/'verification.json',report)
        reports.append(report)
        if not report['passed']:raise ValueError('Label verification failed; inspect '+str(dest/'rows.csv'))
    print(json.dumps(reports,indent=2))
    return reports


def evaluate(root,cfg,size,seed,multiplier,trained_run=None,final_test=False,execute=False,quiet=False):
    """Prepare commands first; execution is explicit and requires real weights."""
    key,entry=profile(size,seed)
    if final_test and cfg['protocol_status']!='frozen':raise ValueError('Final evaluation requires a frozen protocol')
    if execute or final_test:
        from continuation import validate_protocol
        validate_protocol(root,cfg)
    label=budget_name(multiplier)
    parent=Path(root)/entry['adapter_relative_path']
    run=Path(trained_run) if trained_run else work_root(root)/'runs'/f'B_{size}_p{seed}_t42_{label}'
    if final_test and run.name not in cfg['evaluation']['predeclared_final_runs']:
        raise ValueError('Run is absent from predeclared_final_runs')
    eval_dir=work_root(root)/'evaluation'/run.name
    datasets=['test_gsm','test_math500'] if final_test else ['dev_gsm','dev_math']
    commands=[]
    for model_label,adapter,expected in [('parent',parent,cfg['parent_manifest_sha256']),('B',run/'final/adapter',None)]:
        for dataset in datasets:
            if model_label=='B' and (run/'run_config.json').is_file():expected=read(run/'run_config.json')['manifest_sha256']
            if model_label=='B' and expected is None:
                expected=read(work_root(root)/label/'budget_report.json')['selected_manifest_sha256']
            command=[sys.executable,str(HERE/'vendor/evaluation/evaluate.py'),'--eval_file',str(HERE/'vendor/evaluation/data'/f'{dataset}.jsonl'),
                     '--adapter',str(adapter),'--expect_train_manifest_sha256',expected,
                     '--run_id',run.name+'_'+model_label+'_'+dataset,'--out',str(eval_dir/f'{model_label}_{dataset}.jsonl'),
                     '--precision',cfg['evaluation']['precision'],'--batch_size',str(cfg['evaluation']['batch_size']),
                     '--max_new_tokens',str(cfg['evaluation']['max_new_tokens']),'--question_text',cfg['evaluation']['question_text']]
            if final_test:command.append('--final_test')
            commands.append(command)
    plan={'commands':commands,'executed':False,'parent_and_B_use_identical_runtime_settings':True,
          'historical_predictions_not_used':'Evaluate the parent again in the same current runtime as B.',
          'final_test':final_test,'protocol_status':cfg['protocol_status']}
    write_new(eval_dir/('final_evaluation_plan.json' if final_test else 'development_evaluation_plan.json'),plan)
    if execute:
        if cfg['protocol_status']!='frozen':raise ValueError('Freeze the shared evaluation/training protocol before a controlled evaluation')
        check=parent_check(root,key,entry)
        if not check['ready']:raise ValueError(json.dumps(check))
        summary=read(run/'training_summary.json')
        if summary.get('complete') is not True:raise ValueError('B training is incomplete')
        run_config=read(run/'run_config.json')
        if run_config.get('parent_profile')!=key or run_config.get('additional_budget_multiplier')!=multiplier or run_config.get('shared_universe_sha256')!=cfg['common_universe_sha256']:
            raise ValueError('The requested B run uses a different parent, budget, or universe')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Evaluation execution requires exactly one NVIDIA GPU; CPU preparation does not run evaluations')
        model_environment(root)
        for command in commands:subprocess.run(command,check=True)
    if not quiet:print(json.dumps(plan,indent=2))
    return plan


def plan_evaluations(root,cfg):
    planned=[]
    for size in cfg['models']:
        for seed in cfg['parent_training_seeds']:
            for multiplier in cfg['budget_multipliers']:
                plan=evaluate(root,cfg,size,seed,multiplier,quiet=True)
                planned.append({'model_size':size,'parent_training_seed':seed,'budget_multiplier':multiplier,
                                'evaluation_commands':len(plan['commands']),'executed':False})
    write_new(work_root(root)/'evaluation_plan_index.json',{'plans':planned,'final_tests_planned':False,'any_model_execution':False})
    print(f'Prepared {len(planned)} development-evaluation plans. No commands were executed.')
    return planned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--offline',action='store_true',help='Use existing Hugging Face cache only')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['prepare','status','prefetch','verify','plan-evaluations']:commands.add_parser(name)
    p=commands.add_parser('import-parents');p.add_argument('--zip',required=True,type=Path)
    p=commands.add_parser('compare');p.add_argument('--parent-predictions',required=True,type=Path);p.add_argument('--B-predictions',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    for name in ['train','evaluate']:
        p=commands.add_parser(name);p.add_argument('--model',choices=['0.5B','1.5B'],required=True);p.add_argument('--parent-seed',type=int,choices=[17,123],required=True);p.add_argument('--budget',type=int,choices=[1,2,4],default=1)
        if name=='train':
            p.add_argument('--training-seed',type=int,default=42);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--run-id')
        else:
            p.add_argument('--trained-run',type=Path);p.add_argument('--final-test',action='store_true');p.add_argument('--execute',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    if args.offline:os.environ['HF_HUB_OFFLINE']='1'
    if args.command=='prepare':prepare(args.root,cfg)
    elif args.command=='status':status(args.root,cfg)
    elif args.command=='import-parents':import_parents(args.root,args.zip)
    elif args.command=='prefetch':prefetch(args.root,cfg,args.offline)
    elif args.command=='verify':verify(args.root,cfg,args.offline)
    elif args.command=='plan-evaluations':plan_evaluations(args.root,cfg)
    elif args.command=='compare':
        from compare import validate_pair
        report=validate_pair(args.parent_predictions,args.B_predictions);write_new(args.out,report);print(json.dumps(report,indent=2))
    elif args.command=='train':
        from continuation import train
        train(args.root,cfg,args.model,args.parent_seed,args.budget,args.training_seed,args.smoke,args.resume,args.run_id)
    elif args.command=='evaluate':evaluate(args.root,cfg,args.model,args.parent_seed,args.budget,args.trained_run,args.final_test,args.execute)


if __name__=='__main__':main()

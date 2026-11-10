"""Thin, guarded adaptation of Person 2's existing training engine.

The token-weighted loss loop, QLoRA loader, accumulation, clipping, checkpoint
format and reload probe are reused. Changes are explicit: accept the verified
parent metadata, retain parent adapter weights, use a fresh optimizer and
warmup, and enforce the shared global token budget. No parent optimizer/pickle
is loaded. Only checkpoints created by this runner may resume optimizer state.
"""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from common import HERE, sha, read, write_new, config, registry, data_paths, work_root, budget_name, budget_target, profile, load_rows, load_person2


def replace_once(source, old, new):
    if source.count(old)!=1:
        raise RuntimeError('Person 2 source differs from the inspected version; adaptation refused: '+old[:90])
    return source.replace(old,new,1)


def adapted_train_source():
    path=HERE/'vendor/person2/trainer_core.py';source=path.read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    code=ast.get_source_segment(source,function)
    code=replace_once(code,"experiment_status='preliminary' if args.mode == 'preliminary' else args.mode,", "experiment_status='smoke_only' if args.smoke else 'controlled_continuation',")
    code=replace_once(code,"evaluation_readiness='pending' if args.mode == 'preliminary' else 'see_mode_and_clearance',", "evaluation_readiness='smoke_not_evaluable' if args.smoke else 'see_frozen_protocol_and_parent_disclosure',")
    code=replace_once(code,"mode=args.mode, arm=args.arm if args.mode == 'branch' else None,", "mode='debug' if args.smoke else args.mode, arm=args.arm,")
    code=replace_once(code,"branch_optimizer_policy='restore_parent_no_new_warmup' if args.mode == 'branch' else None,", "branch_optimizer_policy='fresh_optimizer_and_scheduler',")
    code=replace_once(code,'    restore = None\n', '    config.update(args.continuation_metadata)\n    restore = None\n')
    start=code.index('    if args.parent:\n');end=code.index('    if args.resume:\n',start)
    replacement='''    if not args.parent:
        raise ValueError('Continuation must start from the verified trained parent adapter')
    pcfg = read_json(Path(args.parent) / 'run_config.json')
    for k in ('model', 'revision'):
        if pcfg.get(k) != config[k]:
            raise ValueError(f'Parent configuration differs: {k}')
    parent_state = args.parent_metadata
    config['branch_budget'] = parent_budget_check(args, rows, encoded, pcfg, parent_state)
    config['parent_adapter_sha256'] = sha(Path(args.parent) / 'adapter' / 'adapter_model.safetensors')
    adapter = Path(args.parent) / 'adapter'
'''
    code=code[:start]+replacement+code[end:]
    old='''    if parent_state:
        optimizer.load_state_dict(parent_state['optimizer']); scaler.load_state_dict(parent_state['scaler'])
        state['parent_optimizer_updates'] = parent_state['state']['optimizer_updates']
        # Same branch seed + fresh deterministic data order; parent optimizer restored for every arm.
'''
    code=replace_once(code,old,'    # Parent weights are retained; parent optimizer/scaler/RNG state is never restored.\n')
    code=replace_once(code,"if args.mode == 'debug' else None", "if (args.mode == 'debug' or args.smoke) else None")
    code=replace_once(code,"lr = args.lr if args.mode == 'branch' else args.lr * min(update / 10.0, 1.0)", "lr = args.lr * min(update / 10.0, 1.0)")
    return code


def warmup_rate(base,update):
    return base*min(update/10.0,1.0)


def run_fingerprint(entry,cfg,training_seed):
    return {'parent_adapter_sha256':entry['expected_adapter_sha256'],'model':entry['model'],'revision':entry['revision'],
            'training_seed':training_seed,'recipe':cfg['training'],
            'effective_train_source_sha256':hashlib.sha256(adapted_train_source().encode()).hexdigest(),
            'continuation_wrapper_sha256':sha(HERE/'continuation.py')}


def validate_protocol(root,cfg):
    if cfg['protocol_status']!='frozen':
        raise ValueError('Main training is blocked until the common experiment protocol is frozen. Data preparation and smoke checks remain independent.')
    path=Path(cfg.get('protocol_evidence') or '')
    if not path.is_absolute():path=Path(root)/path
    if not path.is_file():raise ValueError('Provide the actual shared protocol JSON as protocol_evidence')
    evidence=read(path)
    shared=['common_universe_sha256','parent_manifest_sha256','models','parent_training_seeds',
            'continuation_training_seeds','budget_multipliers','selection_seed','universe_seed',
            'quota_seed','training_order_seed','relative_token_tolerance','training','evaluation']
    for name in shared:
        expected=cfg[name]
        if evidence.get(name)!=expected:raise ValueError('Shared protocol evidence differs: '+name)
    if evidence.get('status')!='frozen' or not evidence.get('recorded_by'):
        raise ValueError('Protocol evidence must identify its real status and recorder')
    return sha(path)


def validate_resume(root,path,entry,manifest_sha,code_sha):
    path=Path(path).resolve()
    if not path.is_relative_to((work_root(root)/'runs').resolve()):
        raise ValueError('Only this runner\'s own continuation checkpoints may restore optimizer state')
    record=read(path/'continuation_checkpoint.json')
    if record.get('kind')!='B_continuation' or record.get('smoke') is not False:
        raise ValueError('Cannot resume main training from a parent/foreign/smoke checkpoint')
    for name,expected in [('parent_adapter_sha256',entry['expected_adapter_sha256']),('manifest_sha256',manifest_sha),('effective_train_source_sha256',code_sha)]:
        if record.get(name)!=expected:raise ValueError('Resume checkpoint differs: '+name)
    for name,expected in record['file_sha256'].items():
        if sha(path/name)!=expected:raise ValueError('Resume checkpoint file changed: '+name)


def budget_check(args,rows,encoded,parent_config,parent_state):
    seen=set(parent_state['state']['seen_groups'])
    if seen & {r['problem_group_id'] for r in rows}:raise ValueError('Fresh B contains a parent-seen group')
    actual=sum(e['counts']['supervised_tokens'] for e in encoded)
    target=budget_target(parent_state['one_pass_supervised_tokens'],args.budget_multiplier)
    if abs(actual-target)*200>target:raise ValueError('Actual label-derived continuation budget exceeds +/-0.5%')
    return {'additional_budget_multiplier':args.budget_multiplier,'actual_supervised_tokens':actual,'target_supervised_tokens':target,
            'parent_historical_processed_tokens':parent_state['state']['processed_tokens'],'global_tolerance':0.005,'tolerance_relaxed':False}


def train(root,cfg,size,parent_seed,multiplier,training_seed=42,smoke=False,resume=None,run_id=None):
    from workbench import parent_check,model_environment
    if multiplier not in cfg['budget_multipliers']:raise ValueError('Budget is absent from the declared experiment matrix')
    key,entry=profile(size,parent_seed)
    if size not in cfg['models'] or parent_seed not in cfg['parent_training_seeds']:
        raise ValueError('Model/parent seed is absent from the declared experiment matrix')
    check=parent_check(root,key,entry)
    if not check['ready']:raise ValueError('Matching parent adapter is not ready: '+json.dumps(check))
    if training_seed not in cfg['continuation_training_seeds']:
        raise ValueError('Continuation training seed is absent from the declared protocol')
    recipe=cfg['training']
    if recipe['epochs']!=1 or recipe['microbatch_size']!=1 or recipe['gradient_accumulation']!=8 or recipe['warmup_updates']!=10 or recipe['learning_rate']!=1e-4:
        raise ValueError('This adaptation preserves the inspected Person 2 recipe; version the implementation before changing it')
    receipt_hash=None if smoke else validate_protocol(root,cfg)
    folder=work_root(root)/budget_name(multiplier)
    data=folder/'train_B.jsonl';budget=read(folder/'budget_report.json')
    if sha(data)!=budget['selected_manifest_sha256']:raise ValueError('Selected manifest changed')
    verification_path=work_root(root)/'label_checks'/size/'verification.json'
    verification=read(verification_path)
    if verification.get('passed') is not True or verification.get('real_tokenizer_execution') is not True or verification.get('model_size')!=size:
        raise ValueError('Real model-specific tokenizer verification has not passed')
    if verification.get('verification_code_sha256')!=sha(HERE/'workbench.py'):
        raise ValueError('Tokenizer verification code changed; refresh its report')
    selected_check=next((x for x in verification['budgets'] if x['multiplier']==multiplier),None)
    if not selected_check or selected_check['manifest_sha256']!=sha(data) or not selected_check['within_0_5_percent']:
        raise ValueError('No current verified labels/budget for this selected manifest')
    parent_rows=load_rows(data_paths(root)['parent'])
    parent_groups={r['problem_group_id'] for r in parent_rows}
    records=HERE/entry['records_relative_path']
    seen={r['problem_group_id'] for r in csv.DictReader((records/'seen_groups.csv').open())}
    if seen!=parent_groups:raise ValueError('Original parent seen-group records differ from the parent manifest')
    parent_summary=read(records/'training_summary.json')
    if parent_summary.get('complete') is not True:raise ValueError('Parent training record is incomplete')
    fingerprint=run_fingerprint(entry,cfg,training_seed)
    smoke_path=work_root(root)/'smokes'/f'{key}_t{training_seed}'/'smoke_report.json'
    if not smoke:
        evidence=read(smoke_path)
        if evidence.get('passed') is not True or evidence.get('fingerprint')!=fingerprint:
            raise ValueError('A real matching parent/recipe loading-update-reload smoke check is required')
    code=adapted_train_source();code_hash=hashlib.sha256(code.encode()).hexdigest()
    if resume:validate_resume(root,resume,entry,sha(data),code_hash)
    default_id=f'B_{size}_p{parent_seed}_t{training_seed}_{budget_name(multiplier)}'
    run_id=run_id or (default_id+'_smoke' if smoke else default_id)
    if Path(run_id).name!=run_id or run_id in ['.','..']:raise ValueError('run-id must be a simple directory name')
    output=work_root(root)/'runs'/run_id
    if output.exists():raise FileExistsError('Use a new run-id/output folder; previous work is preserved')

    # No framework import or model download occurs before the data/parent gates above.
    model_environment(root)
    import torch
    from importlib import metadata
    if not torch.cuda.is_available():raise ValueError('A single NVIDIA/CUDA allocation is required for this stage')
    if torch.cuda.device_count()!=1:raise ValueError('Request/expose exactly one NVIDIA GPU for this recipe')
    expected_env=read(records/'run_config.json')['environment']['packages']
    mismatches={name:{'expected':version,'actual':metadata.version(name)} for name,version in expected_env.items() if metadata.version(name)!=version}
    if mismatches:raise ValueError('Training environment differs from the recorded recipe: '+json.dumps(mismatches))
    p2=load_person2();p2.select_model(size);core=p2.core
    # Recompile only train(); every other original core function remains available.
    exec(compile(code,str(HERE/'vendor/person2/trainer_core.py')+':fresh-continuation','exec'),core.__dict__)
    core.parent_budget_check=budget_check
    def continuation_guard(args,rows,current):
        # Replace the old curve/replay-specific gate with concrete continuation
        # checks, without asserting any unperformed manual-review clearance.
        if args.mode!='branch' or args.arm!='RANDOM' or args.epochs!=1:
            raise ValueError('This entry point permits only one-pass B continuation')
        if current['manifest_sha256']!=selected_check['manifest_sha256']:
            raise ValueError('Verification is for a different exposure manifest')
        if (current['model'],current['revision'])!=(entry['model'],entry['revision']):
            raise ValueError('Wrong model profile')
        if len(rows)!=budget['selected_rows'] or any(r['problem_group_id'] in parent_groups for r in rows):
            raise ValueError('Exposure count/freshness changed')
        if verification['parent_manifest_sha256']!=cfg['parent_manifest_sha256'] or verification['universe_sha256']!=cfg['common_universe_sha256']:
            raise ValueError('Tokenizer verification used different shared data')
        if not args.smoke and not receipt_hash:
            raise ValueError('Controlled main training lacks a shared protocol receipt')
        return None
    core.provenance_guard=continuation_guard
    files=work_root(root)/'training_inputs'
    contract=read(data_paths(root)['contract'])
    template=files/'prompt_template.json';policy=files/'tokenization_policy.json'
    write_new(template,{'template':contract['prompt_template']})
    write_new(policy,{'confirmed_with_person1':True,'add_special_tokens':False,'append_eos':True,'max_total_tokens':contract['max_total_tokens'],
                      'provenance':'Recovered, hash-matching Person 1 v3 encoder and model-specific actual label verification; see source_provenance.json.'})
    core.get_tokenizer=p2.get_tokenizer
    original_save=core.save_checkpoint
    def save(path,model,optimizer,scaler,tok,state,run_config,encoded,dtype):
        original_save(path,model,optimizer,scaler,tok,state,run_config,encoded,dtype)
        path=Path(path)
        write_new(path/'continuation_checkpoint.json',{'kind':'B_continuation','smoke':smoke,
          'parent_adapter_sha256':entry['expected_adapter_sha256'],'manifest_sha256':sha(data),
          'effective_train_source_sha256':code_hash,
          'file_sha256':{name:sha(path/name) for name in ['training_state.pt','probe.pt','run_config.json','adapter/adapter_model.safetensors']}})
    core.save_checkpoint=save
    args=SimpleNamespace(data=str(data),template=str(template),policy=str(policy),verification=str(verification_path),
        out=str(output),mode='branch',arm='RANDOM',parent=str((Path(root)/entry['adapter_relative_path']).parent),
        epochs=1,seed=training_seed,accumulation=8,lr=1e-4,save_every=recipe['checkpoint_every_updates'],
        resume=str(resume) if resume else None,stop_after=1 if smoke else None,clearance=None,
        smoke=smoke,budget_multiplier=multiplier,
        parent_metadata={'state':{'seen_groups':sorted(parent_groups),'processed_tokens':parent_summary['processed_tokens']},
                         'one_pass_supervised_tokens':verification['actual_parent_supervised_tokens']},
        continuation_metadata={'continuation_kind':'fresh_parent','smoke_only':smoke,'parent_profile':key,
          'additional_budget_multiplier':multiplier,'shared_universe_sha256':cfg['common_universe_sha256'],
          'protocol_receipt_sha256':receipt_hash,'continuation_training_seed':training_seed,
          'effective_train_source_sha256':code_hash,'continuation_wrapper_sha256':sha(HERE/'continuation.py'),
          'parent_related_development_groups_to_disclose':7,'initial_optimizer_state':'empty','fresh_warmup_updates':10})
    try:core.train(args)
    finally:core.save_checkpoint=original_save
    if sha(Path(root)/entry['adapter_relative_path']/'adapter_model.safetensors')!=entry['expected_adapter_sha256']:
        raise RuntimeError('Parent adapter changed during continuation')
    result=read(output/'training_summary.json')
    if smoke:
        if result.get('max_adapter_delta_this_segment',0)<=0:raise RuntimeError('Smoke produced no numerical parameter change')
        checkpoint=Path(result['checkpoint']);tok=core.get_tokenizer()
        row=core.encode_row(core.load_rows(data)[0],tok,contract['prompt_template'],contract['max_total_tokens'])
        dtype=torch.bfloat16 if torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported() else torch.float16
        torch.cuda.empty_cache()
        reloaded=core.build_model(dtype,adapter=checkpoint/'adapter',trainable=False)
        observed=core.probe_logits(reloaded,row,dtype)
        saved=torch.load(checkpoint/'probe.pt',map_location='cpu',weights_only=True)
        if not torch.allclose(observed,saved,rtol=1e-3,atol=1e-3):raise RuntimeError('Smoke reload logits differ')
        report={'passed':True,'fingerprint':fingerprint,'actual_gpu_execution':True,'numerical_adapter_change':result['max_adapter_delta_this_segment'],
                'reload_probe_matched':True,'optimizer_updates':result['optimizer_updates'],'processed_tokens':result['processed_tokens'],
                'checkpoint':str(checkpoint),'not_a_main_experiment':True}
        write_new(smoke_path,report);print(json.dumps(report,indent=2))
    else:
        if result.get('complete') is not True or result['processed_tokens']!=selected_check['actual_supervised_tokens'] or result['unique_examples']!=budget['selected_rows']:
            raise RuntimeError('Training consumption does not reconcile with the selected one-pass exposure manifest')
        write_new(output/'B_completion_audit.json',{'passed':True,'actual_consumed_tokens':result['processed_tokens'],
                  'expected_tokens':selected_check['actual_supervised_tokens'],'unique_rows':result['unique_examples'],
                  'parent_adapter_unchanged':True,'fresh_optimizer_and_warmup':True})
    return result

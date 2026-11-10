#!/usr/bin/env python3
"""Run document-aligned Method B in the existing Sol project.

Original workbenches, parents, datasets, environments, and results are preserved.
CP-SAT lives in its own CPU environment; training uses the tested Python 3.12 venv.
"""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
from datetime import datetime, timezone
from fractions import Fraction
from collections import Counter, defaultdict

from evaluation_recovery import archive_prediction_bundle, enter

from select_b import (ORTOOLS_VERSION, VERSION, bounds, cell, csv_bytes, digest,
                      edge_report, fraction, hash32, json_bytes, jsonl, number,
                      parent_edges, read, rebin, rows, solve, utility, validate, write)

HERE = Path(__file__).resolve().parent
STUDY = 'B_method_B_document_v3'


def output(root):
    return Path(root) / 'outputs' / STUDY


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def update(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.incomplete')
    tmp.write_bytes(json_bytes(value)); tmp.replace(path)


def call(command, root, label, *, capture=False):
    log = output(root) / 'logs' / f'{label}_{stamp()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); lines = []
    print('Running:', ' '.join(map(str, command)), flush=True)
    print('Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as f:
        env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        with subprocess.Popen(list(map(str, command)), cwd=str(HERE), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as p:
            for line in p.stdout:
                print(line, end='', flush=True); f.write(line); f.flush()
                if capture:
                    lines.append(line)
            code = p.wait()
    update(log.with_suffix('.status.json'), {'command': list(map(str, command)),
           'exit_code': code, 'wall_seconds': time.monotonic() - started, 'log': str(log)})
    if code:
        raise RuntimeError(f'Stage failed (exit {code}). See {log}. Earlier outputs were preserved.')
    return ''.join(lines)


def training_python(root):
    p = Path(root) / 'venv_b_training_py312/bin/python'
    if not p.is_file():
        raise FileNotFoundError('Existing training environment is missing: ' + str(p))
    return p


def source(root):
    p = Path(root) / 'b_workbench_500'
    manifest = read(HERE / 'source_hashes.json')
    for name, expected in manifest.items():
        if digest(p / name) != expected:
            raise ValueError(f'Your working source differs from the reviewed copy: {p/name}. '
                             'No files were replaced; share the changed source before adapting it.')
    for name, expected in read(p / 'source_provenance.json')['vendor_sha256'].items():
        if digest(p / name) != expected:
            raise ValueError('Pinned vendor source/data differs: ' + str(p / name))
    for name, expected in read(p / 'source_provenance.json')['parent_reference_records_sha256'].items():
        if digest(p / name) != expected:
            raise ValueError('Pinned parent record differs: ' + str(p / name))
    return p


def solver_python(root):
    return Path(root) / 'venv_b_solver_v3/bin/python'


def install_solver(root):
    target = solver_python(root)
    if not target.is_file():
        call([training_python(root), '-m', 'venv', target.parent.parent], root, 'create_solver_venv')
    probe = subprocess.run([str(target), '-c', 'import ortools; print(ortools.__version__)'],
                           capture_output=True, text=True)
    if probe.returncode or probe.stdout.strip() != ORTOOLS_VERSION:
        call([target, '-m', 'pip', 'install', '--only-binary=:all:',
              f'ortools=={ORTOOLS_VERSION}'], root, 'install_ortools')
    call([target, HERE / 'test_method_b.py', '--require-solver'], root, 'solver_acceptance_tests')
    freeze = subprocess.check_output([str(target), '-m', 'pip', 'freeze'])
    write(output(root) / 'solver_requirements.lock.txt', freeze)
    print('SOLVER_READY. Your training environment was not changed.', flush=True)


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def paired_item_interval(parent_path, b_path, replicates=2000):
    """A fixed-checkpoint item bootstrap; it does not estimate training variance."""
    import random
    parent={r['problem_id']:r['correct'] for r in rows(parent_path)}
    b={r['problem_id']:r['correct'] for r in rows(b_path)}
    if set(parent)!=set(b) or not parent:raise ValueError('Cannot bootstrap unpaired/empty predictions')
    differences=[int(b[k])-int(parent[k]) for k in sorted(parent)]
    rng=random.Random(1729)
    sampled=sorted(100*sum(rng.choices(differences,k=len(differences)))/len(differences)
                   for _ in range(replicates))
    return {'paired_item_ci95_low_pp':sampled[int(0.025*(replicates-1))],
            'paired_item_ci95_high_pp':sampled[math_ceil(0.975*(replicates-1))],
            'bootstrap_replicates':replicates,
            'uncertainty_scope':'paired items for these fixed checkpoints; training variability is separate'}


def math_ceil(value):
    import math
    return math.ceil(value)


def build_banks(root):
    old = source(root); cfg = read(old / 'config.json')
    root = Path(root); shared = output(root) / 'shared'
    parent_path = root / 'data/train_n1000.jsonl'
    pool_path = root / 'outputs/common_pool_review_v1/candidate_pool.jsonl'
    summary_path = pool_path.parent / 'pool_summary.json'
    if digest(parent_path) != cfg['parent_manifest_sha256']:
        raise ValueError('Actual parent manifest differs from the verified R1000 manifest')
    if digest(pool_path) != cfg['screened_pool_sha256']:
        raise ValueError('Screened pool differs from the reviewed 9,909-example bank')
    summary = read(summary_path)
    if summary.get('review_queue_complete') is not True or summary.get('pool_sha256') != cfg['screened_pool_sha256']:
        raise ValueError('Missing/incompatible completed screening evidence')
    parent, pool = rows(parent_path), rows(pool_path)
    if len(parent) != 1000 or len(pool) != 9909:
        raise ValueError('Unexpected verified parent/pool row counts')
    validate(parent); validate(pool)
    contract = root / 'inputs/formatting_contract.json'
    if digest(contract) != cfg['formatting_contract_sha256']:
        raise ValueError('Renderer/tokenization contract changed')
    # Check the UNION of actual groups seen by every supplied parent profile.
    union = set()
    for entry in read(old / 'parent_registry.json').values():
        with (old / entry['records_relative_path'] / 'seen_groups.csv').open(newline='') as f:
            union.update(r['problem_group_id'] for r in csv.DictReader(f))
    if union != {r['problem_group_id'] for r in parent}:
        raise ValueError('The supplied parents have different seen manifests; rebuild the shared exclusions first')
    if union & {r['problem_group_id'] for r in pool}:
        raise ValueError('Screened pool overlaps actual parent-seen groups')
    if {r['row_id'] for r in parent} & {r['row_id'] for r in pool}:
        raise ValueError('Screened pool overlaps actual parent row IDs')
    protected = root / 'inputs/protected_questions.csv'
    if not protected.is_file():
        protected = old / 'inputs_snapshot/protected_questions.csv'
    with protected.open(newline='', encoding='utf-8') as f:
        protected_text = {normalize(r['problem']) for r in csv.DictReader(f)}
    if protected_text & {normalize(r['problem']) for r in pool}:
        raise ValueError('Screened pool has an exact protected-question overlap')
    edges = parent_edges(parent)
    parent_rebinned, pool = rebin(parent, edges), rebin(pool, edges)
    numina_parent = [r for r in parent_rebinned if r['dataset_id'] == 'numina']
    if not numina_parent:
        raise ValueError('Cannot reserve the documented transfer condition: parent has no Numina rows')
    # The old parents have unverified development protection. Reserve the
    # documented clean fallback: 1,000 screened, group-disjoint examples,
    # 500/source. Export references for the common evaluator to ingest.
    development=[]
    for source_name in sorted({r['dataset_id'] for r in parent_rebinned}):
        source_parent=[r for r in parent_rebinned if r['dataset_id']==source_name]
        source_counts=Counter(cell(r) for r in source_parent)
        if len(source_parent)!=500:
            raise ValueError('This fallback development reserve expects the verified 500/source parent')
        for h,n in sorted(source_counts.items()):
            candidates=sorted((r for r in pool if cell(r)==h),
                              key=lambda r:(hash32('1729|clean-development|'+r['row_id']),r['row_id']))
            if len(candidates)<n: raise ValueError('Insufficient clean development capacity: '+str(h))
            development.extend(candidates[:n])
    dev_groups={r['problem_group_id'] for r in development}
    remaining=[r for r in pool if r['problem_group_id'] not in dev_groups]
    # Reserve 5x each source-specific parent stratum before any mixed-source draw.
    reserve = []
    by = defaultdict(list)
    for r in remaining:
        by[cell(r)].append(r)
    for h, n in sorted(Counter(cell(r) for r in numina_parent).items()):
        candidates = sorted(by[h], key=lambda r: (hash32('1729|transfer-reserve|' + r['row_id']), r['row_id']))
        if len(candidates) < 5 * n:
            raise ValueError(f'Insufficient screened transfer capacity in {h}: need {5*n}, have {len(candidates)}. '
                             'No main-bank draw was made and no raw candidates were substituted.')
        reserve.extend(candidates[:5*n])
    reserved = {r['problem_group_id'] for r in reserve}
    mixed = sorted((r for r in remaining if r['problem_group_id'] not in reserved), key=lambda r: r['row_id'])
    reserve.sort(key=lambda r: r['row_id'])
    write(shared / 'parent_length_bins.json', edge_report(edges))
    write(shared / 'parent_original_manifest.jsonl', parent_path.read_bytes())
    write(shared / 'parent_seen_groups.csv', ('problem_group_id\n'+'\n'.join(sorted(union))+'\n').encode())
    write(shared / 'parent_rebinned_reference.jsonl', jsonl(parent_rebinned))
    write(shared / 'clean_development_reference.jsonl', jsonl(sorted(development,key=lambda r:r['row_id'])))
    write(shared / 'mixed_candidate_bank.jsonl', jsonl(mixed))
    write(shared / 'transfer_numina_bank.jsonl', jsonl(reserve))
    report = {'version': VERSION, 'source_document_sha256': read(HERE/'source_hashes_meta.json')['method_document_sha256'],
              'parent_manifest_sha256': digest(parent_path), 'parent_rows': len(parent),
              'parent_supervised_tokens': sum(r['supervised_tokens'] for r in parent),
              'screened_pool_rows': len(pool), 'screened_pool_sha256': digest(pool_path),
              'clean_development_rows':len(development),
              'clean_development_sha256':digest(shared/'clean_development_reference.jsonl'),
              'clean_development_evaluator_status':'Reference rows reserved and exported; ingestion/parser validation by the shared evaluator is still required.',
              'mixed_bank_rows': len(mixed), 'mixed_source_rows': dict(Counter(r['dataset_id'] for r in mixed)),
              'mixed_bank_sha256': digest(shared/'mixed_candidate_bank.jsonl'),
              'transfer_rows': len(reserve), 'transfer_bank_sha256': digest(shared/'transfer_numina_bank.jsonl'),
              'transfer_reference_rows': len(numina_parent),
              'transfer_reference_tokens': sum(r['supervised_tokens'] for r in numina_parent),
              'length_bins_sha256': digest(shared/'parent_length_bins.json'),
              'protected_questions_sha256': digest(protected), 'parent_union_checked': True,
              'parent_seen_group_overlap':0,
              'exact_protected_overlap': 0, 'screening_evidence_sha256': digest(summary_path),
              'near_overlap_scope': 'The pinned screened pool and its completed review; no exhaustive pretraining claim',
              'bank_capacity_note': '10,000 is a target subject to screened capacity. '
                  'This study uses all eligible remaining rows after reserving disjoint development and transfer banks.',
              'shared_with_other_methods': 'These artifacts define common inputs for A/B/C; adoption by other runs is not asserted.',
              'parent_related_development_groups_to_disclose': cfg['scientific_scope'].get('parent_related_development_groups_to_disclose'),
              'evaluation_status': 'Preliminary until the shared evaluator and parent/development protection are verified.'}
    write(shared / 'bank_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print('BANKS_READY. Preserve/share this bank, bins, and audit for all fresh selectors.', flush=True)


def setting_root(root, setting, budget):
    return output(root) / setting / f'budget_{float(budget):g}x'


def bridge_root(root, setting, budget):
    return setting_root(root, setting, budget) / 'runtime'


def make_bridge(root, setting, budget, selection, report, parent_seeds):
    old = source(root); shared = output(root) / 'shared'; audit = read(shared/'bank_audit.json')
    dest = bridge_root(root, setting, budget); dest.mkdir(parents=True, exist_ok=True)
    for name in ['common.py', 'workbench.py', 'continuation.py', 'compare.py', 'allocator.py',
                 'parent_registry.json', 'source_provenance.json']:
        content = (old/name).read_text()
        if name == 'common.py':
            content += '\n\n# V3 explicit data routing; original helpers remain intact above.\n'
            content += 'def data_paths(root):\n    cfg=config()\n    return {k:Path(v) for k,v in cfg["resolved_data_paths"].items()}\n'
            content += 'def work_root(root):\n    return Path(config()["resolved_work_root"])\n'
            content += 'def budget_target(parent_tokens,multiplier):\n    value=Fraction(config()["budget_reference_tokens"])*Fraction(str(multiplier))\n    return value.numerator if value.denominator==1 else float(value)\n'
        if name == 'continuation.py':
            needle = 'stop_after=1 if smoke else None'
            if content.count(needle) != 1:
                raise ValueError('Inspected smoke-stop expression changed')
            content = content.replace(needle, 'stop_after=4 if smoke else None')
            label="'controlled_continuation'"
            if content.count(label)!=1:raise ValueError('Inspected experiment-status expression changed')
            content=content.replace(label,"'preliminary_document_aligned_B'")
        write(dest/name, content.encode())
    for directory in ('vendor', 'reference_records'):
        link = dest/directory
        if link.is_symlink():
            if link.resolve() != (old/directory).resolve():
                raise ValueError('Runtime link points to a different pinned source: '+str(link))
        elif link.exists():
            raise FileExistsError('Unexpected runtime source directory: '+str(link))
        else:
            link.symlink_to(old/directory, target_is_directory=True)
    cfg = read(old/'config.json')
    cfg.update(version=VERSION, protocol_status='frozen', protocol_evidence=str(dest/'execution_protocol.json'),
               parent_training_seeds=list(parent_seeds), continuation_training_seeds=[42,137,2026],
               budget_multipliers=[float(budget)],
               common_universe_sha256=audit['mixed_bank_sha256'] if setting=='mixed' else audit['transfer_bank_sha256'],
               resolved_work_root=str(setting_root(root,setting,budget)),
               budget_reference_tokens=audit['parent_supervised_tokens'] if setting=='mixed' else audit['transfer_reference_tokens'],
               reference_scope='full R1000 parent' if setting=='mixed' else 'Numina subset of actual R1000 parent',
               models=['0.5B','1.5B'] if setting=='mixed' else ['0.5B'],
               selection_implementation='document Appendix A CP-SAT',
               per_stratum_token_tolerance='0.05',
               source_document_sha256=read(HERE/'source_hashes_meta.json')['method_document_sha256'])
    protected = Path(root)/'inputs/protected_questions.csv'
    if not protected.is_file(): protected=old/'inputs_snapshot/protected_questions.csv'
    cfg['resolved_data_paths'] = {'parent':str(Path(root)/'data/train_n1000.jsonl'),
          'universe':str(shared/('mixed_candidate_bank.jsonl' if setting=='mixed' else 'transfer_numina_bank.jsonl')),
          'contract':str(Path(root)/'inputs/formatting_contract.json'), 'protected':str(protected)}
    cfg['notes'] = ['Execution settings locked for this B version from the supplied design and user instruction.',
                    'This is a reproducibility lock for B; it does not certify adoption by A/C or evaluator clearance.',
                    '0.5x and 2x are explicitly additional conditions. 0.5x uses largest-remainder stratum counts.',
                    'Every run starts independently from the original matching R1000 parent.',
                    'Training seeds 42/137/2026 are different from parent seeds 17/123.',
                    'All evaluation remains preliminary until the shared development/test protection is verified.']
    write(dest/'config.json', cfg)
    fields = ['common_universe_sha256','parent_manifest_sha256','models','parent_training_seeds',
              'continuation_training_seeds','budget_multipliers','selection_seed','universe_seed',
              'quota_seed','training_order_seed','relative_token_tolerance','training','evaluation']
    protocol = {k:cfg[k] for k in fields}
    protocol.update(status='frozen', recorded_by='Suraj Shinde — requested execution of the supplied Method B design',
                    scope='B execution settings only; cross-method compatibility must be checked against these artifacts',
                    selection_audit_sha256=digest(setting_root(root,setting,budget)/'selection_audit.json'),
                    length_bins_sha256=audit['length_bins_sha256'],
                    source_document_sha256=cfg['source_document_sha256'],
                    evaluation_clearance='not established by a protocol lock')
    write(dest/'execution_protocol.json',protocol)
    return dest


def prepare(root, setting, budget, parent_seeds):
    shared = output(root)/'shared'; audit=read(shared/'bank_audit.json')
    parent = rows(shared/'parent_rebinned_reference.jsonl')
    if setting == 'transfer':
        if float(budget) != 1: raise ValueError('The documented transfer condition uses its own 1x source-specific budget')
        parent = [r for r in parent if r['dataset_id']=='numina']
    bank_path = shared/('mixed_candidate_bank.jsonl' if setting=='mixed' else 'transfer_numina_bank.jsonl')
    if digest(bank_path) != audit['mixed_bank_sha256' if setting=='mixed' else 'transfer_bank_sha256']:
        raise ValueError('Common bank changed after audit')
    bank = rows(bank_path); dest=setting_root(root,setting,budget)
    report_path=dest/'selection_audit.json'
    selected_path=dest/f'{float(budget):g}x/train_B.jsonl'
    if report_path.is_file() and selected_path.is_file():
        report=read(report_path); selection=rows(selected_path)
        if report.get('passed') is not True or report.get('selected_manifest_sha256')!=digest(selected_path) or report.get('bank_sha256')!=digest(bank_path):
            raise ValueError('Existing selection failed identity checks')
        print('Reusing canonical selected dataset:',selected_path,flush=True)
    else:
        selection, report=solve(parent,bank,budget)
        report.update(setting=setting,bank_sha256=digest(bank_path),parent_manifest_sha256=audit['parent_manifest_sha256'],
                      parent_length_bins_sha256=audit['length_bins_sha256'])
        if selection is None:
            update(dest/f'selection_failure_{stamp()}.json',report)
            raise RuntimeError('No conforming selection was produced: '+report['status']+
                               '. Counts and tolerances were preserved; inspect the saved feasibility report.')
        selection.sort(key=lambda r:(hash32('42|training-order|'+r['row_id']),r['row_id']))
        write(selected_path,jsonl(selection))
        report['selected_manifest_sha256']=digest(selected_path)
        write(report_path,report)
        write(dest/f'{float(budget):g}x/selected_ids.txt', ('\n'.join(sorted(r['row_id'] for r in selection))+'\n').encode())
        selection_ids={s['row_id'] for s in selection}
        write(dest/f'{float(budget):g}x/candidate_priorities.csv',csv_bytes([
              {'row_id':r['row_id'],'utility_integer':utility(r['row_id']),
               'raw_first32':hash32('42|random|'+r['row_id']),
               'selected':r['row_id'] in selection_ids}
              for r in sorted(bank,key=lambda r:r['row_id'])]))
    # The old continuation engine expects these accounting fields; no data are changed.
    write(selected_path.parent/'budget_report.json',{
          'budget_multiplier':float(budget),'target_supervised_tokens':report['target_supervised_tokens'],
          'selected_rows':len(selection),'selected_supervised_tokens':report['actual_tokens'],
          'selected_manifest_sha256':digest(selected_path),'quota_deviations':0,
          'difference_percent':report['difference_percent'],'tolerance_relaxed':False,
          'selection_audit_sha256':digest(report_path),'reference_scope':setting})
    bridge=make_bridge(root,setting,budget,selection,report,parent_seeds)
    print(f'SELECTION_PASSED: {setting} {float(budget):g}x; {len(selection)} examples; '
          f'{report["actual_tokens"]} tokens; {report["difference_percent"]:+.5f}%.',flush=True)
    call([training_python(root),Path(__file__).resolve(),'engine','--root',root,'--setting',setting,
          '--budget',str(budget),'--action','verify'],root,f'{setting}_{budget}_verify')
    print('DOCUMENT_ALIGNED_PREPARATION_PASSED',flush=True)


def checked_runtime(root,setting,budget):
    dest=bridge_root(root,setting,budget)
    if not (dest/'config.json').is_file():
        raise FileNotFoundError('Prepare this condition first: '+str(dest))
    sys.path.insert(0,str(dest))
    import common,workbench,continuation
    cfg=common.config()
    common.sources_valid()
    continuation.validate_protocol(root,cfg)
    selection=setting_root(root,setting,budget)/f'{float(budget):g}x/train_B.jsonl'
    audit=read(setting_root(root,setting,budget)/'selection_audit.json')
    if digest(selection)!=audit['selected_manifest_sha256'] or digest(common.data_paths(root)['universe'])!=audit['bank_sha256']:
        raise ValueError('Data changed since constrained selection')
    return common,workbench,continuation,cfg


def engine(args):
    root=args.root
    os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_CACHE=str(root/'hf_cache'),TOKENIZERS_PARALLELISM='false')
    common,wb,cont,cfg=checked_runtime(root,args.setting,args.budget)
    if args.action=='verify':
        wb.verify(root,cfg,offline=True)
        for size in cfg['models']:
            verification=read(common.work_root(root)/'label_checks'/size/'verification.json')
            if verification['actual_parent_supervised_tokens']!=read(output(root)/'shared/bank_audit.json')['parent_supervised_tokens']:
                raise ValueError('Actual parent label counts differ from the verified reference')
            with (common.work_root(root)/'label_checks'/size/'rows.csv').open(newline='') as f:
                actual={r['row_id']:int(r['actual_supervised_tokens']) for r in csv.DictReader(f)}
            for row in rows(common.data_paths(root)['parent'])+rows(common.data_paths(root)['universe']):
                if actual[row['row_id']]!=row['supervised_tokens']:
                    raise ValueError('Actual labels differ from stored selection tokens: '+row['row_id'])
        write(common.work_root(root)/'label_counts_and_strata_verified.json',
              {'passed':True,'all_stored_counts_match_actual_labels':True,
               'selection_audit_sha256':digest(common.work_root(root)/'selection_audit.json'),
               'models':cfg['models']})
        return
    import torch
    verified=read(common.work_root(root)/'label_counts_and_strata_verified.json')
    if verified.get('passed') is not True or verified.get('selection_audit_sha256')!=digest(common.work_root(root)/'selection_audit.json'):
        raise ValueError('Actual label/stratum reconciliation is missing or stale')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('This stage requires a session exposing exactly one NVIDIA GPU')
    key,entry=common.profile(args.model,args.parent_seed)
    if args.parent_seed not in cfg['parent_training_seeds'] or args.model not in cfg['models']:
        raise ValueError('Profile absent from this condition')
    checked=wb.parent_check(root,key,entry)
    if not checked['ready']: raise ValueError(json.dumps(checked))
    work=common.work_root(root)
    if args.action=='smoke':
        path=work/'smokes'/f'{key}_t{args.training_seed}/smoke_report.json'
        if path.is_file():
            old=read(path)
            if old.get('passed') is not True or old.get('actual_gpu_execution') is not True or old.get('reload_probe_matched') is not True or old.get('fingerprint')!=cont.run_fingerprint(entry,cfg,args.training_seed) or old.get('optimizer_updates')!=4:
                raise ValueError('Existing smoke is incompatible; preserve it and use a new study version')
            print('Reusing successful four-update smoke:',key,flush=True);return
        cont.train(root,cfg,args.model,args.parent_seed,args.budget,args.training_seed,True,
                   run_id=f'B_{key}_t{args.training_seed}_{args.budget:g}x_smoke32')
        result=read(path)
        summary=read(Path(result['checkpoint']).parent/'training_summary.json')
        if result['optimizer_updates']!=4 or summary.get('unique_examples')!=32:
            raise ValueError('Smoke completed but did not establish the documented 32-example exposure; inspect summary')
        print('32_EXAMPLE_GPU_SMOKE_PASSED:',key,flush=True);return
    run_name=f'B_{args.model}_p{args.parent_seed}_t{args.training_seed}_{args.budget:g}x'
    index_path=work/'run_index.json'; index=read(index_path) if index_path.is_file() else {}
    current=work/'runs'/index.get(run_name,run_name)
    if args.action=='train':
        if current.exists():
            completion=current/'B_completion_audit.json'
            if completion.is_file() and read(completion).get('passed') is True:
                run_cfg=read(current/'run_config.json')
                if run_cfg.get('parent_profile')!=key or run_cfg.get('continuation_training_seed')!=args.training_seed:
                    raise ValueError('Completed run identity differs')
                if run_cfg.get('manifest_sha256')!=digest(work/f'{args.budget:g}x/train_B.jsonl'):
                    raise ValueError('Completed run used a different dataset')
                print('Already completed:',current,flush=True);return
            checkpoints=[p.parent for p in current.glob('*/continuation_checkpoint.json')]
            if not checkpoints:
                # A killed process before its first checkpoint is a fresh restart.
                resume=None
                print('No completed checkpoint: restart from original parent; previous folder preserved.',flush=True)
            else:
                resume=max(checkpoints,key=lambda p:(p/'continuation_checkpoint.json').stat().st_mtime)
            new_name=run_name+'_resume_'+stamp()
        else:
            resume=None;new_name=run_name
        index[run_name]=new_name;update(index_path,index)
        cont.train(root,cfg,args.model,args.parent_seed,args.budget,args.training_seed,False,
                   resume=resume,run_id=new_name)
        print('B_TRAINING_COMPLETED:',new_name,flush=True);return
    if args.action=='evaluate':
        if not (current/'B_completion_audit.json').is_file() or read(current/'B_completion_audit.json').get('passed') is not True:
            raise ValueError('A complete audited continuation is required before evaluation')
        run_cfg=read(current/'run_config.json')
        if run_cfg.get('parent_profile')!=key or run_cfg.get('continuation_training_seed')!=args.training_seed or run_cfg.get('manifest_sha256')!=digest(work/f'{args.budget:g}x/train_B.jsonl'):
            raise ValueError('Completed run identity/manifest differs from the requested evaluation')
        cont.validate_resume(root,current/'final',entry,digest(work/f'{args.budget:g}x/train_B.jsonl'),
                             cont.run_fingerprint(entry,cfg,args.training_seed)['effective_train_source_sha256'])
        from compare import validate_pair
        plan=wb.evaluate(root,cfg,args.model,args.parent_seed,args.budget,trained_run=current,execute=False,quiet=True)
        results={}
        eval_dir=work/'evaluation'/current.name
        signature_keys=['precision','batch_size','max_new_tokens','question_text','decoding','adapters']
        for command in plan['commands']:
            command=list(command)
            source_eval=Path(command[command.index('--eval_file')+1]);dataset=source_eval.stem
            parent_run=Path(command[command.index('--adapter')+1]).resolve()==Path(checked['adapter']).resolve()
            if parent_run:
                sig={'adapter_sha256':entry['expected_adapter_sha256'],'eval_file_sha256':digest(source_eval),
                     'evaluator_sha256':digest(common.HERE/'vendor/evaluation/evaluate.py'),
                     'settings':{k:cfg['evaluation'][k] for k in signature_keys},
                     'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__}
                cachekey=hashlib.sha256(json_bytes(sig)).hexdigest()[:20]
                pred=output(root)/'parent_evaluation_cache'/f'{key}_{dataset}_{cachekey}.jsonl'
                command[command.index('--out')+1]=str(pred)
                command[command.index('--run_id')+1]=f'{key}_parent_{dataset}_{cachekey}'
            else:
                pred=Path(command[command.index('--out')+1])
                sig={'run':str(current),'run_adapter_sha256':digest(current/'final/adapter/adapter_model.safetensors'),
                     'eval_file_sha256':digest(source_eval),'evaluator_sha256':digest(common.HERE/'vendor/evaluation/evaluate.py')}
            marker=pred.with_suffix('.validated.json')
            valid=False
            if marker.is_file() and pred.is_file():
                try:
                    prior=read(marker)
                    valid=prior.get('signature')==sig and prior.get('predictions_sha256')==digest(pred) and prior.get('config_sha256')==digest(pred.with_suffix('.config.json'))
                except (OSError,ValueError,KeyError):
                    valid=False
            if valid:
                try:
                    saved_runtime=read(pred.with_suffix('.config.json'))
                    valid=(saved_runtime.get('gpu')==torch.cuda.get_device_name(0) and
                           saved_runtime.get('torch')==torch.__version__ and
                           saved_runtime.get('cuda')==torch.version.cuda)
                except (OSError,ValueError):
                    valid=False
            if valid:
                validate_pair(pred,pred)
                print('Reusing complete predictions:',pred,flush=True)
            else:
                if pred.exists() or marker.exists() or pred.with_suffix('.config.json').exists():
                    archive_prediction_bundle(root,pred)
                pred.parent.mkdir(parents=True,exist_ok=True)
                call(command,root,f'eval_{current.name}_{"parent" if parent_run else "B"}_{dataset}')
                validate_pair(pred,pred)
                update(marker,{'signature':sig,'predictions_sha256':digest(pred),
                               'config_sha256':digest(pred.with_suffix('.config.json'))})
            results.setdefault(dataset,{})['parent' if parent_run else 'B']=str(pred)
        comparisons=[]
        for dataset,pair in results.items():
            result=validate_pair(pair['parent'],pair['B'])
            result.update(paired_item_interval(pair['parent'],pair['B']))
            result.update(dataset=dataset,parent_profile=key,training_seed=args.training_seed,
                          budget_multiplier=args.budget,evidence_status='preliminary development comparison',
                          development_protection='Parent/evaluator clearance still requires shared verification')
            comparisons.append(result)
        aggregate=sum(x['B_accuracy'] for x in comparisons)/len(comparisons)
        update(eval_dir/'B_vs_parent.json',{'comparisons':comparisons,
               'equal_weight_mean_B_accuracy':aggregate,
               'equal_weight_mean_parent_accuracy':sum(x['parent_accuracy'] for x in comparisons)/len(comparisons),
               'final_test_used':False,'parent_development_protection_verified':False})
        print(json.dumps(comparisons,indent=2),flush=True)


def gpu_stage(root,setting,budget,action,parent_seeds,training_seeds):
    cfg=read(bridge_root(root,setting,budget)/'config.json')
    for size in cfg['models']:
        for parent_seed in parent_seeds:
            for training_seed in training_seeds:
                command=[training_python(root),Path(__file__).resolve(),'engine','--root',root,
                      '--setting',setting,'--budget',str(budget),'--action',action,'--model',size,
                      '--parent-seed',str(parent_seed),'--training-seed',str(training_seed)]
                call(command,root,f'{setting}_{budget}_{action}_{size}_p{parent_seed}_t{training_seed}')


def summarize(root):
    all_rows=[];checks=[]
    for path in sorted(output(root).glob('*/budget_*/evaluation/*/B_vs_parent.json')):
        record=read(path)
        for r in record['comparisons']:
            all_rows.append(dict(setting=path.relative_to(output(root)).parts[0],**r))
    for path in sorted(output(root).glob('*/budget_*/runs/*/B_completion_audit.json')):
        audit=read(path);cfg=read(path.parent/'run_config.json')
        checks.append({'run':str(path.parent),'passed':audit.get('passed') is True,
                       'parent_profile':cfg.get('parent_profile'),
                       'training_seed':cfg.get('continuation_training_seed'),
                       'additional_budget_multiplier':cfg.get('additional_budget_multiplier'),
                       'actual_consumed_tokens':audit.get('actual_consumed_tokens')})
    import statistics
    grouped=defaultdict(list)
    for r in all_rows:
        grouped[(r['setting'],r['parent_profile'],r['budget_multiplier'],r['dataset'])].append(r)
    seed_summary=[]
    for group,data in sorted(grouped.items()):
        deltas=[r['change_percentage_points'] for r in data]
        seed_summary.append({'setting':group[0],'parent_profile':group[1],'budget_multiplier':group[2],
              'dataset':group[3],'training_seeds':[r['training_seed'] for r in data],
              'mean_change_percentage_points':statistics.mean(deltas),
              'sample_sd_percentage_points':statistics.stdev(deltas) if len(deltas)>1 else None,
              'combined_item_and_training_confidence_interval':'not estimated; limited seed count'})
    update(output(root)/'progress_summary.json',{'completed_training_runs':checks,'development_comparisons':all_rows,
            'training_seed_summaries':seed_summary,
            'final_test_used':False,'remaining_shared_work':['A/C compatibility with the exported bank, bins, and controls',
                  'Clean development/evaluator protection evidence; disclose prior overlap/exposure',
                  'Frozen final model list and common final evaluation',
                  'B-minus-A / C-minus-B paired comparisons after other methods finish']})
    update(output(root)/'development_results.json',all_rows)
    (output(root)/'development_results.csv').write_bytes(csv_bytes(all_rows))
    print('Completed training runs:',len(checks),flush=True)
    for item in checks: print(item,flush=True)
    print('Development comparisons:',len(all_rows),flush=True)
    print('Summary:',output(root)/'progress_summary.json',flush=True)


def estimate(root,setting,budget):
    work=setting_root(root,setting,budget)
    selected=read(work/'selection_audit.json')['actual_tokens']
    estimates=[]
    for path in sorted((work/'runs').glob('*smoke32/training_log.jsonl')):
        ticks=[r for r in rows(path) if 'optimizer_update' in r and r.get('elapsed_seconds',0)>0]
        if not ticks:continue
        last=ticks[-1]
        processed=last.get('cumulative_processed_tokens',0)
        if processed<=0:continue
        cfg=read(path.parent/'run_config.json')
        seconds=selected*last['elapsed_seconds']/processed
        estimates.append({'profile':cfg.get('parent_profile'),'selected_supervised_tokens':selected,
               'measured_smoke_training_seconds':last['elapsed_seconds'],
               'measured_smoke_tokens':processed,
               'projected_training_minutes':round(seconds/60,1),
               'rough_range_minutes':[round(seconds*0.5/60,1),round(seconds*2/60,1)],
               'excludes':'Model loading, checkpoint IO, evaluation, queueing, and long-run variability'})
    update(work/'training_time_projection.json',estimates)
    print(json.dumps(estimates,indent=2),flush=True)
    if not estimates:print('No readable training timing records yet. Timing remains unmeasured for this selection.',flush=True)


def export(root):
    summarize(root)
    target=Path(root)/f'method_b_handoff_{stamp()}.tar.gz'
    base=output(root)
    # Package selection/evidence/logs/source and final adapters; omit duplicate
    # intermediate checkpoints, base models, caches, environments, and vendor copies.
    with tarfile.open(target,'w:gz') as archive:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.is_symlink(): continue
            relative=path.relative_to(base)
            if any(x in relative.parts for x in ('vendor','reference_records','__pycache__')): continue
            if 'runs' in relative.parts:
                parts=list(relative.parts);i=parts.index('runs')
                if len(parts)>i+2 and parts[i+2] not in ('final','training_summary.json','training_log.jsonl',
                                                      'run_config.json','B_completion_audit.json'):
                    continue
                if path.name=='training_state.pt':continue
            archive.add(path,arcname=str(Path(STUDY)/relative),recursive=False)
        for path in sorted(HERE.glob('*')):
            if path.is_file(): archive.add(path,arcname=str(Path('method_b_code')/path.name),recursive=False)
    print('HANDOFF_ARCHIVE:',target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['install-solver','banks','prepare','smoke','train','evaluate','summary','estimate','export','engine'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--setting',choices=['mixed','transfer'],default='mixed')
    p.add_argument('--budget',type=float,choices=[0.5,1,2,4],default=1)
    p.add_argument('--parent-seeds',type=int,nargs='+',default=[17,123])
    p.add_argument('--training-seeds',type=int,nargs='+',default=[42])
    p.add_argument('--action',choices=['verify','smoke','train','evaluate'])
    p.add_argument('--model',choices=['0.5B','1.5B'])
    p.add_argument('--parent-seed',type=int)
    p.add_argument('--training-seed',type=int,default=42)
    a=p.parse_args();a.root=a.root.resolve()
    if not set(a.parent_seeds)<={17,123} or not set(a.training_seeds)<={42,137,2026}:
        raise ValueError('Seeds must come from the declared parent/continuation lists')
    if a.stage=='install-solver':install_solver(a.root)
    elif a.stage=='banks':build_banks(a.root)
    elif a.stage=='prepare':prepare(a.root,a.setting,a.budget,a.parent_seeds)
    elif a.stage=='engine':engine(a)
    elif a.stage in ('smoke','train','evaluate'):gpu_stage(a.root,a.setting,a.budget,a.stage,a.parent_seeds,a.training_seeds)
    elif a.stage=='summary':summarize(a.root)
    elif a.stage=='estimate':estimate(a.root,a.setting,a.budget)
    elif a.stage=='export':export(a.root)


if __name__=='__main__':enter(main)

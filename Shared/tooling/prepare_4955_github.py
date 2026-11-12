#!/usr/bin/env python3
"""Prepare a small screened B/C0/C1 export in a separate clean checkout.

Standard library only. Reads experiment sources; writes new export files only.
Does not run jobs, stage files, commit, push, alter source code or remove files.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
BRANCH = 'Method-B,-C0,-C1-Latest-Files'
REPOSITORY = 'jahnvisethjs/infj-llm-distillation-improvement'
NATIVE_WORK = Path('/scratch/sshind38/capstone_c0_c1_work')
NATIVE_B = Path('/scratch/sshind38/capstone_approach_b')
BANK_SHA = 'ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f'
MAX_FILE = 5 * 1024 * 1024
MAX_TOTAL = 60 * 1024 * 1024
SOURCE_SUFFIXES = {'.py', '.sh', '.sbatch', '.json', '.md', '.toml', '.yaml', '.yml', '.rst'}
SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', 'hf_cache', 'node_modules',
             'outputs', 'data', 'datasets', 'parents', 'checkpoints', 'logs', 'smokes', 'runs'}
SECRETS = [re.compile(p) for p in [r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{30,}\b', r'\bhf_[A-Za-z0-9]{25,}\b',
    r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def csv_bytes(records):
    stream = io.StringIO()
    if records:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in record.items()})
    return stream.getvalue().encode()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def validate_checkout(repo, work, b_root):
    for source in [work, b_root]:
        if repo == source or repo.is_relative_to(source):
            raise ValueError('Use the separate publishing checkout, outside the experiment roots')
    if Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() != repo:
        raise ValueError('--repo must be the root of the separate checkout')
    if git(repo, 'branch', '--show-current') != BRANCH:
        raise ValueError('Wrong branch; expected ' + BRANCH)
    remote = git(repo, 'remote', 'get-url', 'origin')
    if remote not in {f'https://github.com/{REPOSITORY}.git', f'https://github.com/{REPOSITORY}',
                      f'git@github.com:{REPOSITORY}.git'}:
        raise ValueError('Wrong origin, or origin contains embedded credentials')
    if git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Publishing checkout has uncommitted files. Review/stage/commit the existing export instead of recreating it.')
    for name in ['B', 'C0', 'C1', 'Shared', 'Comparison', 'EXPORT_MANIFEST.json', 'SCREENED_4955_README.md', '.gitignore']:
        if (repo / name).exists() or (repo / name).is_symlink():
            raise ValueError('Existing destination preserved: ' + str(repo / name))


def open_zip(data, label):
    archive = zipfile.ZipFile(io.BytesIO(data))
    if archive.testzip() is not None:
        raise ValueError('Damaged ZIP: ' + label)
    return archive


def map_source(native, work, b_root):
    native = Path(native)
    for prefix, source in [(NATIVE_WORK, work), (NATIVE_B, b_root)]:
        if native.is_relative_to(prefix):
            return source / native.relative_to(prefix)
    raise ValueError('Unexpected external run path: ' + str(native))


class Export:
    def __init__(self):
        self.files = {}
        self.entries = []
        self.skipped = []

    def add(self, relative, data, source, transformation=None):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid export destination')
        name = relative.as_posix()
        if len(data) > MAX_FILE:
            raise ValueError('Important file exceeds 5 MiB; preserved on Sol, export stopped: ' + source)
        if name in self.files:
            if self.files[name] != data:
                raise ValueError('Conflicting export members: ' + name)
            return
        if relative.suffix in SOURCE_SUFFIXES or relative.suffix in {'.csv', '.txt', '.ipynb'}:
            text = data.decode('utf-8-sig')
            if any(pattern.search(text) for pattern in SECRETS):
                raise ValueError('Credential-like content detected; file not exported: ' + source)
        self.files[name] = data
        self.entries.append({'path': name, 'bytes': len(data), 'sha256': sha(data),
                             'source': source, 'transformation': transformation})

    def copy(self, source, destination):
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError('Missing required regular source file: ' + str(source))
        self.add(destination, source.read_bytes(), str(source))

    def code_tree(self, source, destination):
        source = Path(source)
        if not source.is_dir() or source.is_symlink():
            raise ValueError('Missing required source directory: ' + str(source))
        pins = read(source / 'PACKAGE_PINS.json') if (source / 'PACKAGE_PINS.json').is_file() else {}
        for name, wanted in pins.items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts or sha((source / rel).read_bytes()) != wanted:
                raise ValueError('Frozen source package changed: ' + str(source / rel))
        for directory, directories, files in os.walk(source, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in SKIP_DIRS
                and not d.startswith('.') and not d.startswith('venv')
                and not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                if (path.is_symlink() or name.startswith('.') or name in {'AGENTS.md', 'RUN_LOCATION.json'}):
                    continue
                if not (path.suffix in SOURCE_SUFFIXES or path.relative_to(source).as_posix() in pins or name in {'LICENSE', 'COPYING', 'NOTICE'}
                        or (path.suffix == '.txt' and name.lower().startswith(('requirements', 'readme', 'license', 'notice')))):
                    continue
                if path.stat().st_size > MAX_FILE:
                    self.skipped.append({'source': str(path), 'reason': 'large non-runtime code/document file'})
                    continue
                self.copy(path, Path(destination) / path.relative_to(source))
        for name in pins:
            if (Path(destination) / name).as_posix() not in self.files:
                raise ValueError('Pinned source dependency was excluded: ' + name)

    def notebook(self, candidates, destination, required_text):
        for source in candidates:
            if not source.is_file() or source.is_symlink():
                continue
            if source.stat().st_size > 64 * 1024 * 1024:
                continue
            notebook = read(source)
            content = '\n'.join(c.get('source', '') if isinstance(c.get('source', ''), str)
                                else ''.join(c['source']) for c in notebook.get('cells', []))
            if required_text not in content:
                continue
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
                cell.pop('attachments', None)
            notebook.setdefault('metadata', {}).pop('widgets', None)
            self.add(destination, encoded(notebook), str(source),
                     'Outputs, execution counts and widget state stripped from the copy; code unchanged')
            return
        raise ValueError('Current notebook not found among: ' + ', '.join(str(p) for p in candidates))


def prepare(args):
    work, b_root, home = Path(args.work).resolve(), Path(args.b_root).resolve(), Path(args.home).resolve()
    repo, kit = Path(args.repo).resolve(), Path(args.kit).resolve()
    validate_checkout(repo, work, b_root)
    kit_manifest = read(kit / 'KIT_FILES.json')
    for name, wanted in kit_manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha((kit / path).read_bytes()) != wanted:
            raise ValueError('Changed or damaged publisher kit: ' + name)
    audited = read(kit / 'assets/verified_matched_results.json')
    if audited['failure_count'] != 0:
        raise ValueError('Source result audit is not clean')
    export = Export()

    # Read only the current screened bank; do not publish its full question/solution text.
    bank_path = work / 'outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17/35f8a5ecbf57cd37/candidate_bank.jsonl'
    if bank_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('Unexpected bank size')
    bank_data = bank_path.read_bytes()
    if sha(bank_data) != BANK_SHA:
        raise ValueError('Wrong eligible bank; original or altered bank is not accepted')
    bank = json_rows(bank_data)
    if len(bank) != 4955 or len({r['row_id'] for r in bank}) != 4955:
        raise ValueError('Expected 4,955 unique candidate IDs')
    bank_ids = {r['row_id']: r for r in bank}
    source_mix = dict(Counter(r['dataset_id'] for r in bank))
    if source_mix != {'orca': 3313, 'numina': 1642}:
        raise ValueError('Screened source mix differs')
    export.add('Shared/provenance/SCREENED_BANK.json', encoded({'rows': 4955, 'source_rows': source_mix,
        'sha256': BANK_SHA, 'native_path': str(bank_path), 'raw_data_in_git': False,
        'scope': 'Same eligible bank for B, C0 and C1; training selections contain 1000 rows each.'}), str(bank_path), 'Bank receipt; raw training text omitted')
    export.add('Shared/provenance/candidate_ids.csv', csv_bytes([
        {k: r[k] for k in ['row_id', 'problem_group_id', 'dataset_id', 'length_bin', 'supervised_tokens']}
        for r in bank]), str(bank_path), 'ID/token metadata only; questions and solutions omitted')
    export.add('Shared/provenance/PARENT_CHECKPOINTS.json', encoded([
        r for r in audited['targets'] if r['method'] == 'parent']), 'Verified common R1000 parent records')
    export.add('Shared/provenance/EVALUATION_CONTRACT.json', encoded(audited['generation_contract']),
               'Verified common development evaluation contract')

    archives = {}
    for key in ['B', 'C', 'Ctrain']:
        source = home / audited['source_files'][key]
        raw = source.read_bytes()
        if sha(raw) != audited['source_sha256'][key]:
            raise ValueError('Handoff differs from verified latest result: ' + str(source))
        archives[key] = open_zip(raw, str(source))
    b_marker = json.loads(archives['B'].read('EVALUATION_COMPLETE.json'))
    b_eval = open_zip(archives['B'].read(b_marker['results_zip']), 'B evaluation')
    c_eval = open_zip(archives['C'].read('C0_C1_Development_Evaluation_Results.zip'), 'C evaluation')
    c_training = open_zip(archives['Ctrain'].read('C0_C1_training_receipts.zip'), 'C training')
    b_plan = json.loads(b_eval.read('PLAN.json'))
    c_plan = json.loads(c_eval.read('PLAN.json'))
    question_ids = {}
    for bench, wanted in audited['question_sha256'].items():
        data = c_eval.read('questions/' + bench + '.jsonl')
        if sha(data) != wanted:
            raise ValueError('Wrong frozen development questions')
        questions = json_rows(data)
        if len(questions) != 500 or any(q.get('split') == 'test' for q in questions):
            raise ValueError('Unexpected development-set scope')
        question_ids[bench] = {q['problem_id'] for q in questions}
        export.add('Shared/provenance/' + bench + '.json', encoded({'rows': 500,
            'sha256': wanted, 'problem_ids': sorted(question_ids[bench]),
            'question_text_published': False, 'reserved_final_test': False}),
            'C evaluation handoff/questions/' + bench + '.jsonl', 'IDs and native question-file digest only')

    targets = [(t, b_eval) for t in b_plan['targets']] + [
        (t, c_eval) for t in c_plan['targets'] if t['method'] in {'C0', 'C1'}]
    if len(targets) != 36 or Counter(t['method'] for t, _ in targets) != {'B_screened': 12, 'C0': 12, 'C1': 12}:
        raise ValueError('Expected 12 screened B + 12 C0 + 12 C1 targets')
    expected_matrix = {(s, p, t) for s in ['0.5B', '1.5B'] for p in [17, 123] for t in [42, 137, 2026]}
    result_count = 0
    for method in ['B_screened', 'C0', 'C1']:
        if {(t['model_size'], t['parent_seed'], t['continuation_seed']) for t, _ in targets if t['method'] == method} != expected_matrix:
            raise ValueError('Incomplete configuration matrix: ' + method)
    for target, source_zip in targets:
        method = 'B' if target['method'] == 'B_screened' else target['method']
        if target['bank_sha256'] != BANK_SHA:
            raise ValueError('Historical/different-pool target rejected: ' + target['key'])
        folder = f"{target['model_size']}_parent{target['parent_seed']}_seed{target['continuation_seed']}"
        export.add(f'{method}/provenance/{folder}/evaluation_target.json', encoded(target),
                   'Native evaluation PLAN.json target', 'Exact target record; original evidence not rewritten')
        native_directory = str(Path(target['adapter']).parent.parent)
        directory = map_source(native_directory, work, b_root)
        receipt_data = (directory / 'completion.json').read_bytes()
        receipt = json.loads(receipt_data)
        config_data = (directory / 'run_config.json').read_bytes()
        summary_data = (directory / 'training_summary.json').read_bytes()
        config, summary = json.loads(config_data), json.loads(summary_data)
        if (not receipt.get('passed') or receipt.get('smoke_only') or not summary.get('complete')
                or receipt['adapter_sha256'] != target['checkpoint_sha256']
                or receipt['bank_sha256'] != BANK_SHA
                or config['manifest_sha256'] != target['manifest_sha256']
                or sha(config_data) != receipt['run_config_sha256']
                or sha(summary_data) != receipt['summary_sha256']):
            raise ValueError('Training receipt/manifest/checkpoint differs: ' + target['key'])
        if target.get('training_receipt_sha256') and sha(receipt_data) != target['training_receipt_sha256']:
            raise ValueError('Training receipt changed after evaluation')
        for path, wanted in target.get('approved_files', {}).items():
            if sha(map_source(path, work, b_root).read_bytes()) != wanted:
                raise ValueError('Changed C training metadata: ' + path)
        selected = set(summary['seen_ids'])
        if len(selected) != 1000 or not selected <= bank_ids.keys() or summary['processed_tokens'] != sum(bank_ids[i]['supervised_tokens'] for i in selected):
            raise ValueError('Wrong selected IDs or label budget: ' + target['key'])
        if Counter((bank_ids[i]['dataset_id'], bank_ids[i]['length_bin']) for i in selected) != Counter({(s, i): 50 for s in ['orca', 'numina'] for i in range(10)}):
            raise ValueError('Wrong source/length quotas')
        training_folder = f'{method}/training/{folder}'
        for name, data in [('completion.json', receipt_data), ('run_config.json', config_data), ('training_summary.json', summary_data)]:
            export.add(training_folder + '/' + name, data, native_directory + '/' + name)
        log = directory / 'training_log.jsonl'
        if log.is_file():
            raw = log.read_bytes()
            export.add(training_folder + '/training_log.jsonl.gz', gzip.compress(raw, mtime=0), str(log), 'Losslessly gzipped; original hash ' + sha(raw))
        selection_folder = f"{method}/selections/{target['model_size']}_parent{target['parent_seed']}"
        manifest_rows = [{k: bank_ids[i][k] for k in ['row_id', 'problem_group_id', 'dataset_id', 'length_bin', 'supervised_tokens']} for i in sorted(selected)]
        export.add(selection_folder + '/selected_ids.csv', csv_bytes(manifest_rows), native_directory + '/training_summary.json', 'One selection per profile; ID/token metadata only')
        export.add(selection_folder + '/MANIFEST.json', encoded({'rows': 1000, 'bank_rows': 4955,
            'bank_sha256': BANK_SHA, 'native_train_jsonl_sha256': target['manifest_sha256'],
            'actual_supervised_tokens': summary['processed_tokens'], 'target_supervised_tokens': 318845,
            'strata': 20, 'rows_per_stratum': 50, 'continuation_seeds': [42, 137, 2026],
            'raw_training_text_published': False}), native_directory + '/run_config.json', 'Export index; native manifest fingerprint preserved')
        for bench in ['dev_gsm', 'dev_math']:
            stem=f"results/{target['key']}/{bench}/complete/"
            result=json.loads(source_zip.read(stem + 'completion.json'))
            if not result['complete'] or result['target_sha256'] != canonical(target):
                raise ValueError('Mismatched evaluation target')
            for name, field in [('predictions.jsonl', 'predictions_sha256'), ('predictions.config.json', 'config_sha256'), ('metrics.json', 'metrics_sha256')]:
                if sha(source_zip.read(stem+name)) != result[field]:
                    raise ValueError('Evaluation result hash differs')
            raw=source_zip.read(stem+'predictions.jsonl');pred=json_rows(raw);metrics=json.loads(source_zip.read(stem+'metrics.json'))
            if (len(pred) != 500 or len({p['problem_id'] for p in pred}) != 500
                    or {p['problem_id'] for p in pred} != question_ids[bench]
                    or any(p['checkpoint_sha256'] != target['checkpoint_sha256'] or p['train_manifest_sha256'] != target['manifest_sha256'] for p in pred)
                    or metrics['correct'] != sum(p['correct'] for p in pred)):
                raise ValueError('Prediction coverage or metric mismatch')
            dest=f'{method}/evaluation/{folder}/{bench}/'
            for name in ['completion.json', 'predictions.config.json', 'metrics.json']:
                export.add(dest+name,source_zip.read(stem+name),'Verified handoff/'+stem+name)
            export.add(dest+'predictions.jsonl.gz',gzip.compress(raw,mtime=0),'Verified handoff/'+stem+'predictions.jsonl','Losslessly gzipped; native uncompressed hash ' + sha(raw))
            result_count += 1

    # Copy frozen source packages intact, without runtime outputs, caches or datasets.
    for name in ['B_Screened_4955_Controls_v1', 'B_Screened_4955_Development_Evaluation_v1']:
        export.code_tree(work/name,'B/code/'+name)
    for method in ['C0', 'C1']:
        for name in ['C0_C1_Overnight_Training_20261110', 'C0_C1_Development_Evaluation_20261110']:
            export.code_tree(work/name,method+'/code/'+name)
    for name in ['C0_Screened_05B_Seed17_20261110', 'C0_C1_Independent_Screen_Proposal_20261110']:
        export.code_tree(work/name,'Shared/code/'+name)
    runtime=b_root/'outputs/B_method_B_document_v3/mixed/budget_1x/runtime'
    export.code_tree(runtime,'Shared/code/B_runtime')
    for name in ['capstone_person4', 'capstone_person4_approachC1']:
        export.code_tree(work/'repo'/name,'Shared/code/selection_repository/'+name)
    export.notebook([home/'B new Approach/B_4955_Full_Training_GSM_MATH_Evaluation.ipynb', home/'B_4955_Full_Training_GSM_MATH_Evaluation.ipynb'],
        'B/notebooks/B_4955_Full_Training_GSM_MATH_Evaluation.ipynb','evaluate_b.py')
    for method in ['C0', 'C1']:
        export.notebook([work/'C0_C1_Overnight_Training_Only.ipynb', home/'C0_C1_Overnight_Training_Only.ipynb'],method+'/notebooks/C0_C1_Overnight_Training_Only.ipynb','nightly.py')
        export.notebook([work/'C0_C1_Development_Evaluation_Overnight.ipynb', home/'C0_C1_Development_Evaluation_Overnight.ipynb'],method+'/notebooks/C0_C1_Development_Evaluation_Overnight.ipynb','evaluate_remaining.py')
    for name in c_training.namelist():
        if name.endswith('/selection/C1/diagnostics.json'):
            profile=name.split('/')[1]
            export.add('C1/diagnostics/'+profile+'.json',c_training.read(name),'C training handoff/'+name)

    for method in ['B', 'C0', 'C1']:
        native_method='B_screened' if method=='B' else method
        export.add(method+'/RESULTS.csv',csv_bytes([r for r in audited['results'] if r['method']==native_method]),'Independent verified results','Current screened method only')
        export.add(method+'/PER_SEED_SCORES.csv',csv_bytes([r for r in audited['per_seed'] if r['method']==native_method]),'Independent verified results','Current screened method only')
        export.add(method+'/README.md',(kit/'method_readmes'/f'{method}.md').read_bytes(),'Publisher kit method README')
    for path in sorted((kit/'comparison').rglob('*')):
        if path.is_file():export.copy(path,'Comparison/'+str(path.relative_to(kit/'comparison')))
    for name in ['SCREENED_4955_README.md', '.gitignore']:
        export.add(name,(kit/name).read_bytes(),'Publisher kit')
    export.copy(Path(__file__).resolve(),'Shared/tooling/prepare_4955_github.py')
    if result_count != 72:
        raise ValueError('Expected exactly 72 screened development files')
    total=sum(len(data) for data in export.files.values())
    if total > MAX_TOTAL:
        raise ValueError('Export exceeds the 60 MiB total cap; no destination files written')
    manifest={'version':'screened-4955-B-C0-C1-code-results-export-v1','branch':BRANCH,
        'candidate_bank_rows':4955,'bank_sha256':BANK_SHA,'selected_rows_per_training_run':1000,
        'trained_models':{'B':12,'C0':12,'C1':12},'development_result_files':72,
        'question_outcomes':36000,'question_file_sha256':audited['question_sha256'],
        'source_archive_sha256':{k:audited['source_sha256'][k] for k in ['B','C','Ctrain']},
        'legacy_B_results_in_export':False,'reserved_final_tests_in_export':False,
        'files':export.entries,'files_count_excluding_this_manifest':len(export.files),
        'bytes_excluding_this_manifest':total,'skipped':export.skipped,
        'external_artifacts':'Original bank text, model/adapter/optimizer weights, base-model cache and venvs remain on Sol; this is a code/results handoff, not a checkpoint backup.'}
    export.add('EXPORT_MANIFEST.json',encoded(manifest),'Generated export receipt')
    # All science/size/checkpoint checks finish before writing the new checkout files.
    for name,data in sorted(export.files.items()):
        path=repo/name
        if not path.resolve().is_relative_to(repo) or path.exists() or path.is_symlink():
            raise ValueError('Destination already exists or escapes checkout: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'status':'EXPORT READY','destination':str(repo),'branch':BRANCH,
        'files':len(export.files),'total_MiB':round(sum(len(d) for d in export.files.values())/1024/1024,2),
        'screened_models':36,'development_files':72,'candidate_rows':4955,
        'source_files_modified':False,'jobs_submitted':False,'committed_or_pushed':False},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    p.add_argument('--kit',default=str(Path(__file__).resolve().parent))
    p.add_argument('--work',default=str(NATIVE_WORK));p.add_argument('--b-root',default=str(NATIVE_B))
    p.add_argument('--home',default='/home/sshind38')
    prepare(p.parse_args())


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,KeyError,subprocess.CalledProcessError,zipfile.BadZipFile) as error:
        print('EXPORT STOPPED: '+str(error),file=sys.stderr);raise SystemExit(1)

"""Import Person 3's question-only frozen manifests; never builds evaluation data."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if 'validate_problem_manifest' not in globals():
    from capstone_person1.pipeline_controls import (
        file_sha256, normalize_problem_for_matching, sha256_text, validate_problem_manifest,
    )

BENCHMARK_MANIFEST_VERSION = 'person3-question-manifest-v1'
EVALUATION_ROLES = {'development', 'fast_development', 'final_test'}
PROVENANCE_FIELDS = ['benchmark_family', 'role', 'dataset_id', 'revision', 'split', 'selection_policy']


def _validate_confirmation(metadata):
    if not str(metadata.get('confirmed_by', '')).strip() or not metadata.get('confirmed_utc'):
        raise ValueError('Person 3 confirmation identity/time is required')
    stamp = datetime.fromisoformat(metadata['confirmed_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Confirmation requires a timezone-aware, nonfuture timestamp')


def _legacy_reserved_sets(metadata):
    """Explicitly map the known historical manifest; never infer roles from filenames."""
    sets = {}
    for set_id, key, family in [('gsm8k_test', 'gsm8k', 'gsm8k'), ('math500_test', 'math500', 'math')]:
        source = metadata['datasets'][key]
        if source['split'] != 'test': raise ValueError('Historical reserved source must use its test split')
        sets[set_id] = dict(source, benchmark_family=family, role='final_test',
                            selection_policy={'method': 'all_official_test_questions'},
                            origin='historical_reserved_question_manifest')
    return sets


def load_question_manifest(path, metadata_path, *, legacy_reserved=False):
    path, metadata_path = Path(path), Path(metadata_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_problem_manifest(frame, 'Person 3 question manifest')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('manifest_sha256') != file_sha256(path):
        raise ValueError('Question manifest content differs from frozen metadata hash')
    if metadata.get('set_counts') != frame['set_id'].value_counts().to_dict():
        raise ValueError('Question manifest set counts differ from frozen metadata')
    legacy = legacy_reserved and 'sets' not in metadata
    if legacy:
        sets = _legacy_reserved_sets(metadata)
        confirmed = False
    else:
        if metadata.get('schema_version') != BENCHMARK_MANIFEST_VERSION:
            raise ValueError('Question metadata requires the explicit Person 3 manifest schema')
        _validate_confirmation(metadata)
        sets, confirmed = metadata.get('sets', {}), True
    if set(sets) != set(frame['set_id']):
        raise ValueError('Every set requires explicit role/source metadata; unknown sets are invalid')
    parts = []
    for set_id, source in sets.items():
        if any(key not in source or source[key] in ('', None) for key in PROVENANCE_FIELDS):
            raise ValueError('Incomplete source/role/selection provenance for ' + set_id)
        if source['role'] not in EVALUATION_ROLES or source['benchmark_family'] not in {'gsm8k', 'math'}:
            raise ValueError('Unsupported evaluation role/family for this experiment')
        if not re.fullmatch(r'[0-9a-f]{40}|sha256:[0-9a-f]{64}', source['revision']):
            raise ValueError('Evaluation source requires an immutable revision or source-file digest')
        selection = source['selection_policy']
        if not isinstance(selection, dict) or not str(selection.get('method', '')).strip():
            raise ValueError('Selection policy requires a documented method')
        if re.search(r'random|sample', str(selection['method']), re.I) and type(selection.get('seed')) is not int:
            raise ValueError('Sampled evaluation questions require an explicit integer seed')
        if source['role'] != 'final_test' and (source['dataset_id'] == 'HuggingFaceH4/MATH-500' or
                (source['dataset_id'] == 'openai/gsm8k' and source['split'] == 'test')):
            raise ValueError('Official reserved final tests cannot supply development questions')
        part = frame.loc[frame['set_id'].eq(set_id), ['set_id', 'problem_id', 'problem']].copy()
        for key in ['benchmark_family', 'role', 'dataset_id', 'revision', 'split']:
            part[key] = source[key]
        part['configuration'] = source.get('configuration', '')
        part['selection_policy_json'] = json.dumps(selection, sort_keys=True)
        part['manifest_origin'] = source.get('origin', 'person3_confirmed_manifest')
        part['parent_set_id'] = source.get('parent_set_id', '')
        part['role_confirmed_by_person3'] = confirmed
        part['source_manifest_sha256'] = metadata['manifest_sha256']
        parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    output['problem_hash'] = output['problem'].map(normalize_problem_for_matching).map(sha256_text)
    for set_id, rows in output.groupby('set_id'):
        if rows['problem_hash'].duplicated().any():
            raise ValueError('Duplicate normalized questions inside ' + set_id)
    return output, metadata


def _validate_roles(frame):
    sets = {set_id: rows for set_id, rows in frame.groupby('set_id')}
    dev = frame[frame['role'].isin(['development', 'fast_development'])]
    final = frame[frame['role'].eq('final_test')]
    if set(dev['problem_hash']) & set(final['problem_hash']):
        raise ValueError('Development questions intersect reserved final tests')
    # Source IDs additionally catch changed wording under the same frozen source identity.
    source_keys = ['dataset_id', 'revision', 'split', 'problem_id']
    if set(map(tuple, dev[source_keys].values)) & set(map(tuple, final[source_keys].values)):
        raise ValueError('Development/final source question identities intersect')
    for set_id, rows in sets.items():
        if rows.iloc[0]['role'] != 'fast_development': continue
        parent = rows.iloc[0]['parent_set_id']
        if not parent or parent not in sets or sets[parent].iloc[0]['role'] != 'development':
            raise ValueError('Fast-development set requires a supplied full-development parent')
        full = sets[parent]
        for field in ['benchmark_family', 'dataset_id', 'revision', 'configuration', 'split']:
            if rows.iloc[0][field] != full.iloc[0][field]:
                raise ValueError('Fast/full source provenance differs')
        full_ids = dict(zip(full['problem_id'], full['problem_hash']))
        if any(full_ids.get(row.problem_id) != row.problem_hash for row in rows.itertuples()):
            raise ValueError('Fast-development IDs/text must be an exact subset of its full parent')


def load_protection_inputs(reserved_path, reserved_metadata_path, development_path=None, development_metadata_path=None):
    reserved, reserved_metadata = load_question_manifest(reserved_path, reserved_metadata_path, legacy_reserved=True)
    if not reserved['role'].eq('final_test').all():
        raise ValueError('Reserved manifest may contain only explicit final-test sets')
    if set(reserved['benchmark_family']) != {'gsm8k', 'math'}:
        raise ValueError('This experiment requires both configured final-test families')
    inputs = {'reserved_test_sha256': file_sha256(reserved_path),
              'reserved_metadata_sha256': file_sha256(reserved_metadata_path)}
    development, parts = None, [reserved]
    if (development_path is None) != (development_metadata_path is None):
        raise ValueError('Development manifest and metadata must be supplied together')
    if development_path is not None:
        exists = [Path(development_path).exists(), Path(development_metadata_path).exists()]
        if any(exists) and not all(exists):
            raise ValueError('Incomplete development manifest/metadata pair must be fixed')
        if all(exists):
            development, _ = load_question_manifest(development_path, development_metadata_path)
            if development['role'].eq('final_test').any():
                raise ValueError('Development input cannot redefine reserved final tests')
            if set(development['set_id']) & set(reserved['set_id']):
                raise ValueError('Development and final tests require distinct set IDs')
            parts.append(development)
            inputs.update(development_sha256=file_sha256(development_path),
                          development_metadata_sha256=file_sha256(development_metadata_path))
    combined = pd.concat(parts, ignore_index=True)
    _validate_roles(combined)
    full = combined[combined['role'].eq('development')]
    fast = combined[combined['role'].eq('fast_development')]
    development_complete = set(full['benchmark_family']) == {'gsm8k', 'math'}
    fast_complete = set(fast['benchmark_family']) == {'gsm8k', 'math'}
    report = {
        'schema_version': BENCHMARK_MANIFEST_VERSION, 'question_manifests_validated': True,
        'inputs': inputs, 'set_counts': combined['set_id'].value_counts().to_dict(),
        'role_counts': combined['role'].value_counts().to_dict(),
        'protected_question_entries': len(combined), 'unique_protected_question_texts': combined['problem_hash'].nunique(),
        'person3_development_coverage': 'complete' if development_complete else 'pending',
        'person3_fast_development_coverage': 'complete' if fast_complete else 'pending',
        'person3_final_test_role_confirmation': 'confirmed' if reserved['role_confirmed_by_person3'].all() else 'pending',
        'development_protected': development_complete,
        'fast_subsets_validated': bool(len(fast)),
        'overlap_checks_status': 'pending_pipeline_execution',
        'person1_data_work_may_continue': True,
        'evaluation_data_created_by_person1': False,
        'limitations': ['This validates frozen question inputs; it does not run generation/scoring or certify near-match recall.',
                        'Fast development entries are nested duplicates of full development; do not add their counts as unique questions.'],
    }
    return {'reserved': reserved, 'development': development, 'questions': combined,
            'inputs': inputs, 'development_protected': development_complete, 'scope': report}


def export_protection_inputs(bundle, processed_dir, report_dir):
    processed_dir, report_dir = Path(processed_dir), Path(report_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / 'protected_questions.csv'
    csv_text = bundle['questions'].to_csv(index=False, lineterminator='\n')
    if path.exists() and path.read_text(encoding='utf-8') != csv_text:
        raise ValueError('Protected-question export changed; start a new immutable run')
    path.write_text(csv_text, encoding='utf-8', newline='')
    report = dict(bundle['scope'], protected_questions_sha256=file_sha256(path))
    (report_dir / 'evaluation_protection_scope.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report

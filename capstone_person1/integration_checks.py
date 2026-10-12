"""Trainer evidence checks. Data readiness does not require a trained parent."""
import argparse
import json
from pathlib import Path

import pandas as pd

from capstone_person1.pipeline_controls import file_sha256, reconcile_training_tokens, validate_parent_evidence
from capstone_person1.snapshot_stream import atomic_json


def read_records(path):
    return pd.DataFrame(json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip())


def readiness(*, full_scope, data_checks, reviews_complete, trainer_verified=False, evaluation_frozen=False, parent_bound=False):
    data_ready = full_scope and all(value=='passed' for value in data_checks.values()) and reviews_complete
    main_ready = data_ready and trainer_verified and evaluation_frozen
    continuation_ready = main_ready and parent_bound
    state = 'CONTINUATION_READY' if continuation_ready else 'MAIN_RUN_READY' if main_ready else 'DATA_READY_FOR_TRAINER' if data_ready else 'DIAGNOSTIC_ONLY'
    return dict(status=state,data_ready_for_trainer=data_ready,main_run_ready=main_ready,continuation_ready=continuation_ready,
        data_checks=data_checks,review_status='passed' if reviews_complete else 'pending',full_source_scope='passed' if full_scope else 'pending',
        person2_label_verification='passed' if trainer_verified else 'pending',person3_frozen_evaluation='passed' if evaluation_frozen else 'pending',
        actual_parent_binding='passed' if parent_bound else 'pending')


def verify_person2_tokens(release, scope, actual_path, metadata_path):
    release = Path(release)
    if scope not in {'debug_32','train_n1000'}: raise ValueError('Verification scope must be debug_32 or train_n1000')
    manifest = release/'manifests'/(scope+'.jsonl')
    expected = read_records(manifest)
    actual = pd.read_csv(actual_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    required = ['trainer_version','contract_version','manifest_sha256','prompt_mask_verified','eos_label_verified','padding_by_position_verified','confirmed_by','confirmed_utc']
    if any(not metadata.get(key) for key in required): raise ValueError('Trainer verification evidence incomplete')
    if metadata['manifest_sha256']!=file_sha256(manifest): raise ValueError('Trainer evidence refers to a different manifest')
    from capstone_person1.person1_data_contract import CONTRACT_VERSION
    if metadata['contract_version']!=CONTRACT_VERSION: raise ValueError('Trainer contract version differs')
    for key in ['prompt_mask_verified','eos_label_verified','padding_by_position_verified']:
        if metadata[key] is not True: raise ValueError('Actual masking/EOS/padding handling not verified')
    result = reconcile_training_tokens(expected,actual)
    report = dict(status='passed',scope=scope,rows=len(result),actual_supervised_tokens=int(result['actual_supervised_tokens'].sum()),
        manifest_sha256=file_sha256(manifest),actual_report_sha256=file_sha256(actual_path),trainer_metadata=metadata)
    return report


def bind_parent(release, actual_manifest, seen_path, metadata_path):
    release = Path(release)
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if not metadata.get('actual_run_id'): raise ValueError('Actual parent run ID required')
    planned = release/'manifests/train_n1000.jsonl'
    records = validate_parent_evidence(pd.read_csv(seen_path),actual_manifest,metadata,expected_manifest_sha256=file_sha256(planned))
    selection = read_records(release/'manifests/selection_bank.jsonl')
    if set(record['problem_group_id'] for record in records)&set(selection['problem_group_id']):
        raise ValueError('Selection bank intersects actual parent-seen groups')
    return dict(status='passed',actual_parent_run_id=metadata['actual_run_id'],checkpoint_id=metadata['checkpoint_id'],
        checkpoint_sha256=metadata['checkpoint_sha256'],actual_manifest_sha256=file_sha256(actual_manifest),seen_groups=len(records),
        selection_rows=len(selection),selection_sha256=file_sha256(release/'manifests/selection_bank.jsonl'),freshness='verified_disjoint_actual_parent')


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command',required=True)
    tokens = subs.add_parser('tokens')
    tokens.add_argument('--release',required=True);tokens.add_argument('--scope',required=True)
    tokens.add_argument('--actual',required=True);tokens.add_argument('--metadata',required=True)
    parent = subs.add_parser('parent')
    parent.add_argument('--release',required=True);parent.add_argument('--actual-manifest',required=True)
    parent.add_argument('--seen',required=True);parent.add_argument('--metadata',required=True)
    for item in [tokens,parent]: item.add_argument('--output',required=True,help='Write outside the frozen release')
    args = parser.parse_args()
    output,release = Path(args.output).resolve(),Path(args.release).resolve()
    if output==release or release in output.parents: raise ValueError('Integration results must not mutate the frozen release')
    report = verify_person2_tokens(release,args.scope,args.actual,args.metadata) if args.command=='tokens' else bind_parent(release,args.actual_manifest,args.seen,args.metadata)
    atomic_json(output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

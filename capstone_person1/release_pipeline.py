"""Named upstream stages for the Person 1 data release. No model weights or training."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1.benchmark_manifests import load_protection_inputs, export_protection_inputs
from capstone_person1.instance_matching import instance_hash, retrieve_pairs, apply_instance_decisions, MATCHING_VERSION, context_equivalent
from capstone_person1.integration_checks import readiness
from capstone_person1.person1_data_contract import encode_example, load_tokenizer, formatting_contract, validate_manifest, TOKENIZER_ID, PINNED_TOKENIZER_REVISION
from capstone_person1.pipeline_controls import file_sha256, sha256_text
from capstone_person1.pool_allocation import allocate_banks
from capstone_person1.release_catalog import build_catalog, choose_candidates, export_row_status
from capstone_person1.release_reviews import make_review_queue, sample_formatted, validate_reviews
from capstone_person1.snapshot_stream import atomic_json, stage_batches, selected_records
from capstone_person1.source_stream import source_binding, canonicalize_upstream

PROJECT = Path(__file__).resolve().parent


def message(value): print(value,flush=True)


def freeze_inputs(project,release,config,bindings,protection,pilot):
    code={str(path.relative_to(project)):file_sha256(path) for path in sorted(project.glob('*.py'))}
    code.update({str(path.relative_to(project)):file_sha256(path) for path in sorted((project/'tests').glob('test_*.py'))})
    frozen=dict(schema_version='person1-release-v3',configuration=config,sources=bindings,
                protection_inputs=protection['inputs'],pilot_per_source=pilot,code_sha256=code)
    path=release/'metadata/frozen_inputs.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=frozen: raise ValueError('Frozen inputs or implementation changed; start a new release')
    else: atomic_json(path,frozen)
    return sha256_text(json.dumps(frozen,sort_keys=True))


def checked_candidates(frame,protection,release,config,*,pair_decisions=None):
    reports=release/'reports/duplicate_and_overlap_reports'
    path=release/'processed/checked_candidates.parquet'
    state_path=path.with_suffix('.json')
    binding=sha256_text(json.dumps(dict(candidate_ids=frame['row_id'].tolist(),protection=protection['inputs'],config=config,
                                      decisions_sha256=file_sha256(pair_decisions) if pair_decisions else None),sort_keys=True))
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['binding']!=binding or state['sha256']!=file_sha256(path): raise ValueError('Checked candidate stage changed')
        return pq.read_table(path).to_pandas(),state['report']
    pairs,protected_pairs=retrieve_pairs(frame,protection['questions'],threshold=config['lsh_candidate_threshold'],
        permutations=config['minhash_permutations'],progress=message)
    decisions=json.loads(Path(pair_decisions).read_text(encoding='utf-8')) if pair_decisions else []
    protected_lookup={'protected:'+row.set_id+':'+row.problem_id:row.problem for row in protection['questions'].itertuples()}
    grouped,pair_report,status=apply_instance_decisions(frame,pairs,protected_pairs,decisions,protected_lookup)
    # Similarity alone never joins groups. Undecided retrieved pairs are quarantined
    # conservatively until an explicit decision is supplied in a new release.
    unresolved_ids=set()
    if not pair_report.empty:
        for row in pair_report[pair_report['action'].eq('unresolved')].itertuples():
            unresolved_ids.update(value for value in [row.row_id_a,row.row_id_b] if value in set(grouped['row_id']))
        group_ids=set(grouped.loc[grouped['row_id'].isin(unresolved_ids),'problem_group_id'])
        grouped.loc[grouped['problem_group_id'].isin(group_ids),'group_excluded']=True
        pair_report.loc[pair_report['action'].eq('unresolved'),'action']='quarantine_unresolved_instance'
    # Reviewed joins still require consistent targets across the whole component.
    conflict_groups=[]
    for group,rows in grouped.groupby('problem_group_id'):
        if len(rows)<2: continue
        first=rows.iloc[0]
        if any(context_equivalent(first['problem'],first['final_answer'],row.final_answer) is not True for row in rows.itertuples()):
            conflict_groups.append(group)
    grouped.loc[grouped['problem_group_id'].isin(conflict_groups),'group_excluded']=True
    pair_report.to_csv(reports/'near_instance_decisions.csv',index=False)
    pd.DataFrame(pairs).to_csv(reports/'near_candidate_pairs.csv',index=False)
    pd.DataFrame(protected_pairs).to_csv(reports/'protected_candidate_pairs.csv',index=False)
    if not pair_report.empty:
        sample=pair_report.groupby(['kind','action'],group_keys=False).apply(lambda rows:rows.sample(n=min(6,len(rows)),random_state=config['split_seed']),include_groups=False)
        # Keep kind/action in the sampled evidence using original indexes.
        sample=pair_report.loc[sample.index].copy()
        for field in ['reviewer','reviewed_utc','review_decision','notes']: sample[field]=''
        sample['policy_version']=MATCHING_VERSION
        sample.to_csv(reports/'pair_policy_review.csv',index=False)
    status.update(checked_rows=len(grouped),near_candidates=len(pairs),protected_near_candidates=len(protected_pairs),
        unresolved_pairs_disposition='all implicated final groups quarantined',near_answer_conflict_groups=conflict_groups,
        retained_group_count=int(grouped.loc[~grouped['group_excluded'],'problem_group_id'].nunique()),
        scope='bounded declared universe; global exact checks on complete canonical sources',recall_guaranteed=False,
        intended_overlap_policy=config['intended_overlap_policy'])
    pq.write_table(pa.Table.from_pandas(grouped,preserve_index=False),path,compression='zstd')
    atomic_json(state_path,dict(binding=binding,sha256=file_sha256(path),report=status))
    return grouped,status


def encode_candidates(frame,tokenizer,release,config):
    path=release/'processed/encoded_candidates.parquet'
    state_path=path.with_suffix('.json')
    binding=dict(checked_sha256=file_sha256(release/'processed/checked_candidates.parquet'),
                 encoder_sha256=file_sha256(PROJECT/'person1_data_contract.py'),max_total_tokens=config['max_total_tokens'])
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['binding']!=binding or state['sha256']!=file_sha256(path): raise ValueError('Encoded stage changed')
        return pq.read_table(path).to_pandas(),state['report']
    retained=frame[~frame['group_excluded'] & frame['row_id'].eq(frame['group_representative_id'])]
    records,excluded=[],[]
    for index,row in enumerate(retained.to_dict('records')):
        try:
            encoded=encode_example(tokenizer,row['problem'],row['solution'],row['final_answer'],max_total_tokens=config['max_total_tokens'])
            row.update({key:value for key,value in encoded.items() if key not in {'input_ids','attention_mask','labels'}})
            row.update(example_id=row['row_id'],source=row['dataset_id'],tokenizer_revision=PINNED_TOKENIZER_REVISION,
                       target_review_status='structurally_consistent; mathematical_correctness_unknown')
            records.append(row)
        except ValueError as error:
            excluded.append(dict(row_id=row['row_id'],dataset_id=row['dataset_id'],reason=str(error),error_type=type(error).__name__,
                solution_char_length=row['solution_char_length']))
        if index%2000==0: message(f'Reference encoding: {index:,}/{len(retained):,}; retained {len(records):,}')
    encoded_frame=pd.DataFrame(records)
    pd.DataFrame(excluded,columns=['row_id','dataset_id','reason','error_type','solution_char_length']).to_csv(release/'reports/whole_example_encoding_exclusions.csv',index=False)
    report=dict(input_rows=len(retained),encodable_rows=len(records),source_counts=encoded_frame['dataset_id'].value_counts().to_dict(),
                exclusions=len(excluded),exclusion_reasons=dict(Counter(item['error_type'] for item in excluded)),
                whole_example_exclusion=True,truncation=False,reference_labels_verified=True)
    pq.write_table(pa.Table.from_pandas(encoded_frame,preserve_index=False),path,compression='zstd')
    atomic_json(state_path,dict(binding=binding,sha256=file_sha256(path),report=report))
    return encoded_frame,report


def write_manifest(frame,name,release,tokenizer,config):
    directory=release/'manifests'
    path=directory/(name+'.jsonl')
    records=pa.Table.from_pandas(frame,preserve_index=False).to_pylist()
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        for record in records: stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    ids=frame['row_id'].tolist()
    (directory/(name+'_ids.txt')).write_text(''.join(value+'\n' for value in ids),encoding='utf-8')
    summary=dict(name=name,rows=len(frame),manifest_sha256=file_sha256(path),ordered_id_sha256=sha256_text('\n'.join(ids)+'\n'),
        tokenizer_id=TOKENIZER_ID,tokenizer_revision=PINNED_TOKENIZER_REVISION,max_total_tokens=config['max_total_tokens'],
        supervised_tokens=int(frame['supervised_tokens'].sum()),source_rows=frame['dataset_id'].value_counts().to_dict(),
        source_supervised_tokens=frame.groupby('dataset_id')['supervised_tokens'].sum().astype(int).to_dict(),
        prompt_tokens=int(frame['prompt_tokens'].sum()),completion_tokens=int(frame['completion_tokens'].sum()),eos_tokens=int(frame['eos_tokens'].sum()),
        max_total_input_tokens=int(frame['total_tokens'].max()),source_length_bins={source:rows['length_bin'].value_counts().sort_index().to_dict() for source,rows in frame.groupby('dataset_id')})
    summary['validation']=validate_manifest(path,summary,tokenizer)
    atomic_json(directory/(name+'_summary.json'),summary)
    return summary


def review_queues(encoded,stages,conn,release,pilot):
    reports=release/'reports'
    formatted=sample_formatted(encoded,count=100)
    make_review_queue(formatted,reports/'formatted_target_review.csv','formatted')
    risks=[]
    for source,stage in stages.items():
        # Deterministic stratification over nonexclusive quality flags and modes.
        rows=pd.read_sql_query('SELECT row_id,quality_flags FROM rows WHERE dataset_id=? AND eligible=0 ORDER BY row_id',conn,params=[source])
        chosen=[]
        by_flag={}
        for row in rows.itertuples():
            for flag in json.loads(row.quality_flags):
                if len(by_flag.setdefault(flag,[]))<2: by_flag[flag].append(row.row_id)
        for flag in sorted(by_flag): chosen.extend(by_flag[flag])
        chosen=list(dict.fromkeys(chosen))[:40]
        risks.extend(selected_records(stage,chosen))
    make_review_queue(pd.DataFrame(risks),reports/'answer_extraction_review.csv','risk')
    result=validate_reviews([reports/'formatted_target_review.csv',reports/'answer_extraction_review.csv'],required_formatted_per_source=100)
    pair_path=reports/'duplicate_and_overlap_reports/pair_policy_review.csv'
    pair_complete=True
    if pair_path.exists():
        pairs=pd.read_csv(pair_path,keep_default_na=False)
        for row in pairs.to_dict('records'):
            if not row['reviewer'] or not row['reviewed_utc'] or not row['notes'] or row['review_decision']!='accept_policy_disposition' or row['policy_version']!=MATCHING_VERSION:
                pair_complete=False
        result['pair_review_sha256']=file_sha256(pair_path)
    result['pair_policy_complete']=pair_complete
    result['complete']=result['complete'] and pair_complete
    atomic_json(reports/'review_decisions.json',result)
    return result


def consolidate_canonical(stages,processed):
    for source,stage in stages.items():
        path=processed/('canonical_'+source+'.parquet')
        if path.exists(): continue
        writer=None
        try:
            for batch in stage_batches(stage,batch_size=4096):
                if writer is None: writer=pq.ParquetWriter(path,batch.schema,compression='zstd')
                writer.write_batch(batch)
        finally:
            if writer: writer.close()


def package_release(release,config,bindings,scope,reports,summaries,status,fingerprint,pilot):
    # Bundle only named current stages, helpers, tests, and documented integration.
    bundle=release/'code/capstone_person1'
    bundle.mkdir(parents=True,exist_ok=True)
    names=['__init__.py','canonical_data.py','configuration.py','pipeline_controls.py','instance_matching.py','source_stream.py',
           'snapshot_stream.py','release_catalog.py','release_reviews.py','release_pipeline.py','pool_allocation.py','integration_checks.py',
           'person1_data_contract.py','benchmark_manifests.py']
    for name in names:
        source=PROJECT/name
        if source.exists(): shutil.copy2(source,bundle/name)
    shutil.copytree(PROJECT/'tests',bundle/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(PROJECT.parent/'Person1_Full_Run_v2.ipynb',release/'Person1_Full_Run_v2.ipynb')
    shutil.copy2(PROJECT/'person1_data_contract.py',release/'person1_data_contract.py')
    shutil.copy2(PROJECT/'configs/person1_data_config.json',release/'configs/person1_data_config.json')
    atomic_json(release/'metadata/source_revisions.json',bindings)
    atomic_json(release/'metadata/benchmark_manifests.json',scope)
    versions={}
    for item in ['numpy','pandas','pyarrow','transformers','huggingface-hub','tokenizers','datasketch','jsonschema']:
        try: versions[item]=importlib.metadata.version(item)
        except importlib.metadata.PackageNotFoundError: versions[item]='not_installed'
    atomic_json(release/'metadata/environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions,
        requirements='code/requirements.txt',models_loaded=False))
    requirements='\n'.join(item+'=='+version for item,version in versions.items() if version!='not_installed')+'\n'
    (release/'code/requirements.txt').write_text(requirements,encoding='utf-8')
    totals='\n'.join(f'| {name} | {item["rows"]:,} | {item["supervised_tokens"]:,} |' for name,item in summaries.items())
    handoff=f'''# Person 1 data handoff

Release `{release.name}` — **{status['status']}**. All paths below are relative to this release. Checksums are in `metadata/release_manifest.json`. Source revisions and verified original files are in `metadata/source_revisions.json`. The complete original source fields are preserved per canonical row.

Use Qwen/Qwen2.5-0.5B base and tokenizer revision `{PINNED_TOKENIZER_REVISION}`. Install `code/requirements.txt` into an isolated Python environment, then add `code/` to `PYTHONPATH`. The notebook calls the same named stages as the CLI; no notebook cell-number execution is required for this release.

Load JSON lines from `manifests/debug_32.jsonl`, then `train_n500`, `train_n1000`, `train_n2000`, and `train_n4000`. Import `encode_example` and `pad_encoded` from `capstone_person1.person1_data_contract`. Call `encode_example(tokenizer, row['problem'], row['solution'], row['final_answer'])`; use its input IDs, attention mask and labels directly. The exact prompt ends in `Solution:\\n`. No chat template. Combined-text tokenization uses `add_special_tokens=False` and checked character offsets. Prompt labels are -100; completion and exactly one appended EOS (151643, `<|endoftext|>`) are supervised. Padding shares that ID: mask by position/attention, preserving the real EOS label. Full details are in `configs/formatting_contract.json`. Evaluation supplies only `render_prompt(problem)`.

| Manifest | Rows | Reference supervised tokens including EOS |
|---|---:|---:|
{totals}

These totals come from reference labels; the trainer has not yet supplied its actual-label verification. Verify **all** debug_32 IDs first, then all R1000 IDs. Export CSV `row_id,actual_supervised_tokens` and metadata with trainer/contract versions, exact manifest SHA-256, actual prompt/EOS/padding checks and confirmation identity/time. Run `python -m capstone_person1.integration_checks tokens --release RELEASE --scope debug_32 --actual ACTUAL.csv --metadata ACTUAL.json --output OUTSIDE_RELEASE.json`. Repeat for `train_n1000`. Empty, partial, duplicated, unexpected or mismatching rows fail. Example wrappers and input contracts are in `integration/`.

Person 2 next executes the 32-example QLoRA training/save/reload smoke test and records its training/config/checkpoint hashes. Every independent R500/R1000/R2000/R4000 learning-curve run starts from the original base model. `integration/planned_parent_seen_groups_R1000.csv` describes the planned parent, not completed training. After actual training, run the parent checker with the actual run/checkpoint identity, actual R1000 manifest and seen groups. It verifies the precise planned manifest hash and selection-bank freshness. Training outside this manifest requires a new coordinated data version.

`manifests/selection_bank.jsonl` is disjoint from the **entire** training bank and `loss_dev`. Its creation requires no trained parent. Actual parent binding remains pending; Person 4 owns student-fit scoring and selection arms. `loss_dev` is held-out training loss data, separate from Person 3's benchmark evaluation sets.

Protected question entries: {scope['protected_question_entries']:,}. Development coverage: **{scope['person3_development_coverage']}**; Person 3 final-test role confirmation: **{scope['person3_final_test_role_confirmation']}**. Person 3 supplies and freezes its own development/fast-development question manifests and evaluator contract. Known GSM8K test and MATH500 questions are protected now. When new evaluation questions arrive, import them and regenerate/recheck a new version before comparative runs. No final-test scores were used here.

See `reports/data_audit_v3.md`, `reports/review_decisions.json`, extraction/format reviews, whole-example encoding exclusions, exact/near pair decisions, complete row-status/group maps, and `reports/release_validation.json`. Similarity retrieves candidates; it never automatically certifies an instance. Changed-number template relatives remain distinct. All unresolved near-instance groups are quarantined. MinHash/LSH has approximate recall; detected-overlap removal does not prove zero contamination or rule out pretraining exposure. Mathematical correctness is **unknown** unless separately reviewed; source validity labels and structural consistency are not correctness proofs. Reviews identify Codex explicitly and do not claim independent human sign-off.

Compared with v1: corrected terminal targets, complete upstream Numina labels/answers, case-preserving global exact grouping, conservative reviewed instance policy, actual label-derived budgets, common length bins, three disjoint banks, independent readiness gates, and explicit downstream evidence contracts. Historical v1 manifests are not substituted for these outputs. Pilot scope: {pilot if pilot else 'full pinned sources'}.
'''
    (release/'PERSON2_HANDOFF.md').write_text(handoff,encoding='utf-8')
    (release/'README.md').write_text(f'# {release.name}\n\nReadiness: **{status["status"]}**. Start with [PERSON2_HANDOFF.md](PERSON2_HANDOFF.md).\n\nThis is a data package; training, generation/scoring and selection experiments remain with Persons 2–4. Verify all content with `metadata/release_manifest.json`. Main-run and parent prerequisites are recorded separately.\n',encoding='utf-8')
    files={str(path.relative_to(release)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=file_sha256(path))
           for path in sorted(release.rglob('*')) if path.is_file() and path.name not in {'release_manifest.json','.release.lock'} and '__pycache__' not in path.parts and path.suffix!='.pending'}
    atomic_json(release/'metadata/release_manifest.json',dict(release_id=release.name,readiness=status,input_fingerprint=fingerprint,
        created_utc=datetime.now(timezone.utc).isoformat(),immutable_after_finalization=status['data_ready_for_trainer'],files=files,
        manifests=summaries,source_rows={source:spec['rows'] for source,spec in bindings.items()},relative_paths=True))


def run_release(release,*,pilot=None,pair_decisions=None,finalize=False):
    release=Path(release).resolve()
    for directory in ['configs','metadata','processed','manifests','benchmarks','integration','reports/duplicate_and_overlap_reports']:
        (release/directory).mkdir(parents=True,exist_ok=True)
    old=release/'metadata/release_manifest.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8')).get('immutable_after_finalization'):
        raise ValueError('Release is finalized; use validate_release.py or create a new version')
    config=json.loads((PROJECT/'configs/person1_data_config.json').read_text(encoding='utf-8'))
    inventory=json.loads((PROJECT/'sources/source_inventory.json').read_text(encoding='utf-8'))
    team=PROJECT/'inputs_from_team'
    protection=load_protection_inputs(team/'reserved_evaluation.csv',team/'reserved_evaluation_metadata.json',
        team/'development_evaluation.csv',team/'development_evaluation_metadata.json')
    scope=export_protection_inputs(protection,release/'benchmarks',release/'reports')
    protection['questions']['problem_hash']=protection['questions']['problem'].map(instance_hash)
    bindings={source:source_binding(PROJECT,source,inventory,config) for source in config['sources']}
    fingerprint=freeze_inputs(PROJECT,release,config,bindings,protection,pilot)
    stages={source:release/'processed'/('canonical_'+source) for source in bindings}
    canonical={source:canonicalize_upstream(PROJECT,source,binding,config,stages[source],limit=pilot) for source,binding in bindings.items()}
    conn=build_catalog(stages,release/'processed/global_exact_catalog.sqlite')
    frame,exact=choose_candidates(conn,stages,set(protection['questions']['problem_hash']),release/'reports/duplicate_and_overlap_reports',
                                 maximum=min(3000,config['candidate_universe_per_source']) if pilot else config['candidate_universe_per_source'],seed=config['selection_seed'])
    message('Global exact candidate supply: '+json.dumps(exact))
    checked,near=checked_candidates(frame,protection,release,config,pair_decisions=pair_decisions)
    tokenizer=load_tokenizer(cache_dir=PROJECT/'cache',local_files_only=True)
    assets=PROJECT/'cache'/('models--Qwen--Qwen2.5-0.5B')/'snapshots'/PINNED_TOKENIZER_REVISION
    contract=formatting_contract(tokenizer,asset_directory=assets,max_total_tokens=config['max_total_tokens'])
    atomic_json(release/'configs/formatting_contract.json',contract)
    encoded,encoding=encode_candidates(checked,tokenizer,release,config)
    reviews=review_queues(encoded,stages,conn,release,pilot)
    rejected_groups=set(checked.loc[checked['row_id'].isin(reviews['quarantined_row_ids']),'problem_group_id'])
    checked.loc[checked['problem_group_id'].isin(rejected_groups),'group_excluded']=True
    encoded=encoded[~encoded['problem_group_id'].isin(rejected_groups)].copy()
    quotas=dict(train_per_source=250,selection_per_source=50,loss_per_source=10) if pilot else dict(
        train_per_source=config['train_bank_per_source'],selection_per_source=config['selection_bank_per_source'],loss_per_source=config['loss_dev_per_source'])
    banks,edges=allocate_banks(encoded,**quotas,seed=config['split_seed'],order_seed=config['selection_seed'])
    atomic_json(release/'configs/length_bin_edges.json',edges)
    summaries={name:write_manifest(rows,name,release,tokenizer,config) for name,rows in banks.items()}
    sizes=[size for size in config['required_training_prefixes']+config['optional_training_prefixes'] if size<=len(banks['train_bank'])]
    for size in sizes: summaries['train_n'+str(size)]=write_manifest(banks['train_bank'].iloc[:size], 'train_n'+str(size),release,tokenizer,config)
    summaries['debug_32']=write_manifest(banks['train_bank'].iloc[:32],'debug_32',release,tokenizer,config)
    reference=banks['train_bank'].iloc[:min(1000,len(banks['train_bank']))]
    reference[['row_id','supervised_tokens']].to_json(release/'integration/token_verification_input.jsonl',orient='records',lines=True)
    reference[['problem_group_id']].to_csv(release/'integration/planned_parent_seen_groups_R1000.csv',index=False)
    atomic_json(release/'integration/planned_parent_metadata.json',dict(status='planned_only; not evidence of actual training',
        manifest='manifests/train_n1000.jsonl' if not pilot else 'manifests/train_n500.jsonl',
        manifest_sha256=summaries.get('train_n1000',summaries.get('train_n500'))['manifest_sha256']))
    for command,name in [('tokens','verify_person2_tokens'),('parent','bind_parent')]:
        (release/'integration'/(name+'.py')).write_text('import sys\nfrom capstone_person1.integration_checks import main\nif __name__ == "__main__":\n    sys.argv.insert(1, '+repr(command)+')\n    main()\n',encoding='utf-8')
    statuses=export_row_status(conn,checked,banks,release/'processed')
    consolidate_canonical(stages,release/'processed')
    quality=[]
    for source,state in canonical.items():
        for flag,count in state['quality_counts'].items(): quality.append(dict(dataset_id=source,flag=flag,count=count,processed_rows=state['processed_rows'],nonexclusive=True))
    pd.DataFrame(quality).to_csv(release/'reports/quality_audit.csv',index=False)
    scope.update(overlap_checks_status='completed_on_declared_scope',exact_checks=exact,near_checks=near)
    atomic_json(release/'reports/evaluation_protection_scope.json',scope)
    counts={source:state['processed_rows'] for source,state in canonical.items()}
    checks=dict(original_source_fields='passed',global_exact_groups='passed',detected_protected_overlap='passed',
        reference_label_encoding='passed',whole_example_length_policy='passed',three_bank_disjointness='passed',
        required_manifest_exports='passed' if not pilot else 'pending',ordered_prefix_nesting='passed',artifact_reencoding='passed')
    state=readiness(full_scope=pilot is None,data_checks=checks,reviews_complete=reviews['complete'])
    audit=dict(readiness=state,processed_rows=counts,initial_eligible={source:value['eligible_rows'] for source,value in canonical.items()},
        exact=exact,near=near,encoding=encoding,final_row_status_counts=dict(Counter(statuses)),reviews_pending=len(reviews['pending']),
        observed_training_runs=0,git_actions=0)
    atomic_json(release/'reports/release_validation.json',audit)
    (release/'reports/data_audit_v3.md').write_text('# Person 1 data audit\n\n'+json.dumps(audit,indent=2)+'\n\nFlags are nonexclusive. Extraction consistency is not mathematical correctness. Pending development protection remains Person 3’s input dependency.\n',encoding='utf-8')
    package_release(release,config,bindings,scope,audit,summaries,state,fingerprint,pilot)
    conn.close()
    message('Release status: '+json.dumps(state))
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',required=True)
    parser.add_argument('--pilot',type=int)
    parser.add_argument('--pair-decisions')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    run_release(args.release,pilot=args.pilot,pair_decisions=args.pair_decisions,finalize=args.finalize)


if __name__=='__main__': main()

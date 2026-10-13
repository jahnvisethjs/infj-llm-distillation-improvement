"""Inspect saved release artifacts independently of pipeline print statements."""
import argparse
import json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from capstone_person1.quality_review_policy import cohort_flags
from capstone_person1.instance_matching import instance_hash
from capstone_person1.pipeline_controls import file_sha256,sha256_text
from capstone_person1.person1_data_contract import validate_manifest,load_tokenizer

def validate_release(release,*,tokenizer=None,reencode=False):
    release=Path(release).resolve()
    metadata=json.loads((release/'metadata/release_manifest.json').read_text(encoding='utf-8'))
    config=json.loads((release/'configs/person1_data_config.json').read_text(encoding='utf-8'))
    if metadata['release_id']!=release.name: raise ValueError('Release ID changed')
    for relative,expected in metadata['files'].items():
        path=(release/relative).resolve()
        if release not in path.parents or not path.is_file(): raise ValueError('Missing or unsafe artifact path')
        if path.stat().st_size!=expected['bytes'] or file_sha256(path)!=expected['sha256']: raise ValueError('Artifact content changed: '+relative)
    sets,orders={},{}
    for name,summary in metadata['manifests'].items():
        path=release/'manifests'/(name+'.jsonl')
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        ids=[row['row_id'] for row in rows];groups=[row['problem_group_id'] for row in rows]
        if len(rows)!=summary['rows'] or len(set(ids))!=len(rows) or len(set(groups))!=len(rows): raise ValueError('Manifest coverage/uniqueness differs')
        if sum(row['supervised_tokens'] for row in rows)!=summary['supervised_tokens']: raise ValueError('Reference token total differs')
        if dict(Counter(row['dataset_id'] for row in rows))!=summary['source_rows']: raise ValueError('Source totals differ')
        if sha256_text('\n'.join(ids)+'\n')!=summary['ordered_id_sha256']: raise ValueError('Ordered IDs changed')
        if any(row['total_tokens']>summary['max_total_tokens'] or row['eos_tokens']!=1 or row['supervised_tokens']!=row['completion_tokens']+1 for row in rows): raise ValueError('Token contract violated')
        if metadata['readiness']['data_ready_for_trainer']:
            bank='train_bank' if name.startswith('train_n') or name=='debug_32' else name
            if any(row.get('split')!=bank or cohort_flags(row,config) for row in rows):raise ValueError('Manifest violates bank membership or reviewed quality policy')
        orders[name]=ids;sets[name]=set(groups)
        if reencode:
            if tokenizer is None: tokenizer=load_tokenizer()
            validate_manifest(path,summary,tokenizer)
    for left,right in [('train_bank','selection_bank'),('train_bank','loss_dev'),('selection_bank','loss_dev')]:
        if sets[left]&sets[right]: raise ValueError('Bank groups intersect')
    for name,ids in orders.items():
        if name.startswith('train_n') or name=='debug_32':
            if ids!=orders['train_bank'][:len(ids)]: raise ValueError('Manifest is not an ordered prefix')
    for source in metadata['source_rows']:
        canonical=release/'processed'/('canonical_'+source+'.parquet')
        checkpoint=json.loads((release/'processed'/('canonical_'+source)/'checkpoint.json').read_text(encoding='utf-8'))
        if pq.ParquetFile(canonical).metadata.num_rows!=sum(shard['rows'] for shard in checkpoint['shards']): raise ValueError('Canonical shard/combined counts differ')
    audit=json.loads((release/'reports/release_validation.json').read_text(encoding='utf-8'))
    if pq.ParquetFile(release/'processed/row_status_audit.parquet').metadata.num_rows!=sum(audit['processed_rows'].values()): raise ValueError('Full row-status coverage differs')
    exclusions=pd.read_csv(release/'reports/whole_example_encoding_exclusions.csv',dtype=str)
    if len(exclusions):
        rejected=pq.read_table(release/'processed/row_status_audit.parquet',columns=['row_id','final_status'],filters=[('row_id','in',exclusions['row_id'].tolist())]).to_pandas()
        if len(rejected)!=len(exclusions) or not rejected['final_status'].eq('reference_encoding_exclusion').all():raise ValueError('Encoding exclusions are mislabeled in the complete row audit')
    protected=pd.read_csv(release/'benchmarks/protected_questions.csv',dtype=str)
    protected_hashes=set(protected['problem'].map(instance_hash))
    if metadata['readiness']['data_ready_for_trainer']:
        if protected['problem_hash'].tolist()!=protected['problem'].map(instance_hash).tolist():raise ValueError('Protected hashes use a different identity normalization')
        for name in ['train_bank','selection_bank','loss_dev']:
            with (release/'manifests'/(name+'.jsonl')).open(encoding='utf-8') as stream:
                if any(instance_hash(json.loads(line)['problem']) in protected_hashes for line in stream):raise ValueError('Bank retains an exact protected question')
    if metadata['readiness']['data_ready_for_trainer']:
        config=json.loads((release/'configs/person1_data_config.json').read_text(encoding='utf-8'))
        for name,key in [('train_bank','train_bank_per_source'),('selection_bank','selection_bank_per_source'),('loss_dev','loss_dev_per_source')]:
            if metadata['manifests'][name]['source_rows']!={'orca':config[key],'numina':config[key]}: raise ValueError('Full release quotas differ')
        if audit['processed_rows']!=metadata['source_rows']: raise ValueError('Full source coverage differs')
        if not all('train_n'+str(size) in metadata['manifests'] for size in [500,1000,2000,4000]): raise ValueError('Required manifests missing')
    return dict(status='passed',release_id=release.name,artifacts_checked=len(metadata['files']),manifests_checked=len(metadata['manifests']),complete_row_status_rows=sum(audit['processed_rows'].values()),readiness=metadata['readiness']['status'],all_manifest_reencoding=reencode)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);parser.add_argument('--reencode',action='store_true')
    args=parser.parse_args();print(json.dumps(validate_release(args.release,reencode=args.reencode),indent=2))

if __name__=='__main__':main()

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1.canonical_data import canonicalize_orca, canonicalize_numina
from capstone_person1.integration_checks import readiness,verify_person2_tokens,bind_parent
from capstone_person1.pipeline_controls import file_sha256
from capstone_person1.source_stream import canonicalize_upstream,source_binding
from capstone_person1.release_catalog import build_catalog,choose_candidates,export_row_status
from capstone_person1.release_reviews import make_review_queue,validate_reviews,sample_formatted


class ReleaseStageTests(unittest.TestCase):
    def test_added_review_cases_must_belong_to_verified_population(self):
        from capstone_person1.release_reviews import include_added_review_rows
        population=pd.DataFrame([dict(row_id=str(i),problem='Pinned question '+str(i)) for i in range(3)])
        sample=population.iloc[:2].copy()
        result=include_added_review_rows(sample,population,['0','1','2'])
        self.assertEqual(result['row_id'].tolist(),['0','1','2'])
        self.assertEqual(result.iloc[-1]['problem'],'Pinned question 2')
        with self.assertRaises(ValueError):include_added_review_rows(sample,population,['0','1','unverified'])
        with self.assertRaises(ValueError):include_added_review_rows(sample,population,['0','1','2','2'])
    def test_prior_rejections_are_hash_bound_and_cover_alternate_group_members(self):
        import sqlite3
        from capstone_person1.release_reviews import prior_rejections, reviewed_rejected_groups
        from capstone_person1.pipeline_controls import sha256_text
        row=dict(row_id='old-member',dataset_id='numina',problem='Find x.',solution='Incorrect reasoning.',final_answer='3',decision='quarantine',reviewer='Codex',reviewed_utc='2026-10-13T00:00:00+00:00',notes='Inspected contradiction.')
        row['source_sha256']=sha256_text(json.dumps({key:row[key] for key in ['row_id','problem','solution','final_answer']},sort_keys=True,ensure_ascii=False))
        reports=self.root/'reports';reports.mkdir()
        path=reports/'prior_rejected_targets.json';path.write_text(json.dumps({'records':[row]}))
        self.assertEqual(prior_rejections(self.root),[row])
        conn=sqlite3.connect(':memory:');self.addCleanup(conn.close)
        conn.execute('CREATE TABLE rows(row_id TEXT,problem_hash TEXT)')
        conn.execute('INSERT INTO rows VALUES(?,?)',['old-member','same-problem'])
        checked=pd.DataFrame([dict(row_id='representative',problem_hash='same-problem',problem_group_id='group-1')])
        self.assertEqual(reviewed_rejected_groups(checked,conn,['old-member']),{'group-1'})
        row['solution']='Altered';path.write_text(json.dumps({'records':[row]}))
        with self.assertRaises(ValueError):prior_rejections(self.root)
    def test_review_sampling_uses_actual_extraction_and_answer_types(self):
        records=[]
        for source in ['orca','numina']:
            for index in range(96):
                mode='terminal_answer_clause' if index>=93 else 'boxed'
                kind='symbolic_scalar' if 90<=index<93 else 'numeric_scalar'
                records.append(dict(row_id=source+str(index),dataset_id=source,answer_parse_status='parsed',
                    answer_extraction_json=json.dumps({'mode':mode}),answer_type=kind))
        sample=sample_formatted(pd.DataFrame(records),count=20)
        for source,rows in sample.groupby('dataset_id'):
            self.assertEqual(len(rows),20)
            self.assertEqual(sum(rows['answer_type'].eq('symbolic_scalar')),3)
            self.assertEqual(sum(rows['answer_extraction_json'].map(lambda value:json.loads(value)['mode']=='terminal_answer_clause')),3)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config={'sources':{'orca':dict(dataset_id='orca-fixture',revision='a'*40,split='train')},'split_seed':1729,'exclude_synthetic_numina':False}
        self.raw=[dict(question=f'How many items at index {i}?',answer=f'Add the quantities to find the count.\nCount = {i+10}\n\nTherefore there are {i+10} items.') for i in range(12)]
        path=self.root/'source.parquet';pq.write_table(pa.Table.from_pylist(self.raw),path)
        item=dict(local_path=path.name,sha256=file_sha256(path),lfs_sha256=file_sha256(path))
        self.binding=source_binding(self.root,'orca',{'orca':dict(self.config['sources']['orca'],files=[item])},self.config)

    def test_complete_original_fields_and_resumable_upstream_sampling(self):
        stage=self.root/'stage'
        result=canonicalize_upstream(self.root,'orca',self.binding,self.config,stage,limit=7,batch_size=4)
        self.assertEqual(result['processed_rows'],7)
        from capstone_person1.snapshot_stream import stage_batches
        rows=[row for batch in stage_batches(stage) for row in batch.to_pylist()]
        self.assertTrue(all(json.loads(row['upstream_fields_json'])==self.raw[int(row['original_row_id'])] for row in rows))
        again=canonicalize_upstream(self.root,'orca',self.binding,self.config,stage,limit=7,batch_size=4)
        self.assertEqual(again['shards'],result['shards'])
        with self.assertRaises(ValueError): canonicalize_upstream(self.root,'orca',self.binding,self.config,stage,limit=8,batch_size=4)
        (stage/result['shards'][0]['name']).write_bytes(b'corrupted')
        with self.assertRaises(ValueError): canonicalize_upstream(self.root,'orca',self.binding,self.config,stage,limit=7,batch_size=4)

    def test_missing_or_falsified_numina_validity_is_quarantined(self):
        base=dict(problem='Find the number.',solution='We add the values in the calculation.\n\nFinal answer: 3',answer='3',source='fixture',synthetic=False,question_type='math-word-problem')
        spec=dict(dataset_id='numina-fixture',revision='b'*40)
        invalid=canonicalize_numina(base,1,spec=spec,config=self.config)
        self.assertFalse(invalid['eligible'])
        self.assertEqual(invalid['numina_solution_validity_raw'],'')
        valid=canonicalize_numina(dict(base,problem_is_valid='Yes',solution_is_valid='Yes'),1,spec=spec,config=self.config)
        self.assertTrue(valid['eligible'])
        conflict=canonicalize_numina(dict(base,problem_is_valid='Yes',solution_is_valid='Yes',answer='4'),1,spec=spec,config=self.config)
        self.assertIn('unproven_solution_target_consistency',conflict['quality_flags'])

    def test_data_readiness_does_not_require_downstream_evidence(self):
        status=readiness(full_scope=True,data_checks={'data':'passed'},reviews_complete=True)
        self.assertEqual(status['status'],'DATA_READY_FOR_TRAINER')
        self.assertEqual(readiness(full_scope=True,data_checks={'data':'passed'},reviews_complete=True,trainer_verified=True,evaluation_frozen=True)['status'],'MAIN_RUN_READY')
        self.assertEqual(readiness(full_scope=True,data_checks={'data':'passed'},reviews_complete=True,trainer_verified=True,evaluation_frozen=True,parent_bound=True)['status'],'CONTINUATION_READY')
        self.assertEqual(readiness(full_scope=False,data_checks={'data':'passed'},reviews_complete=True)['status'],'DIAGNOSTIC_ONLY')
        self.assertEqual(readiness(full_scope=True,data_checks={'data':'passed'},reviews_complete=False)['status'],'DIAGNOSTIC_ONLY')

    def test_unreviewed_or_changed_source_cannot_approve_release(self):
        row=canonicalize_orca(self.raw[0],0,spec=self.config['sources']['orca'],config=self.config)
        path=self.root/'formatted_target_review.csv'
        make_review_queue(pd.DataFrame([row]),path,'formatted')
        self.assertFalse(validate_reviews([path],required_formatted_per_source=1)['complete'])
        frame=pd.read_csv(path,keep_default_na=False);frame.loc[0,'solution']='Changed';frame.to_csv(path,index=False)
        with self.assertRaises(ValueError): validate_reviews([path],required_formatted_per_source=1)

    def test_catalog_exact_protection_and_full_status_mapping(self):
        stage=self.root/'stage';canonicalize_upstream(self.root,'orca',self.binding,self.config,stage,batch_size=4)
        conn=build_catalog({'orca':stage},self.root/'catalog.sqlite');self.addCleanup(conn.close)
        from capstone_person1.instance_matching import instance_hash
        protected={instance_hash(self.raw[0]['question'])}
        frame,report=choose_candidates(conn,{'orca':stage},protected,self.root,maximum=20)
        self.assertEqual(report['exact_protected_rows'],1)
        self.assertEqual(len(frame),11)
        frame['group_representative_id']=frame['row_id'];frame['group_excluded']=False
        statuses=export_row_status(conn,frame,{},self.root)
        self.assertEqual(len(statuses),12)
        self.assertEqual(statuses.count('exact_protected_overlap'),1)

    def test_reference_encoding_exclusions_have_their_own_final_status(self):
        stage=self.root/'stage';canonicalize_upstream(self.root,'orca',self.binding,self.config,stage,batch_size=4)
        conn=build_catalog({'orca':stage},self.root/'catalog.sqlite');self.addCleanup(conn.close)
        frame,_=choose_candidates(conn,{'orca':stage},set(),self.root,maximum=20)
        frame['group_representative_id']=frame['row_id'];frame['group_excluded']=False
        processed=self.root/'release/processed';processed.mkdir(parents=True)
        reports=processed.parent/'reports';reports.mkdir()
        rejected=frame.iloc[0]['row_id']
        pd.DataFrame([dict(row_id=rejected,error_type='ContextLengthError',reason='Whole example exceeds context')]).to_csv(reports/'whole_example_encoding_exclusions.csv',index=False)
        statuses=export_row_status(conn,frame,{},processed)
        self.assertEqual(statuses.count('reference_encoding_exclusion'),1)
        rows=pq.read_table(processed/'row_status_audit.parquet').to_pandas()
        self.assertEqual(rows.set_index('row_id').loc[rejected,'final_status'],'reference_encoding_exclusion')


if __name__=='__main__':unittest.main()

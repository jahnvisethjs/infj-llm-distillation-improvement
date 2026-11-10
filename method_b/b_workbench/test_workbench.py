"""Focused scientific controls, using labelled synthetic fixtures only."""
import ast
import copy
import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import allocator
import common
import continuation
import workbench
import run_weights_free
from compare import validate_pair,COMPARE_FIELDS


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.cfg=common.config()
    def tearDown(self):self.temp.cleanup()

    def test_CPU_preparation_bootstraps_before_encoder_checks(self):
        # Simulate child processes only; this tests orchestration, not installation.
        with patch.object(run_weights_free.subprocess,'Popen') as start, redirect_stdout(io.StringIO()):
            start.return_value.stdout=[]
            start.return_value.wait.return_value=0
            report=run_weights_free.run(self.root,skip_prefetch=True)
        commands=[call.args[0] for call in start.call_args_list]
        self.assertEqual(Path(commands[0][1]).name,'bootstrap_env.py')
        self.assertEqual(report['stages'][0]['stage'],'software_environment')
        prepared_python=str(self.root/'venv_b_training_py312/bin/python')
        self.assertTrue(all(command[0]==prepared_python for command in commands[1:]))
        self.assertIn('test_workbench',commands[1])
        self.assertTrue(report['weights_free_preparation_complete'])

    def test_skipping_install_with_no_environment_fails_before_checks(self):
        with patch.object(run_weights_free.subprocess,'Popen') as start:
            with self.assertRaisesRegex(FileNotFoundError,'prepared Python environment is absent'):
                run_weights_free.run(self.root,skip_install=True)
        start.assert_not_called()

    def test_preparation_stops_if_bootstrap_fails(self):
        with patch.object(run_weights_free.subprocess,'Popen') as start, redirect_stdout(io.StringIO()):
            start.return_value.stdout=['synthetic installer failure\n']
            start.return_value.wait.return_value=1
            with self.assertRaisesRegex(RuntimeError,'software_environment failed'):
                run_weights_free.run(self.root)
        self.assertEqual(start.call_count,1)
        progress=common.read(common.work_root(self.root)/'preparation_logs/progress.json')
        self.assertEqual(progress['stages'][0]['status'],'failed')
        self.assertNotIn('weights_free_preparation_complete',progress)

    def test_missing_parents_do_not_block_data_readiness(self):
        with redirect_stdout(io.StringIO()):report=workbench.status(self.root,self.cfg)
        self.assertTrue(report['data_ready'])
        self.assertFalse(report['continuation_ready'])
        self.assertEqual(len(report['parent_adapters']),4)
        self.assertTrue(all(not p['ready'] for p in report['parent_adapters']))

    def test_missing_adapter_blocks_before_framework_import(self):
        with patch.dict('sys.modules',{'torch':None}):
            with self.assertRaisesRegex(ValueError,'Matching parent adapter is not ready'):
                continuation.train(self.root,self.cfg,'0.5B',17,1)

    def test_draft_protocol_blocks_main_execution(self):
        with self.assertRaisesRegex(ValueError,'protocol is frozen'):
            continuation.validate_protocol(self.root,self.cfg)

    def test_fake_frozen_label_is_not_enough(self):
        cfg=copy.deepcopy(self.cfg);cfg['protocol_status']='frozen'
        with self.assertRaisesRegex(ValueError,'actual shared protocol'):
            continuation.validate_protocol(self.root,cfg)

    def test_frozen_protocol_must_match_shared_recipe(self):
        cfg=copy.deepcopy(self.cfg);cfg['protocol_status']='frozen';cfg['protocol_evidence']='receipt.json'
        evidence={**cfg,'status':'frozen','recorded_by':'synthetic fixture, not project approval'}
        (self.root/'receipt.json').write_text(json.dumps(evidence))
        self.assertEqual(len(continuation.validate_protocol(self.root,cfg)),64)
        evidence['training']={**evidence['training'],'learning_rate':0.1}
        (self.root/'receipt.json').write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError,'training'):
            continuation.validate_protocol(self.root,cfg)

    def test_parent_optimizer_and_rng_never_loaded(self):
        source=continuation.adapted_train_source();ast.parse(source)
        self.assertNotIn('load_training_state(args.parent)',source)
        self.assertNotIn("parent_state['optimizer']",source)
        self.assertNotIn("parent_state['scaler']",source)
        self.assertIn('load_training_state(args.resume)',source)
        self.assertIn('fresh_optimizer_and_scheduler',source)

    def test_loss_and_accumulation_implementation_preserved(self):
        source=continuation.adapted_train_source()
        for fragment in ["ignore_index=-100, reduction='sum'",'loss = loss_sum / target_total',
                         'scaler.scale(loss).backward()',"clip_grad_norm_(params, 1.0)",
                         "state['processed_tokens'] += target_total"]:
            self.assertIn(fragment,source)

    def test_fresh_warmup_and_resume_update_numbers(self):
        self.assertAlmostEqual(continuation.warmup_rate(1e-4,1),1e-5)
        self.assertAlmostEqual(continuation.warmup_rate(1e-4,10),1e-4)
        self.assertAlmostEqual(continuation.warmup_rate(1e-4,125),1e-4)
        self.assertIn('args.lr * min(update / 10.0, 1.0)',continuation.adapted_train_source())

    def test_foreign_or_parent_pickle_cannot_resume(self):
        _,entry=common.profile('0.5B',17)
        with self.assertRaisesRegex(ValueError,"own continuation"):
            continuation.validate_resume(self.root,self.root/'parents/0.5B_seed17/final',entry,'a','b')

    def test_budget_rejects_seen_group(self):
        args=type('Args',(),{'budget_multiplier':1})()
        with self.assertRaisesRegex(ValueError,'parent-seen'):
            continuation.budget_check(args,[{'problem_group_id':'seen'}],[{'counts':{'supervised_tokens':10}}],{},
                                      {'state':{'seen_groups':['seen'],'processed_tokens':10},'one_pass_supervised_tokens':10})

    def test_actual_labels_budget_has_strict_tolerance(self):
        args=type('Args',(),{'budget_multiplier':2})()
        parent={'state':{'seen_groups':[],'processed_tokens':1000},'one_pass_supervised_tokens':1000}
        row=[{'problem_group_id':'fresh'}]
        self.assertEqual(continuation.budget_check(args,row,[{'counts':{'supervised_tokens':2010}}],{},parent)['actual_supervised_tokens'],2010)
        with self.assertRaisesRegex(ValueError,'0.5%'):
            continuation.budget_check(args,row,[{'counts':{'supervised_tokens':2011}}],{},parent)

    def test_archive_path_traversal_rejected_before_write(self):
        archive=self.root/'fixture.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.py','bad')
        with self.assertRaisesRegex(ValueError,'Unsafe archive'):
            workbench.import_parents(self.root,archive)
        self.assertFalse((self.root.parent/'escape.py').exists())

    def test_wrong_parent_hash_is_rejected(self):
        archive=self.root/'fixture.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('adapter/adapter_model.safetensors',b'not actual parent weights')
            z.writestr('adapter/adapter_config.json','{}')
        with self.assertRaisesRegex(ValueError,'Unrecognized parent'):
            workbench.import_parents(self.root,archive)

    def test_corrupt_safetensors_header_rejected(self):
        with self.assertRaises(ValueError):workbench.safetensors_header(b'bad')
        with self.assertRaises(ValueError):workbench.safetensors_header(struct.pack('<Q',1000000000)+b'{}')

    def test_safetensors_offsets_are_checked(self):
        header=json.dumps({'lora_A.weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        good=struct.pack('<Q',len(header))+header+b'\x00'*4
        self.assertIn('lora_A.weight',workbench.safetensors_header(good))
        with self.assertRaises(ValueError):workbench.safetensors_header(good[:-1])

    def test_final_questions_are_locked(self):
        with self.assertRaisesRegex(ValueError,'frozen protocol'):
            workbench.evaluate(self.root,self.cfg,'0.5B',17,1,final_test=True)

    def test_development_commands_can_be_planned_without_parents_or_torch(self):
        folder=common.work_root(self.root)/'1x';folder.mkdir(parents=True)
        (folder/'budget_report.json').write_text(json.dumps({'selected_manifest_sha256':'synthetic-selected-manifest'}))
        with patch.dict('sys.modules',{'torch':None}):
            plan=workbench.evaluate(self.root,self.cfg,'0.5B',17,1,quiet=True)
        self.assertFalse(plan['executed'])
        self.assertEqual(len(plan['commands']),4)
        self.assertTrue(all('--final_test' not in command for command in plan['commands']))

    def test_missing_B_predictions_do_not_produce_fake_accuracy(self):
        with self.assertRaises(FileNotFoundError):validate_pair(self.root/'missing_parent.jsonl',self.root/'missing_B.jsonl')

    def pair(self,change=None):
        a=self.root/'parent.jsonl';b=self.root/'B.jsonl'
        a.write_text(json.dumps({'problem_id':'fixture1','correct':False,'benchmark':'fixture'})+'\n')
        b.write_text(json.dumps({'problem_id':'fixture1','correct':True,'benchmark':'fixture'})+'\n')
        config={name:'same fixture setting' for name in COMPARE_FIELDS}
        a.with_suffix('.config.json').write_text(json.dumps(config))
        if change:config.update(change)
        b.with_suffix('.config.json').write_text(json.dumps(config))
        return a,b

    def test_pair_comparison_aligns_and_computes_real_flags(self):
        a,b=self.pair();self.assertEqual(validate_pair(a,b,require_full_frozen_set=False)['change_percentage_points'],100)

    def test_pair_comparison_rejects_mixed_runtime(self):
        a,b=self.pair({'torch':'different fixture runtime'})
        with self.assertRaisesRegex(ValueError,'torch'):validate_pair(a,b)

    def test_same_parser_name_with_different_code_is_rejected(self):
        a,b=self.pair();ac=common.read(a.with_suffix('.config.json'));bc=common.read(b.with_suffix('.config.json'))
        ac['answer_parser_sha256']='fixture1';bc['answer_parser_sha256']='fixture2'
        a.with_suffix('.config.json').write_text(json.dumps(ac));b.with_suffix('.config.json').write_text(json.dumps(bc))
        with self.assertRaisesRegex(ValueError,'answer_parser_sha256'):validate_pair(a,b)

    def test_pair_comparison_rejects_different_questions(self):
        a,b=self.pair();b.write_text(json.dumps({'problem_id':'other fixture','correct':True})+'\n')
        with self.assertRaisesRegex(ValueError,'same question IDs'):validate_pair(a,b)

    def test_partial_evaluation_cannot_be_called_complete(self):
        a,b=self.pair();hash_value=common.sha(common.HERE/'vendor/evaluation/data/dev_gsm.jsonl')
        for path in [a,b]:
            cfg=common.read(path.with_suffix('.config.json'));cfg['eval_file_sha256']=hash_value
            path.with_suffix('.config.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'complete frozen evaluation'):
            validate_pair(a,b)

    def test_pinned_encoder_matches_parent_executable_sources(self):
        expected=common.read(common.HERE/'reference_records/0.5B_seed17/run_config.json')['reference_code_sha256']
        for name in ['person1_data_contract.py','pipeline_controls.py','instance_matching.py']:
            self.assertEqual(common.sha(common.HERE/'vendor/capstone_person1'/name),expected[name])
        tree=ast.parse((common.HERE/'vendor/capstone_person1/__init__.py').read_text())
        self.assertTrue(all(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) for n in tree.body))

    def test_original_pipeline_positional_padding_with_pad_equal_eos(self):
        p2=common.load_person2()
        encoded={'input_ids':[10,20,151643],'labels':[-100,20,151643],'attention_mask':[1,1,1],'supervised_tokens':2}
        padded=p2.contract.pad_encoded(encoded,6)
        self.assertEqual(padded['labels'],[-100,20,151643,-100,-100,-100])
        self.assertEqual(padded['attention_mask'],[1,1,1,0,0,0])

    def fake_tokenizer(self,offset=0):
        class FixtureTokenizer:
            eos_token_id=151643;pad_token_id=151643
            eos_token='<|endoftext|>';pad_token='<|endoftext|>';is_fast=True
            def convert_tokens_to_ids(self,text):return self.eos_token_id
            def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
                result={'input_ids':[1000+ord(c)+offset for c in text]}
                if return_offsets_mapping:result['offset_mapping']=[(i,i+1) for i in range(len(text))]
                return result
        return FixtureTokenizer()

    def fixture_record(self):
        p2=common.load_person2();tok=self.fake_tokenizer()
        row={'row_id':'synthetic-fixture','problem_group_id':'synthetic-group','problem':'What is 1 + 1?',
             'solution':'1 + 1 = 2.\nFinal answer: 2','final_answer':'2'}
        row.update(p2.contract.encode_example(tok,row['problem'],row['solution'],row['final_answer']))
        return row

    def test_recovered_encoder_has_correct_causal_targets(self):
        p2=common.load_person2();p2.select_model('0.5B');row=self.fixture_record()
        e=p2.encode(row,self.fake_tokenizer(),p2.contract.PROMPT_TEMPLATE,2048)
        self.assertEqual(sum(v!=-100 for v in e['labels'][1:]),row['supervised_tokens'])
        self.assertEqual(e['labels'][-1],151643)
        self.assertTrue(all(v==-100 for v in e['labels'][:row['prompt_tokens']]))

    def test_equal_counts_with_different_1_5B_token_ids_are_rejected(self):
        p2=common.load_person2();row=self.fixture_record();p2.select_model('1.5B')
        try:
            with patch.object(p2,'model_tokenizer',return_value=self.fake_tokenizer()):
                with self.assertRaisesRegex(ValueError,'tokenizer differs'):
                    p2.encode(row,self.fake_tokenizer(offset=1),p2.contract.PROMPT_TEMPLATE,2048)
        finally:p2.select_model('0.5B')

    def test_target_bytes_are_checked_in_addition_to_counts(self):
        p2=common.load_person2();row=self.fixture_record();row['completion']+=' '
        with self.assertRaisesRegex(ValueError,'shared encoder'):
            p2.contract.validate_manifest_record(row,self.fake_tokenizer())


if __name__=='__main__':unittest.main()

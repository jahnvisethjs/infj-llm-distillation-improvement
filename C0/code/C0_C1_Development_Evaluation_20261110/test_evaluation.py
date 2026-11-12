"""CPU integrity and end-to-end orchestration tests; synthetic compute is never a real result."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import evaluate_remaining as n

REAL_EXPECTED=n.expected()


def arguments(root):
    return n.parser().parse_args(['worker','--work',str(root),'--b-root',str(root/'b'),
                                 '--worker-seconds','6000','--allocations','3'])


def dump_rows(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))


class Fixture:
    def __enter__(self):
        self.stack=ExitStack()
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.a=arguments(self.root)
        self.e=deepcopy(REAL_EXPECTED)
        self.e['approved_training_files']={}
        self.targets=[]
        for profile,entry in self.e['parents'].items():
            self.targets.append({'key':'parent_'+profile,'profile':profile,'model_size':entry['model_size'],
                'parent_seed':entry['parent_training_seed'],'continuation_seed':entry['parent_training_seed'],
                'method':'parent','model':entry['model'],'revision':entry['revision'],
                'manifest_sha256':entry['parent_manifest_sha256'],'bank_sha256':None,
                'parent_sha256':entry['expected_adapter_sha256'],'required':True})
        self.targets.append(deepcopy(self.e['screened_B_pilot']))
        self.targets.extend(deepcopy(self.e['C_runs']))
        legacy=deepcopy(self.e['C_runs'][0])
        legacy.update(key='B_legacy_0.5B_seed17_t42',method='B_legacy',required=False,
                      bank_sha256=self.e['legacy_bank_sha256'])
        self.targets.append(legacy)
        for target in self.targets:
            target['approved_files']={}
            target['adapter']=str(self.root/'adapters'/target['key']/'final/adapter')
            weights=Path(target['adapter'])/'adapter_model.safetensors'
            weights.parent.mkdir(parents=True)
            weights.write_text('synthetic weight placeholder '+target['key'])
            target['checkpoint_sha256']=n.sha(weights)
            n.save(weights.parent/'adapter_config.json',{'r':16,'lora_alpha':32,'lora_dropout':.05,
                'peft_type':'LORA','target_modules':['q_proj','k_proj'],'bias':'none'})
            n.save(weights.parent.parent/'run_config.json',{'model':target['model'],'revision':target['revision'],
                'manifest_sha256':target['manifest_sha256'],'seed':target['continuation_seed'],
                'mode':'preliminary' if target['method']=='parent' else 'branch'})
            if target['method']!='parent':
                n.save(weights.parent.parent.parent/'training_summary.json',{'complete':True,'unique_examples':1000})
        parent_adapters={t['profile']:t['adapter'] for t in self.targets if t['method']=='parent'}
        for target in self.targets:
            target['parent_adapter']=parent_adapters[target['profile']]
        self.bench_hashes={}
        self.questions={}
        for bench in ('dev_gsm','dev_math'):
            questions=[{'problem_id':f'{bench}_{i:03d}','question':'Synthetic question '+str(i),
                'gold_answer':'42','benchmark':'gsm' if bench=='dev_gsm' else 'math','split':'development'} for i in range(500)]
            path=self.root/'questions'/(bench+'.jsonl')
            dump_rows(path,questions)
            self.bench_hashes[bench]=n.sha(path)
            self.questions[bench]=questions
        vendor=n.runtime_root(self.a)/'vendor/evaluation'
        vendor.mkdir(parents=True)
        for filename in ('answer_parser.py','evaluate.py'):
            (vendor/filename).write_bytes((Path(n.__file__).parent/'test_sources'/filename).read_bytes())
        self.stack.enter_context(patch.object(n,'control_root',return_value=self.root))
        self.stack.enter_context(patch.object(n,'package_check',return_value='fixture-package'))
        self.stack.enter_context(patch.object(n,'expected',return_value=self.e))
        self.stack.enter_context(patch.object(n,'BENCHMARKS',self.bench_hashes))
        self.stack.enter_context(patch.object(n,'ACTIVE_ROOT',self.root))
        self.plan={'version':n.VERSION,'targets':self.targets,'legacy_B_unavailable':[],
            'matched_B_missing':[{'model_size':m,'parent_seed':p,'continuation_seed':s,'status':'training_and_evaluation_required'}
                for m,p in n.PROFILES for s in n.SEEDS if (m,p,s)!=('0.5B',17,42)],'limitations':['SYNTHETIC TEST ONLY']}
        n.save(self.root/'PLAN.json',self.plan)
        n.save(self.root/'FREEZE.json',{'package_sha256':'fixture-package','plan_sha256':n.digest(self.plan),
                                      'expected_training_sha256':n.sha(n.HERE/'expected_training.json')})
        n.save(self.root/'IMPORT_STATUS.json',{'legacy_B_unavailable':[], 'legacy_B_prediction_files_imported':2,'rejected_sources':[]})
        self.e['C_runs']=[t for t in self.targets if t['method'] in ('C0','C1')]
        return self

    def __exit__(self,*args):
        return self.stack.__exit__(*args)

    def target(self,method='C1'):
        return next(t for t in self.targets if t['method']==method)

    def config(self,t,bench):
        config=n.expected_config(self.a,t,bench)
        config.update(run_id='synthetic_'+t['key']+'_'+bench,checkpoint_id=t['adapter'])
        config['eval_config_hash']=hashlib.sha256(json.dumps({k:config[k] for k in n.GENERATION_FIELDS},sort_keys=True).encode()).hexdigest()[:12]
        return config

    def predictions(self,t,bench):
        config=self.config(t,bench)
        correct_count={'C0':250,'C1':260,'parent':200,'B_screened':240,'B_legacy':235}[t['method']]
        return [{'problem_id':q['problem_id'],'correct':i<correct_count,'parse_status':'ok',
            'parsed_answer':'42' if i<correct_count else '0','gold_answer':'42',
            'raw_output':'Final answer: '+('42' if i<correct_count else '0'),
            'checkpoint_sha256':t['checkpoint_sha256'],'model_revision':t['model']+'@'+t['revision'],
            'train_manifest_sha256':t['manifest_sha256'],'parser_version':config['parser_version'],
            'eval_config_hash':config['eval_config_hash'],'benchmark':q['benchmark'],'run_id':config['run_id'],
            'generated_tokens':10,'latency':.1,'repetition_score':0.,'finish_reason':'eos'}
            for i,q in enumerate(self.questions[bench])]

    def raw(self,t,bench,attempt=1,predictions=None):
        path=n.result_root(self.a,t,bench).parent/f'attempt{attempt:02d}/raw.jsonl'
        dump_rows(path,self.predictions(t,bench) if predictions is None else predictions)
        n.current_json(path.with_suffix('.config.json'),self.config(t,bench))
        return path

    def finalize(self,t,bench):
        path=self.raw(t,bench)
        return n.finalize_result(self.a,t,bench,path,path.with_suffix('.config.json'),'synthetic_compute_fixture')


class PredictionIntegrity(unittest.TestCase):
    def test_adapter_weights_and_scaling_configuration_must_both_match(self):
        with Fixture() as f:
            t=f.target();n.verify_checkpoint(t)
            path=Path(t['adapter'])/'adapter_config.json'
            config=n.read(path);config['lora_alpha']=64;n.current_json(path,config)
            with self.assertRaises(n.IntegrityError):n.verify_checkpoint(t)

    def test_historical_parent_mode_label_does_not_override_exact_weight_identity(self):
        with Fixture() as f:
            t=f.target('parent');p=f.predictions(t,'dev_gsm');cfg=f.config(t,'dev_gsm')
            cfg['train_mode']='preliminary'
            self.assertEqual(len(n.validate_predictions(f.a,t,'dev_gsm',p,cfg)),500)

    def test_full_500_predictions_match_frozen_contract_and_metrics(self):
        with Fixture() as f:
            t=f.target()
            p=f.predictions(t,'dev_gsm')
            self.assertEqual(sum(n.validate_predictions(f.a,t,'dev_gsm',p,f.config(t,'dev_gsm')).values()),260)
            self.assertEqual(n.metrics(p)['accuracy'],.52)

    def test_duplicate_missing_unknown_ids_wrong_checkpoint_and_wrong_gold_refused(self):
        with Fixture() as f:
            t=f.target();p=f.predictions(t,'dev_gsm');cfg=f.config(t,'dev_gsm')
            for altered in (p[:-1],p+[p[0]], [{**p[0],'problem_id':'unknown'},*p[1:]],
                [{**p[0],'checkpoint_sha256':'wrong'},*p[1:]], [{**p[0],'gold_answer':'99'},*p[1:]]):
                with self.assertRaises(n.IntegrityError):n.validate_predictions(f.a,t,'dev_gsm',altered,cfg)

    def test_partial_files_validated_without_claiming_full_coverage(self):
        with Fixture() as f:
            t=f.target();p=f.predictions(t,'dev_gsm')[:16]
            self.assertEqual(len(n.validate_predictions(f.a,t,'dev_gsm',p,f.config(t,'dev_gsm'),partial=True)),16)
            with self.assertRaises(n.IntegrityError):n.metrics(p)

    def test_prompt_parser_platform_and_hash_drift_block_reuse(self):
        with Fixture() as f:
            t=f.target();p=f.predictions(t,'dev_math');cfg=f.config(t,'dev_math')
            for key,bad in [('precision','bf16'),('parser_version','other'),('gpu','different GPU'),
                            ('eval_file_sha256','other'),('eval_config_hash','not-the-native-hash')]:
                with self.assertRaises(n.IntegrityError):
                    n.validate_predictions(f.a,t,'dev_math',p,{**cfg,key:bad})

    def test_final_test_question_file_and_unrecognized_benchmark_are_blocked(self):
        with Fixture() as f:
            with self.assertRaises(n.IntegrityError):n.question_file(f.a,'test_math500')
            q=deepcopy(f.questions['dev_math']);q[0]['split']='test'
            path=f.root/'questions/dev_math.jsonl';dump_rows(path,q)
            f.bench_hashes['dev_math']=n.sha(path)
            with self.assertRaises(n.IntegrityError):n.question_file(f.a,'dev_math')

    def test_published_result_is_atomic_and_existing_artifacts_immutable(self):
        with Fixture() as f:
            t=f.target();f.finalize(t,'dev_gsm')
            root=n.result_root(f.a,t,'dev_gsm')
            before=n.sha(root/'completion.json')
            f.finalize(t,'dev_gsm')
            self.assertEqual(before,n.sha(root/'completion.json'))
            (root/'predictions.jsonl').write_text('changed')
            with self.assertRaises(n.IntegrityError):n.validate_result(f.a,t,'dev_gsm')


class Recovery(unittest.TestCase):
    def test_compatible_partial_prediction_file_resumes_with_same_contract(self):
        with Fixture() as f:
            t=f.target();original=f.predictions(t,'dev_math');path=f.raw(t,'dev_math',predictions=original[:16])
            prefix=path.read_bytes();calls=[]
            def run(command,**kwargs):
                calls.append(command)
                output=Path(command[command.index('--out')+1]);self.assertEqual(output,path)
                with output.open('a') as stream:
                    stream.write(''.join(json.dumps(p)+'\n' for p in original[16:]))
                return SimpleNamespace(returncode=0)
            with patch.object(n,'verify_checkpoint'),patch.object(n,'native_runtime_check'),patch.object(n.subprocess,'run',side_effect=run):
                n.run_evaluation(f.a,t,'dev_math')
            self.assertEqual(len(calls),1)
            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertNotIn('--final_test',calls[0]);self.assertIn('fp32',calls[0]);self.assertIn('1024',calls[0])
            self.assertEqual(len(n.rows(n.result_root(f.a,t,'dev_math')/'predictions.jsonl')),500)

    def test_complete_native_file_recovers_without_model_reload_or_config_rewrite(self):
        with Fixture() as f:
            t=f.target();path=f.raw(t,'dev_gsm');before=path.with_suffix('.config.json').read_bytes()
            with patch.object(n,'verify_checkpoint'),patch.object(n,'native_runtime_check'),patch.object(n.subprocess,'run') as run:
                n.run_evaluation(f.a,t,'dev_gsm');run.assert_not_called()
            self.assertEqual(before,path.with_suffix('.config.json').read_bytes())

    def test_corrupted_partial_is_preserved_and_new_attempt_used(self):
        with Fixture() as f:
            t=f.target();path=f.raw(t,'dev_gsm',predictions=f.predictions(t,'dev_gsm')[:16])
            with path.open('a') as stream:stream.write('{truncated')
            before=path.read_bytes()
            def run(command,**kwargs):
                output=Path(command[command.index('--out')+1]);self.assertEqual(output.parent.name,'attempt02')
                dump_rows(output,f.predictions(t,'dev_gsm'));n.current_json(output.with_suffix('.config.json'),f.config(t,'dev_gsm'))
                return SimpleNamespace(returncode=0)
            with patch.object(n,'verify_checkpoint'),patch.object(n,'native_runtime_check'),patch.object(n.subprocess,'run',side_effect=run):
                n.run_evaluation(f.a,t,'dev_gsm')
            self.assertEqual(path.read_bytes(),before)

    def test_checker_execution_errors_are_retried_and_recorded(self):
        with Fixture() as f:
            t=f.target();p=f.predictions(t,'dev_math');p[0]['parse_status']='verify_error';p[0]['correct']=False
            path=f.raw(t,'dev_math',predictions=p)
            real=n.load_parser(f.a)
            grade={'parsed_answer':'42','gold_answer':'42','parse_status':'ok','correct':True}
            fake=SimpleNamespace(**{k:getattr(real,k) for k in ('_plain','_strip_thousands','gsm_gold','math_gold')},
                score_math=lambda *args:grade)
            with patch.object(n,'load_parser',return_value=fake):
                _,_,receipt=n.finalize_result(f.a,t,'dev_math',path,path.with_suffix('.config.json'),'synthetic_checker_retry')
            self.assertEqual(receipt['checker_retry_rows'],1)
            self.assertEqual(n.rows(path)[0]['parse_status'],'verify_error')
            self.assertTrue((n.result_root(f.a,t,'dev_math')/'checker_retries.jsonl').exists())

    def test_unresolved_checker_errors_block_instead_of_counting_wrong(self):
        with Fixture() as f:
            t=f.target();p=f.predictions(t,'dev_math');p[0]['parse_status']='verify_error';p[0]['correct']=False
            path=f.raw(t,'dev_math',predictions=p);real=n.load_parser(f.a)
            fake=SimpleNamespace(**{k:getattr(real,k) for k in ('_plain','_strip_thousands','gsm_gold','math_gold')},
                score_math=lambda *args:{'parse_status':'verify_error','correct':False})
            with patch.object(n,'load_parser',return_value=fake),self.assertRaises(n.SelectionBlocked):
                n.finalize_result(f.a,t,'dev_math',path,path.with_suffix('.config.json'),'synthetic_unresolved')
            self.assertFalse(n.result_root(f.a,t,'dev_math').exists())


class Comparisons(unittest.TestCase):
    def test_known_paired_difference_and_seed_independence(self):
        earlier={str(i):i<250 for i in range(500)};later={str(i):i<260 for i in range(500)}
        result=n.paired_difference(later,earlier,resamples=200)
        self.assertEqual(result['difference_pp'],2.)
        self.assertEqual(result['wrong_to_correct'],10);self.assertEqual(result['correct_to_wrong'],0)
        self.assertTrue(result['ci95_low_pp']<=2<=result['ci95_high_pp'])
        self.assertIn('not training-seed uncertainty',result['uncertainty_scope'])

    def test_unpaired_ids_and_different_training_banks_are_rejected(self):
        with self.assertRaises(n.IntegrityError):n.paired_difference({'x':True},{'y':False})
        with Fixture() as f:
            self.assertFalse(n.controls_match(f.target('C1'),f.target('B_legacy')))
            self.assertTrue(n.controls_match(f.target('C1'),f.target('C0')))

    def test_incomplete_results_cannot_produce_a_success_marker(self):
        with Fixture() as f:
            n.report(f.a,{},'incomplete_blocked')
            n.build_bundle(f.a,n.read(f.root/'STATUS.json'))
            self.assertFalse((f.root/'EVALUATION_COMPLETE.json').exists())
            with self.assertRaises(n.IntegrityError):n.summarize(f.a)


class FullController(unittest.TestCase):
    def exercise(self,interrupt=False,block=False):
        with Fixture() as f:
            # Canonical results simulate completed GPU work. The real subprocess/controller/receipt/statistical path runs below.
            for target in f.targets:
                for bench in n.BENCHMARKS:f.finalize(target,bench)
            fixture={'root':str(f.root),'expected':f.e,'benchmarks':f.bench_hashes}
            n.save(f.root/'fixture.json',fixture)
            script=f.root/'fixture_driver.py'
            script.write_text(textwrap.dedent('''
                import importlib.util,json,os,pathlib,signal,sys
                spec=importlib.util.spec_from_file_location('eval_fixture',sys.argv[1]);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
                payload=json.loads(pathlib.Path(sys.argv[2]).read_text());root=pathlib.Path(payload['root'])
                n.control_root=lambda a:root;n.package_check=lambda:'fixture-package';n.expected=lambda:payload['expected']
                n.BENCHMARKS=payload['benchmarks'];n.ACTIVE_ROOT=root
                a=n.parser().parse_args(['unit','--work',str(root),'--b-root',str(root/'b')])
                unit=next(u for u in n.units() if u['id']==sys.argv[3]);mode=sys.argv[4]
                if mode=='interrupt' and not (root/'interrupted_once').exists():
                    (root/'interrupted_once').write_text('preserved');os.kill(os.getpid(),signal.SIGTERM)
                if mode=='block' and unit['id']==n.units()[0]['id']:sys.exit(65)
                n.complete_unit(a,unit)
            '''))
            ledger={'package_sha256':'fixture-package','control_directory':str(f.root),
                    'jobs':[{'slot':0,'job_id':'901'},{'slot':1,'job_id':'902'}]}
            n.save(f.root/'submission.json',ledger)
            f.a.submission=str(f.root/'submission.json')
            def child(a,action,**kwargs):
                return [
                    sys.executable,str(script),n.__file__,str(f.root/'fixture.json'),kwargs['unit_id'],
                    'interrupt' if interrupt else 'block' if block else 'normal']
            with patch.object(n,'child',side_effect=child),patch.object(n,'cancel_future') as cancel, \
                    patch.dict(os.environ,{'SLURM_JOB_ID':'901'}),redirect_stdout(io.StringIO()):
                # Keep the 2000-resample unit tests separate; a fixed smaller draw makes full integration fast.
                original=n.paired_difference
                with patch.object(n,'paired_difference',side_effect=lambda later,earlier:original(later,earlier,resamples=80)):
                    n.worker(f.a)
                    if interrupt:
                        self.assertEqual(n.read(f.root/'STATUS.json')['status'],'continuing_in_next_allocation')
                        f.a.slot=1
                        with patch.dict(os.environ,{'SLURM_JOB_ID':'902'}):n.worker(f.a)
                if block:
                    self.assertFalse((f.root/'EVALUATION_COMPLETE.json').exists())
                    self.assertEqual(n.read(f.root/'STATUS.json')['status'],'incomplete_blocked')
                else:
                    self.assertTrue((f.root/'EVALUATION_COMPLETE.json').exists())
                    report=n.read(f.root/'reports/comparison_report.json')
                    self.assertEqual(len(report['paired_comparisons']),78)
                    self.assertEqual(len([r for r in report['paired_comparisons'] if r['comparison']=='C1_minus_C0']),24)
                    self.assertFalse(report['full_matched_B_C0_C1_comparison_complete'])
                    self.assertEqual(len(report['matched_B_missing']),11)
                    self.assertFalse(any('legacy' in r['comparison'] for r in report['paired_comparisons']))
                    self.assertTrue(any(r['method']=='B_legacy' for r in report['results']))
                    s=n.validate_final_marker(f.a)
                    self.assertEqual(s['C0_models_evaluated_of_12'],12);self.assertEqual(s['C1_models_evaluated_of_12'],12)
                    self.assertEqual(s['completed_evaluation_files'],58)
                    before=n.sha(f.root/'STATUS.json')
                    with patch.dict(os.environ,{'SLURM_JOB_ID':'902' if f.a.slot else '901'}):n.worker(f.a)
                    self.assertEqual(before,n.sha(f.root/'STATUS.json'))
                self.assertTrue(cancel.called)

    def test_all_required_evaluations_and_supported_comparisons_complete(self):self.exercise()
    def test_disconnected_allocation_continues_with_pre_submitted_successor(self):self.exercise(interrupt=True)
    def test_integrity_failure_cannot_create_a_complete_marker(self):self.exercise(block=True)


# The already exercised held-submission and duplicate-job tests are inserted by the builder.
class Scheduler(unittest.TestCase):
    def test_unknown_queue_and_accounting_status_cannot_submit_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=arguments(root)
            n.save(root/'submissions/submission_001.json',{'jobs':[{'job_id':'123','slot':0}],'submission_complete':True})
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=1,stdout='',stderr='unknown job')
            with patch.object(n,'preflight'),patch.object(n,'control_root',return_value=root), \
                    patch.object(n.subprocess,'run',side_effect=run),self.assertRaises(n.IntegrityError):n.submit(a)
            self.assertEqual([c[0] for c in calls],['squeue','sacct'])

    def test_completed_accounting_allows_resume_after_queue_records_expire(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=arguments(root);a.allocations=1
            n.save(root/'submissions/submission_001.json',{'jobs':[{'job_id':'123','slot':0}],'submission_complete':True})
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                if command[0]=='squeue':return SimpleNamespace(returncode=1,stdout='',stderr='not in queue')
                if command[0]=='sacct':return SimpleNamespace(returncode=0,stdout='123|COMPLETED|\n',stderr='')
                return SimpleNamespace(returncode=0,stdout='777\n' if command[0]=='sbatch' and '--test-only' not in command else '',stderr='')
            with patch.object(n,'preflight'),patch.object(n,'control_root',return_value=root), \
                    patch.object(n,'HERE',root/'package'),patch.object(n,'package_check',return_value='package'), \
                    patch.object(n,'validate_unit_receipts',return_value=set()),patch.object(n.subprocess,'run',side_effect=run),redirect_stdout(io.StringIO()):n.submit(a)
            self.assertEqual(n.read(root/'submissions/submission_002.json')['jobs'][0]['job_id'],'777')

    def test_allocation_chain_is_held_recorded_and_released_serially(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = arguments(root)
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if command[0] == "sbatch" and "--test-only" not in command:
                    count = sum(c[0] == "sbatch" and "--test-only" not in c for c in calls)
                    return SimpleNamespace(returncode=0, stdout=str(7000+count)+"\n", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch.object(n, "preflight"), patch.object(n, "control_root", return_value=root), \
                    patch.object(n, "validate_unit_receipts", return_value=set()), \
                    patch.object(n, "package_check", return_value="package"), \
                    patch.object(n, "HERE", root / "package"), \
                    patch.object(n.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                n.submit(a)
            ledger = n.read(root / "submissions/submission_001.json")
            self.assertEqual([j["job_id"] for j in ledger["jobs"]], ["7001", "7002", "7003"])
            batches = [c for c in calls if c[0] == "sbatch" and "--test-only" not in c]
            self.assertTrue(all("--hold" in c and "04:00:00" in c and "a100:1" in c for c in batches))
            self.assertNotIn("--dependency", batches[0])
            self.assertIn("afterany:7001", batches[1])
            self.assertIn("afterany:7002", batches[2])
            releases = [c for c in calls if c[:2] == ["scontrol", "release"]]
            self.assertEqual([c[-1] for c in releases], ["7001", "7002", "7003"])
            self.assertTrue(ledger["submission_complete"])

    def test_existing_active_submission_does_not_duplicate_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = arguments(root)
            n.save(root / "submissions/submission_001.json", {"jobs": [{"job_id":"123", "slot":0}], "submission_complete":True})
            with patch.object(n, "preflight"), patch.object(n, "control_root", return_value=root), \
                    patch.object(n, "HERE", root / "package"), \
                    patch.object(n.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="123 RUNNING None", stderr="")) as runner, \
                    redirect_stdout(io.StringIO()):
                n.submit(a)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0][0], "squeue")

    def test_launch_recovers_a_recorded_held_chain_after_a_disconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=arguments(root)
            n.save(root/"submissions/submission_001.json",{"jobs":[{"job_id":"123","slot":0},{"job_id":"124","slot":1}],
                "submission_complete":True,"released_job_ids":["123"]})
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0,stdout="123 RUNNING None\n124 PENDING JobHeldUser\n" if command[0]=="squeue" else "",stderr="")
            with patch.object(n,"preflight"),patch.object(n,"control_root",return_value=root), \
                    patch.object(n,"HERE",root/"package"),patch.object(n.subprocess,"run",side_effect=run),redirect_stdout(io.StringIO()):
                n.submit(a)
            self.assertEqual([c for c in calls if c[0]=="scontrol"],[["scontrol","release","124"]])
            self.assertFalse(any(c[0]=="sbatch" for c in calls))


if __name__=='__main__':
    unittest.main(verbosity=2)

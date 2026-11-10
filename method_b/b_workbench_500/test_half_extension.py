"""Tests of fractional budgets and isolated migration; no models or GPU."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import allocator
import common
import continuation
import run_500_preparation
import workbench


class HalfBudgetTests(unittest.TestCase):
    def test_budget_target_is_exact_without_rounding(self):
        self.assertEqual(common.budget_target(318845,0.5),159422.5)
        self.assertEqual(common.budget_target(318845,1),318845)
        self.assertEqual(common.budget_target(318845,4),1275380)

    def test_cli_names_preserve_existing_integer_run_ids(self):
        for text in ['0.5','1','2','4']:
            self.assertEqual(common.budget_name(common.parse_budget(text)),text+'x')
        self.assertEqual(common.budget_name(1.0),'1x')
        for value in [True,False,0,-0.5,0.25,8,float('nan')]:
            with self.assertRaises(ValueError):common.budget_name(value)

    def test_allocator_keeps_fractional_target_and_integer_bounds(self):
        rows=[{'row_id':name,'problem_group_id':name,'dataset_id':'orca','length_bin':0,'supervised_tokens':tokens}
              for name,tokens in [('a',4),('b',8),('c',12)]]
        selected,report=allocator.allocate(rows,{'a':0,'b':1,'c':2},{('orca',0):1},Fraction(25,2),tolerance='0.04')
        self.assertEqual(report['target_tokens'],12.5)
        self.assertEqual(report['allowed_integer_token_range'],[12,13])
        self.assertEqual(sum(row['supervised_tokens'] for row in selected),12)

    def test_training_half_budget_rejects_rounding_induced_upper_slack(self):
        args=type('Args',(),{'budget_multiplier':0.5})()
        parent={'state':{'seen_groups':[],'processed_tokens':318845},'one_pass_supervised_tokens':318845}
        rows=[{'problem_group_id':'fresh'}]
        valid=continuation.budget_check(args,rows,[{'counts':{'supervised_tokens':160219}}],{},parent)
        self.assertEqual(valid['target_supervised_tokens'],159422.5)
        with self.assertRaisesRegex(ValueError,'0.5%'):
            continuation.budget_check(args,rows,[{'counts':{'supervised_tokens':160220}}],{},parent)

    def test_outputs_use_a_new_version(self):
        self.assertEqual(common.work_root('/example').name,'B_workbench_v2_500')
        self.assertEqual(common.config()['budget_multipliers'],[0.5,1,2,4])

    def test_half_evaluation_can_be_planned_without_parent_or_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=common.work_root(root)/'0.5x'
            common.write_new(folder/'budget_report.json',{'selected_manifest_sha256':'synthetic fixture only'})
            with patch.dict('sys.modules',{'torch':None}):
                result=workbench.evaluate(root,common.config(),'0.5B',17,0.5,quiet=True)
            self.assertFalse(result['executed']);self.assertEqual(len(result['commands']),4)
            self.assertTrue(all('0.5x' in ' '.join(command) for command in result['commands']))

    def test_half_training_cli_reaches_real_parent_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            command=['workbench.py','--root',temp,'train','--model','0.5B','--parent-seed','17','--budget','0.5','--smoke']
            with patch('sys.argv',command),patch.dict('sys.modules',{'torch':None}):
                with self.assertRaisesRegex(ValueError,'Matching parent adapter is not ready'):
                    workbench.main()

    def test_full_four_budget_matrix_has_sixteen_plans_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for mult in [0.5,1,2,4]:
                common.write_new(common.work_root(root)/common.budget_name(mult)/'budget_report.json',
                                 {'selected_manifest_sha256':'synthetic fixture only'})
            with redirect_stdout(io.StringIO()),patch.dict('sys.modules',{'torch':None}):
                plans=workbench.plan_evaluations(root,common.config())
            self.assertEqual(len(plans),16)
            self.assertEqual(sum(p['budget_multiplier']==0.5 for p in plans),4)
            self.assertTrue(all(p['executed'] is False for p in plans))

    def test_extension_uses_existing_environment_and_cache_without_install(self):
        # Synthetic child processes test scheduling only, not execution evidence.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);python=root/'venv_b_training_py312/bin/python'
            python.parent.mkdir(parents=True);python.touch()
            with patch.object(run_500_preparation.subprocess,'Popen') as start,redirect_stdout(io.StringIO()):
                process=start.return_value.__enter__.return_value
                process.stdout=[];process.wait.return_value=0
                result=run_500_preparation.run(root)
            commands=[call.args[0] for call in start.call_args_list]
            self.assertTrue(all(command[0]==str(python) for command in commands))
            self.assertIn('--offline',commands[2]);self.assertEqual(commands[2][-1],'verify')
            self.assertEqual(len(commands),5)
            self.assertFalse(any('pip' in command or 'prefetch' in command for command in commands))
            self.assertTrue(result['CPU_extension_complete'])
            self.assertFalse((root/'outputs/B_workbench_v1').exists())


if __name__=='__main__':unittest.main()

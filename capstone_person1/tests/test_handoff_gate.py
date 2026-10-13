"""Regression tests for the preserved diagnostic notebook's legacy approval cell."""
import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

import pandas as pd

from capstone_person1 import pipeline_controls as controls

ROOT = Path(__file__).resolve().parents[2]


class HandoffGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reports = self.root / 'reports'
        self.manifests = self.root / 'manifests'
        self.reports.mkdir()
        self.manifests.mkdir()
        self.policy = {'version': 'test-v2', 'overlap_scope': 'conservative_lexical_variants'}
        self.train = pd.DataFrame([{'row_id': 'a', 'dataset_id': 'orca', 'original_row_id': '1', 'problem_group_id': 'ga'}])
        self.fresh = pd.DataFrame([{'row_id': 'b', 'dataset_id': 'numina', 'original_row_id': '2', 'problem_group_id': 'gb'}])
        self.sheets = {
            'orca_inspection_25.csv': pd.DataFrame([{'dataset': 'orca', 'original_index': '1', 'manual_keep_quarantine': 'keep', 'manual_solution_complete_yes_no': 'yes'}]),
            'numina_inspection_25.csv': pd.DataFrame([{'dataset': 'numina', 'original_index': '2', 'manual_keep_quarantine': 'keep', 'manual_solution_complete_yes_no': 'yes'}]),
            'orca_answer_extraction_review_25.csv': pd.DataFrame([{'row_id': 'a', 'manual_extraction_correct_yes_no': 'yes'}]),
            'formatted_examples_review_20.csv': pd.DataFrame([{'row_id': 'a', 'manual_prompt_correct_yes_no': 'yes'}]),
            'near_duplicate_review_sample.csv': pd.DataFrame([{'row_id_a': 'a', 'row_id_b': 'outside', 'jaccard_5gram': 0.5, 'manual_same_problem_yes_no': 'no', 'manual_overlap_kind': 'unrelated'}]),
            'evaluation_near_review_sample.csv': pd.DataFrame([{'eval_set_id': 'test', 'eval_problem_id': 'eval1', 'training_row_id': 'outside', 'jaccard_5gram': 0.5, 'manual_same_problem_yes_no': 'no', 'manual_overlap_kind': 'unrelated'}]),
            'missed_match_review.csv': pd.DataFrame([{'row_id_a': 'a', 'row_id_b': 'outside', 'search_method': 'parameter_normalized_signature', 'manual_same_problem_yes_no': 'no', 'manual_overlap_kind': 'unrelated'}]),
        }
        for sheet in self.sheets.values():
            sheet['reviewer'] = 'Fixture reviewer'
            sheet['reviewed_utc'] = '2026-01-01T00:00:00Z'
            sheet['policy_version'] = 'test-v2'
            sheet['resolution'] = 'accepted'
            if 'manual_same_problem_yes_no' in sheet:
                sheet['manual_policy_action'] = 'keep'
        self.train_path = self.manifests / 'train_n1000_full.jsonl'
        self.fresh_path = self.manifests / 'fresh_candidate_pool_R1000_full.jsonl'
        self.train.to_json(self.train_path, orient='records', lines=True)
        self.fresh.to_json(self.fresh_path, orient='records', lines=True)
        self.prompt_path = self.reports / 'prompt_specification.json'
        self.prompt_path.write_text('{"template":"fixture"}', encoding='utf-8')
        self.prompt_hash = controls.file_sha256(self.prompt_path)
        self.train_hash = controls.file_sha256(self.train_path)
        self.fresh_hash = controls.file_sha256(self.fresh_path)
        self.debug_path = self.manifests / 'debug_32_full.jsonl'
        self.debug_path.write_bytes(self.train_path.read_bytes())

    def run_gate(self, approve=True):
        for name, sheet in self.sheets.items():
            sheet.to_csv(self.reports / name, index=False)
        if approve:
            approval = {
                'run_fingerprint': 'fixture', 'policy': self.policy,
                'reviewer': 'Fixture reviewer', 'reviewed_utc': '2026-01-01T00:00:00Z',
                'review_file_sha256': {name: controls.file_sha256(self.reports / name) for name in self.sheets},
                'missed_match_audit': {'notes': 'Fixture independent search completed'},
                'artifact_sha256': {
                    self.train_path.name: self.train_hash, self.fresh_path.name: self.fresh_hash,
                    self.debug_path.name: self.train_hash, self.prompt_path.name: self.prompt_hash,
                },
            }
            (self.reports / 'review_approval.json').write_text(json.dumps(approval), encoding='utf-8')
        summary = {'sha256': self.train_hash}
        fresh_summary = {'sha256': self.fresh_hash}
        ns = dict(vars(controls))
        ns.update({
            'pd': pd, 'REPORT_DIR': self.reports, 'MANIFEST_DIR': self.manifests,
            'RUN_TAG': 'full', 'RUN_FINGERPRINT': 'fixture', 'REVIEW_POLICY': self.policy,
            'working_pool': self.train, 'fresh_pool': self.fresh, 'fresh_summary': fresh_summary,
            'PARENT_CONFIRMED': True, 'DEVELOPMENT_PROTECTED': True, 'TOKEN_LABELS_VERIFIED': True,
            'VERIFIED_R1000_TOKEN_BUDGET': 7, 'RUN_MODE': 'full', 'INTEGRATION_BLOCKERS': [],
            'summaries': {1000: summary}, 'debug_summary': {'sha256': self.train_hash},
            'PROMPT_SPEC_HASH': self.prompt_hash,
        })
        n = json.loads((ROOT / 'capstone_person1/preserved_inputs/pre_named_stages_v3/Person1_Full_Run_v2.ipynb').read_text(encoding='utf-8'))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(''.join(n['cells'][-1]['source']), 'notebook_final_approval', 'exec'), ns)
        return json.loads((self.reports / 'handoff_readiness.json').read_text(encoding='utf-8'))

    def test_valid_fixture_approval(self):
        self.assertTrue(self.run_gate()['research_ready'])

    def test_missing_approval_cannot_release(self):
        with self.assertRaises(RuntimeError): self.run_gate(approve=False)

    def test_quarantined_source_cannot_enter_fresh_pool(self):
        sheet = self.sheets['numina_inspection_25.csv']
        sheet['manual_keep_quarantine'] = 'quarantine'
        sheet['resolution'] = 'quarantined'
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_below_threshold_reviewed_duplicate_cannot_survive(self):
        sheet = self.sheets['near_duplicate_review_sample.csv']
        sheet['row_id_b'] = 'b'
        sheet['manual_same_problem_yes_no'] = 'yes'
        sheet['manual_overlap_kind'] = 'duplicate_instance'
        sheet['manual_policy_action'] = 'remove'
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_reviewed_protected_overlap_cannot_enter_fresh_pool(self):
        sheet = self.sheets['evaluation_near_review_sample.csv']
        sheet['training_row_id'] = 'b'
        sheet['manual_same_problem_yes_no'] = 'yes'
        sheet['manual_overlap_kind'] = 'duplicate_instance'
        sheet['resolution'] = 'quarantined'
        sheet['manual_policy_action'] = 'remove'
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_inconsistent_pair_classification_cannot_release(self):
        self.sheets['near_duplicate_review_sample.csv']['manual_same_problem_yes_no'] = 'yes'
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_readiness_binds_fresh_and_prompt_hashes(self):
        ready = self.run_gate()
        self.assertEqual(ready['fresh_manifest_sha256'], controls.file_sha256(self.fresh_path))
        self.assertEqual(ready['prompt_specification_sha256'], controls.file_sha256(self.prompt_path))

    def test_modified_export_cannot_release(self):
        self.fresh_path.write_text('{}\n', encoding='utf-8')
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_modified_prompt_cannot_release(self):
        self.prompt_path.write_text('{}', encoding='utf-8')
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_empty_mandatory_target_review_cannot_release(self):
        self.sheets['formatted_examples_review_20.csv'] = self.sheets['formatted_examples_review_20.csv'].iloc[:0]
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_accepted_quarantine_decision_cannot_release(self):
        self.sheets['numina_inspection_25.csv']['manual_keep_quarantine'] = 'quarantine'
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_changed_review_cannot_reuse_old_approval(self):
        self.run_gate()
        self.sheets['formatted_examples_review_20.csv']['manual_notes'] = 'Changed after approval'
        with self.assertRaises(RuntimeError): self.run_gate(approve=False)

    def test_variant_requires_explicit_policy_removal(self):
        self.sheets['near_duplicate_review_sample.csv']['manual_overlap_kind'] = 'parameter_or_word_variant'
        with self.assertRaises(RuntimeError): self.run_gate()

    def test_historical_parent_cannot_replace_corrected_parent(self):
        parent = self.root / 'parent.jsonl'
        parent.write_text('{"row_id":"a","problem_group_id":"ga","problem":"Find x."}\n', encoding='utf-8')
        metadata = {
            'checkpoint_id': 'old-parent', 'checkpoint_sha256': 'a' * 64,
            'actual_manifest_sha256': controls.file_sha256(parent),
            'confirmed_by': 'Fixture reviewer', 'confirmed_utc': '2026-01-01T00:00:00Z',
        }
        with self.assertRaisesRegex(ValueError, 'corrected R1000'):
            controls.validate_parent_evidence(pd.DataFrame({'problem_group_id': ['ga']}), parent, metadata,
                                              expected_manifest_sha256='b' * 64)


if __name__ == '__main__':
    unittest.main()

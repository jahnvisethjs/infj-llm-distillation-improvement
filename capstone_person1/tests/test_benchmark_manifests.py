import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from capstone_person1.benchmark_manifests import (
    BENCHMARK_MANIFEST_VERSION, export_protection_inputs, load_protection_inputs,
)
from capstone_person1.pipeline_controls import file_sha256
from capstone_person1 import pipeline_controls as controls
from capstone_person1 import benchmark_manifests as benchmarks


class BenchmarkManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reserved = self.root / 'reserved.csv'
        self.reserved_metadata = self.root / 'reserved.json'
        self.dev = self.root / 'dev.csv'
        self.dev_metadata = self.root / 'dev.json'
        self.write(self.reserved, self.reserved_metadata,
            [{'set_id': 'gtest', 'problem_id': 'g1', 'problem': 'Final GSM question?'},
             {'set_id': 'mtest', 'problem_id': 'm1', 'problem': 'Final MATH question?'}],
            {'gtest': self.source('gsm8k', 'final_test', 'openai/gsm8k', 'test'),
             'mtest': self.source('math', 'final_test', 'HuggingFaceH4/MATH-500', 'test')})

    def source(self, family, role, dataset_id, split, parent=''):
        return dict(benchmark_family=family, role=role, dataset_id=dataset_id, split=split,
                    revision='a' * 40, configuration='main', parent_set_id=parent,
                    selection_policy={'method': 'fixed_seed_sample', 'seed': 1729})

    def write(self, path, metadata_path, rows, sets):
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
        metadata = dict(schema_version=BENCHMARK_MANIFEST_VERSION, confirmed_by='Person 3 fixture',
                        confirmed_utc='2026-01-01T00:00:00Z', sets=sets,
                        set_counts=frame['set_id'].value_counts().to_dict(), manifest_sha256=file_sha256(path))
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

    def make_dev(self):
        rows = [{'set_id': 'gdev', 'problem_id': 'gd1', 'problem': 'Development GSM question?', 'gold_answer': 'private'},
                {'set_id': 'mdev', 'problem_id': 'md1', 'problem': 'Development MATH question?', 'gold_answer': 'private'},
                {'set_id': 'gfast', 'problem_id': 'gd1', 'problem': 'Development GSM question?', 'gold_answer': 'private'},
                {'set_id': 'mfast', 'problem_id': 'md1', 'problem': 'Development MATH question?', 'gold_answer': 'private'}]
        sets = {'gdev': self.source('gsm8k', 'development', 'openai/gsm8k', 'train'),
                'mdev': self.source('math', 'development', 'person3/math-source', 'train'),
                'gfast': self.source('gsm8k', 'fast_development', 'openai/gsm8k', 'train', 'gdev'),
                'mfast': self.source('math', 'fast_development', 'person3/math-source', 'train', 'mdev')}
        self.write(self.dev, self.dev_metadata, rows, sets)

    def load(self):
        return load_protection_inputs(self.reserved, self.reserved_metadata, self.dev, self.dev_metadata)

    def mutate_metadata(self, fn):
        value = json.loads(self.dev_metadata.read_text())
        fn(value)
        self.dev_metadata.write_text(json.dumps(value))

    def test_absent_development_allows_person1_work_and_exports_question_scope(self):
        bundle = self.load()
        self.assertFalse(bundle['development_protected'])
        self.assertEqual(bundle['scope']['person3_development_coverage'], 'pending')
        self.assertTrue(bundle['scope']['person1_data_work_may_continue'])
        report = export_protection_inputs(bundle, self.root / 'processed', self.root / 'reports')
        self.assertEqual(report['protected_question_entries'], 2)
        self.assertEqual(report['overlap_checks_status'], 'pending_pipeline_execution')
        again = export_protection_inputs(bundle, self.root / 'processed', self.root / 'reports')
        self.assertEqual(report['protected_questions_sha256'], again['protected_questions_sha256'])

    def test_valid_frozen_full_and_fast_sets_are_nested_without_gold_export(self):
        self.make_dev()
        bundle = self.load()
        self.assertTrue(bundle['development_protected'])
        self.assertEqual(bundle['scope']['person3_fast_development_coverage'], 'complete')
        self.assertEqual(bundle['scope']['protected_question_entries'], 6)
        self.assertEqual(bundle['scope']['unique_protected_question_texts'], 4)
        self.assertNotIn('gold_answer', bundle['questions'].columns)

    def test_changed_fast_text_and_unknown_parent_fail(self):
        self.make_dev()
        frame = pd.read_csv(self.dev, keep_default_na=False)
        frame.loc[frame['set_id'].eq('gfast'), 'problem'] = 'Different question'
        frame.to_csv(self.dev, index=False)
        self.mutate_metadata(lambda value: value.update(manifest_sha256=file_sha256(self.dev)))
        with self.assertRaisesRegex(ValueError, 'exact subset'): self.load()
        self.make_dev()
        self.mutate_metadata(lambda value: value['sets']['gfast'].update(parent_set_id='unknown'))
        with self.assertRaisesRegex(ValueError, 'parent'): self.load()

    def test_development_final_text_overlap_fails(self):
        self.make_dev()
        frame = pd.read_csv(self.dev, keep_default_na=False)
        frame.loc[frame['set_id'].isin(['gdev', 'gfast']), 'problem'] = '  FINAL GSM question?  '
        frame.to_csv(self.dev, index=False)
        self.mutate_metadata(lambda value: value.update(manifest_sha256=file_sha256(self.dev)))
        with self.assertRaisesRegex(ValueError, 'intersect'): self.load()

    def test_official_final_test_cannot_be_development(self):
        self.make_dev()
        self.mutate_metadata(lambda value: value['sets']['gdev'].update(split='test'))
        with self.assertRaisesRegex(ValueError, 'cannot supply'): self.load()

    def test_stale_hash_incomplete_pair_and_missing_confirmation_fail(self):
        self.make_dev()
        self.dev.write_text(self.dev.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash'): self.load()
        self.make_dev()
        self.dev_metadata.unlink()
        with self.assertRaisesRegex(ValueError, 'pair'): self.load()
        self.make_dev()
        self.mutate_metadata(lambda value: value.update(confirmed_by=''))
        with self.assertRaisesRegex(ValueError, 'confirmation'): self.load()

    def test_roles_are_explicit_and_source_revision_is_immutable(self):
        self.make_dev()
        self.mutate_metadata(lambda value: value['sets']['gdev'].update(role=''))
        with self.assertRaisesRegex(ValueError, 'provenance'): self.load()
        self.make_dev()
        self.mutate_metadata(lambda value: value['sets']['gdev'].update(revision='main'))
        with self.assertRaisesRegex(ValueError, 'immutable'): self.load()
        self.make_dev()
        self.mutate_metadata(lambda value: value['sets']['gdev']['selection_policy'].pop('seed'))
        with self.assertRaisesRegex(ValueError, 'seed'): self.load()

    def test_actual_notebook_exact_and_near_stages_record_honest_scope(self):
        # Exercise the notebook stage seam, including its real variable names.
        metadata_path = self.root / 'reserved_evaluation_metadata.json'
        metadata_path.write_bytes(self.reserved_metadata.read_bytes())
        notebook_path = Path(__file__).resolve().parents[2] / 'Person1_Full_Run_v2.ipynb'
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        ns = dict(vars(controls))
        ns.update(vars(benchmarks))
        ns.update(EVAL_MANIFEST_PATH=self.reserved, INPUT_DIR=self.root,
                  DEV_MANIFEST_PATH=self.dev, DEV_METADATA_PATH=self.dev_metadata,
                  REPORT_DIR=self.root / 'reports', PROCESSED_DIR=self.root / 'processed',
                  INTEGRATION_BLOCKERS=[], universe_per_source=1, log_decision=lambda message: None,
                  NEAR_DUPLICATE_THRESHOLD=.9, DECONTAMINATION_THRESHOLD=.75, MINHASH_PERMUTATIONS=128,
                  REVIEW_POLICY={'near_duplicate_threshold': .9, 'decontamination_threshold': .75,
                                 'overlap_scope': 'conservative_lexical_variants'})
        ns['REPORT_DIR'].mkdir()
        ns['PROCESSED_DIR'].mkdir()
        ns['candidate_df'] = pd.DataFrame([
            {'row_id': 'overlap', 'problem_hash': controls.sha256_text(controls.normalize_problem_for_matching('Final GSM question?'))},
            {'row_id': 'unique', 'problem_hash': 'unique_hash'},
        ])
        exec(compile(''.join(notebook['cells'][20]['source']), 'protection_stage', 'exec'), ns)
        self.assertEqual(ns['candidate_df']['row_id'].tolist(), ['unique'])
        ns['near_pairs_df'] = pd.DataFrame(columns=['row_id_a', 'row_id_b', 'jaccard_5gram'])
        ns['eval_pair_candidates_df'] = pd.DataFrame(columns=['training_row_id', 'jaccard_5gram'])
        exec(compile(''.join(notebook['cells'][24]['source']), 'near_scope_stage', 'exec'), ns)
        report = json.loads((ns['REPORT_DIR'] / 'evaluation_protection_scope.json').read_text())
        self.assertEqual(report['exact_candidate_checks']['overlapping_rows_removed'], 1)
        self.assertEqual(report['person3_development_coverage'], 'pending')
        self.assertEqual(report['near_candidate_checks']['candidate_pairs'], 0)
        self.assertFalse(report['near_candidate_checks']['recall_guaranteed'])


if __name__ == '__main__': unittest.main()

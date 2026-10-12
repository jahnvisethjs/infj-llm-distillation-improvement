import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from capstone_person1.pipeline_controls import (
    answers_equivalent, build_completion, extract_orca_final_answer,
    has_image_reference, looks_answer_only, reconcile_training_tokens,
    save_review_sheet, review_evidence, validate_problem_manifest, validate_parent_evidence,
)


class PipelineControlsTests(unittest.TestCase):
    def test_reported_given_is_not_target(self):
        answer, _ = extract_orca_final_answer(
            'Subtract 6 from the number and divide by 13; result is 2. Find the number.',
            'The result is 2. (x-6)/13 = 2, so x = 32. So the number you are thinking of is 32.')
        self.assertEqual(answer, '32')

    def test_terminal_answer_overrides_earlier_phrase(self):
        self.assertEqual(extract_orca_final_answer('Find the missing number.',
            'The answer is 11. 12 - x = 11.\n\nThe missing number is 1.')[0], '1')
        self.assertEqual(extract_orca_final_answer('What would be the result?',
            'The result is 7.\n\nSo, if 6 is subtracted from the number and then divided by 8, the result would be 6.')[0], '6')

    def test_multi_part_and_verification_total_quarantined(self):
        answer, status = extract_orca_final_answer('How much did each friend spend?',
            'Friend 1: 46. Friend 2: 11. Friend 3: 62.\n\nThe total matches $150.')
        self.assertEqual(answer, '')
        self.assertTrue(status.startswith('ambiguous'))
        self.assertEqual(extract_orca_final_answer('Find the radius and area.',
            'The area is 25.')[0], '')

    def test_no_search_back_into_derivation(self):
        self.assertEqual(extract_orca_final_answer('Find x.',
            'The answer is 7.\n\nThis calculation cannot determine the requested quantity.')[0], '')
        self.assertEqual(extract_orca_final_answer('Find x.', r'We use \boxed{7}. Then x is 9.')[0], '9')

    def test_symbolic_box_and_numeric_ambiguity(self):
        self.assertEqual(extract_orca_final_answer('Find x.', r'Thus \boxed{\frac{9}{2}}.')[0], r'\frac{9}{2}')
        self.assertEqual(extract_orca_final_answer('Find x.', 'Values are 2 and 3.')[0], '')
        self.assertEqual(extract_orca_final_answer('Find x.', 'The answer is 4.5.')[0], '9/2')
        self.assertEqual(extract_orca_final_answer('Find x.', r'\boxed{2}, \boxed{3}.')[0], '')
        self.assertEqual(extract_orca_final_answer('Find x.', r'Therefore x = \sqrt{2}.')[0], '')

    def test_completion_cannot_keep_contradictory_marker(self):
        with self.assertRaises(ValueError): build_completion('x = 32.\nFinal answer: 2', '32')
        self.assertEqual(build_completion('x = 32.\nFinal answer: 32', '32').count('Final answer:'), 1)

    def test_safe_scalar_equivalence(self):
        for a, b in [('9/3', '3'), (r'\frac{9}{2}', '4.5'), ('b = 2', '2'), ('50%', '1/2')]:
            self.assertIs(answers_equivalent(a, b), True)
        self.assertIs(answers_equivalent('3', '4'), False)
        self.assertIsNone(answers_equivalent(r'\sqrt{2}', '1.414'))
        self.assertIsNone(answers_equivalent('__import__("os")', '3'))

    def test_diagram_forms_and_answer_only(self):
        for marker in ['[asy]draw();[/asy]', '![circle](a.png)', '![circle][ref]', '<img src="a">']:
            self.assertTrue(has_image_reference(marker, ''))
        self.assertTrue(looks_answer_only(r'\boxed{\sqrt{2}}', r'\sqrt{2}'))
        self.assertFalse(looks_answer_only('Subtract six from both sides of the equation. Then divide both sides by three to get x = 4.', '4'))

    def test_label_coverage_and_values(self):
        expected = pd.DataFrame({'row_id': ['a', 'b'], 'supervised_tokens': [3, 4]})
        verified = pd.DataFrame({'row_id': ['b', 'a'], 'actual_supervised_tokens': [4, 3]})
        self.assertEqual(len(reconcile_training_tokens(expected, verified)), 2)
        for bad in [verified.iloc[:0], verified.iloc[:1], pd.concat([verified, verified.iloc[:1]]),
                    verified.assign(row_id=['c', 'a']), verified.assign(actual_supervised_tokens=[5, 3]),
                    verified.assign(actual_supervised_tokens=[None, 3])]:
            with self.assertRaises(ValueError): reconcile_training_tokens(expected, bad)

    def test_review_preserved_and_stale_or_blank_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'review.csv'
            frame = pd.DataFrame({'row_id': ['a'], 'manual_extraction_correct_yes_no': ['']})
            save_review_sheet(frame, path)
            with self.assertRaises(ValueError): review_evidence(path, 'v2')
            saved = pd.read_csv(path, keep_default_na=False)
            saved.loc[0, ['manual_extraction_correct_yes_no', 'reviewer', 'reviewed_utc', 'policy_version', 'resolution']] = [
                'yes', 'Reviewer', '2026-01-01T00:00:00Z', 'v2', 'accepted']
            saved.to_csv(path, index=False)
            self.assertEqual(save_review_sheet(frame, path).loc[0, 'reviewer'], 'Reviewer')
            review_evidence(path, 'v2')
            with self.assertRaises(ValueError): save_review_sheet(frame.assign(row_id=['b']), path)

    def test_development_manifest_validation(self):
        with self.assertRaises(ValueError): validate_problem_manifest(pd.DataFrame(columns=['set_id','problem_id','problem']), 'dev')
        with self.assertRaises(ValueError): validate_problem_manifest(pd.DataFrame({'set_id':['dev'], 'problem_id':['x'], 'problem':[None]}), 'dev')
        good = pd.DataFrame({'set_id':['dev'], 'problem_id':['x'], 'problem':['Find x.']})
        validate_problem_manifest(good, 'dev')
        with self.assertRaises(ValueError): validate_problem_manifest(pd.concat([good, good]), 'dev')

    def test_empty_parent_cannot_mean_unseen(self):
        with self.assertRaises(ValueError): validate_parent_evidence(pd.DataFrame({'problem_group_id': []}), 'missing.jsonl', {})


if __name__ == '__main__':
    unittest.main()

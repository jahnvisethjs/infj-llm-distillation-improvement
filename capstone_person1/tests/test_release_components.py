import unittest
import pandas as pd

from capstone_person1.instance_matching import pair_kind, normalize_instance, context_equivalent, apply_instance_decisions, MATCHING_VERSION
from capstone_person1.pipeline_controls import sha256_text
from capstone_person1.pipeline_controls import extract_orca_final_answer
from capstone_person1.pool_allocation import allocate_banks, assign_bins


class ReleaseComponentsTests(unittest.TestCase):
    def test_terminal_quantity_requires_corroborating_derivation(self):
        solution = 'Amount given by dad = $225 - $109.65\nAmount given by dad = $115.35\n\nCarl\'s dad gave him $115.35.'
        self.assertEqual(extract_orca_final_answer('How much did his dad give him?', solution), ('2307/20', 'corroborated_terminal_quantity'))
        self.assertEqual(extract_orca_final_answer('How many?', 'Count = 10\n\nTherefore there were 10 seasons.'), ('10', 'corroborated_terminal_quantity'))
        self.assertNotEqual(extract_orca_final_answer('How many?', 'Count = 9\n\nTherefore there were 10 seasons.')[1], 'corroborated_terminal_quantity')
        self.assertNotEqual(extract_orca_final_answer('How many?', 'The earlier count is 10.')[1], 'corroborated_terminal_quantity')
        self.assertEqual(extract_orca_final_answer('Find x.', 'x = 2\n\nTherefore x is 2 times pi.')[0], '')

    def test_changed_word_and_number_are_template_relatives(self):
        self.assertEqual(pair_kind('Arrange letters of ELLIPSE.', 'Arrange letters of BALLOON.'), 'template_relative')
        self.assertEqual(pair_kind('Buy 3 apples at 4 dollars.', 'Buy 5 apples at 4 dollars.'), 'template_relative')
        self.assertNotEqual(normalize_instance('Find A + a.'), normalize_instance('Find a + a.'))

    def test_assignment_equivalence_needs_problem_context(self):
        self.assertIsNone(context_equivalent('How many items?', 'b=2', '2'))
        self.assertTrue(context_equivalent('Find b.', 'b=2', '2'))
        self.assertTrue(context_equivalent('Find the quantity.', '9/3', '3'))

    def test_similarity_does_not_automatically_group_instances(self):
        frame = pd.DataFrame([{'row_id':'a','problem':'Buy 3 apples.'},{'row_id':'b','problem':'Buy 5 apples.'}])
        pair = dict(row_id_a='a',row_id_b='b',problem_a='Buy 3 apples.',problem_b='Buy 5 apples.',similarity=.99)
        grouped, report, status = apply_instance_decisions(frame, [pair], [])
        self.assertEqual(grouped['problem_group_id'].nunique(), 2)
        self.assertEqual(report.iloc[0]['action'], 'retain_template_relative')

    def test_below_threshold_review_and_group_exclusion_propagate(self):
        frame = pd.DataFrame([{'row_id':'a','problem':'Find x.'},{'row_id':'b','problem':'Determine x.'}])
        decision = dict(row_id_a='a',row_id_b='b',problem_a_sha256=sha256_text('Find x.'),
            problem_b_sha256=sha256_text('Determine x.'),action='join_instances', reviewer='fixture',
            reviewed_utc='2026-01-01T00:00:00Z',policy_version=MATCHING_VERSION)
        protected = [dict(row_id_a='a',row_id_b='protected:test:q',problem_a='Find x.',problem_b='Find x.',similarity=1)]
        grouped, report, _ = apply_instance_decisions(frame, [], protected, [decision])
        self.assertEqual(grouped['problem_group_id'].nunique(), 1)
        self.assertTrue(grouped['group_excluded'].all())

    def test_disjoint_deterministic_banks_and_common_bins(self):
        rows = pd.DataFrame([dict(row_id=f'{source}{i}',problem_group_id=f'g{source}{i}',dataset_id=source,
                                  supervised_tokens=(i % 5)+10) for source in ['orca','numina'] for i in range(40)])
        banks, edges = allocate_banks(rows, train_per_source=20,selection_per_source=5,loss_per_source=2)
        again, _ = allocate_banks(rows.sample(frac=1, random_state=3).reset_index(drop=True), train_per_source=20,selection_per_source=5,loss_per_source=2)
        for name, count in [('train_bank',40),('selection_bank',10),('loss_dev',4)]:
            self.assertEqual(len(banks[name]), count)
            self.assertEqual(banks[name]['row_id'].tolist(), again[name]['row_id'].tolist())
            rebound = assign_bins(banks[name], edges['source_edges'])
            self.assertEqual(rebound['length_bin'].tolist(), banks[name]['length_bin'].tolist())
        all_ids = [value for frame in banks.values() for value in frame['problem_group_id']]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_shortage_and_duplicate_groups_cannot_be_filled(self):
        rows = pd.DataFrame([dict(row_id=s,problem_group_id=s,dataset_id=s,supervised_tokens=10) for s in ['orca','numina']])
        with self.assertRaises(ValueError): allocate_banks(rows)
        with self.assertRaises(ValueError): allocate_banks(pd.concat([rows,rows]))


if __name__ == '__main__': unittest.main()

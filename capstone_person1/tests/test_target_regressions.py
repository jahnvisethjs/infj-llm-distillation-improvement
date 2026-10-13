"""Regression cases for the target defects identified during alignment planning."""
import unittest

from capstone_person1.pipeline_controls import (
    compact_answer, extract_orca_final_answer, build_completion, target_quality_flags,
    diagram_flags, extract_answer_metadata, answer_type,
)


class TargetRegressionTests(unittest.TestCase):
    def test_percentage_display_preserves_normalized_value(self):
        from capstone_person1.person1_data_contract import render_completion
        solution = 'Revenue is 40 and cost 4. Profit divided by cost is 9.\n\nThe answer is 900%.'
        self.assertTrue(render_completion(solution, '9').endswith('Final answer: 900%'))
        solution = 'Interest is 90 on 500 over four years.\n\nThe answer is 4.5%.'
        self.assertTrue(render_completion(solution, '9/200').endswith('Final answer: 4.5%'))
        marked = 'Compute the ratio carefully.\n\nFinal answer: 25%'
        self.assertEqual(render_completion(marked, '1/4'), marked)

    def test_interior_marker_is_not_terminal(self):
        solution = ('Use the equation to isolate the variable and simplify both sides.\n'
                    'Final answer: 2\nAfter correcting the arithmetic, the number is 32.')
        answer, mode = extract_orca_final_answer('Find the number.', solution)
        self.assertNotEqual(answer, '2')
        with self.assertRaises(ValueError): build_completion(solution, '2')
        self.assertIn('nonterminal_or_multiple_final_markers',
                      target_quality_flags('Find the number.', solution, '2', mode))

    def test_existing_correct_block_is_preserved(self):
        solution = 'Subtract 6 from both sides, giving x = 32.\n\nFinal answer: 32'
        self.assertEqual(build_completion(solution, '32'), solution)
        # Equivalent target representation must not rewrite the original correct answer block.
        fraction_solution = 'Divide nine by two to obtain the requested value.\n\nFinal answer: 4.5'
        self.assertEqual(build_completion(fraction_solution, '9/2'), fraction_solution)

    def test_missing_values_are_not_answers(self):
        for value in [None, float('nan')]: self.assertEqual(compact_answer(value), '')

    def test_explicit_alternatives_are_quarantined(self):
        solution = ('Use the equation to isolate the variable and consider both possible values.\n\n'
                    'Final answer: 3 or 4')
        answer, mode = extract_orca_final_answer('Find x.', solution)
        self.assertEqual(answer, '')
        self.assertTrue(mode.startswith('ambiguous'))
        with self.assertRaises(ValueError): build_completion(solution, '3 or 4')

    def test_compound_numeric_expression_cannot_be_partial(self):
        for conclusion in ['The answer is 1/2 + 1/3 = 5/6.', 'Final answer: 1/2 + 1/3 = 5/6']:
            answer, mode = extract_orca_final_answer('Find the sum.', conclusion)
            self.assertNotEqual(answer, '1/2')
            self.assertTrue(answer == '' or answer == '5/6')

    def test_negative_decimal_percentage_and_nested_box(self):
        for solution, expected in [('The value is -4.5.', '-9/2'), ('The answer is 25%.', '1/4'),
                                   (r'Thus \boxed{\frac{-9}{2}}.', r'\frac{-9}{2}')]:
            self.assertEqual(extract_orca_final_answer('Find the value.', solution)[0], expected)

    def test_leading_decimal_and_operator_words(self):
        self.assertEqual(extract_orca_final_answer('Find x.', 'The answer is .5.')[0], '1/2')
        self.assertEqual(extract_orca_final_answer('Find x.', 'The answer is 2 times pi.')[0], '')
        self.assertNotEqual(answer_type('2 times pi'), 'quantity_scalar')

    def test_structured_evidence_span(self):
        solution = 'Rearrange and calculate the value carefully.\nFinal answer: -4.5'
        evidence = extract_answer_metadata('Find x.', solution)
        self.assertEqual(solution[evidence.evidence_start:evidence.evidence_end], evidence.evidence_text)
        self.assertEqual(evidence.confidence, 'marked_terminal')
        self.assertEqual(evidence.answer_type, 'numeric_scalar')

    def test_inline_and_external_diagrams_are_distinct(self):
        self.assertEqual(diagram_flags('<svg viewBox="0 0 1 1"></svg>', ''), ['inline_diagram_code'])
        self.assertEqual(diagram_flags('![triangle](figure.png)', ''), ['missing_or_external_diagram'])


if __name__ == '__main__': unittest.main()

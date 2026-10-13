import unittest
from capstone_person1.quality_review_policy import cohort_flags

class QualityReviewPolicyTests(unittest.TestCase):
    def row(self,problem,solution='',source='cn_k12'):
        return dict(dataset_id='numina',original_source=source,problem=problem,solution=solution)
    def test_reviewed_source_blacklist_does_not_infer_math_correctness(self):
        flags=cohort_flags(self.row('Find 2+3.','The answer is 5.',source='synthetic_math'),{'review_source_blacklist_numina':['synthetic_math']})
        self.assertIn('numina_source_requires_independent_correctness_review',flags)
        self.assertEqual(cohort_flags(self.row('Find 2+3.','The answer is 5.'),{}),[])
    def test_explicit_numbered_requests_and_index_ocr(self):
        text='Given a sequence.\n1. Find its general term.\n2. Find its sum.'
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row(text),{}))
        conditional='(1) If x>0, find x.\n(2) Another sequence is defined. Find its sum.'
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row(conditional),{}))
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row('(Ⅰ) Prove the identity.\n(Ⅱ) Find its sum.'),{}))
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row('Given f(x).\n$(1)$ Discuss monotonicity.\n$(2)$ Find its minimum.'),{}))
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row('Given a sequence. (1) Derive its term. (2) Find its sum.'),{}))
        escaped=r'Given a circle. \\((\\)I\\()\\) Discuss its locus. \\((\\)II\\()\\) Find the area.'
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row(escaped),{}))
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row('1. A value is given.\n2. Determine its square.'),{}))
        self.assertIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row('(1) Complete the square.\n(2) Factorize the polynomial.'),{}))
        self.assertIn('escaped_math_index_requires_source_review',cohort_flags(self.row(r'Find a\_n.'),{}))
        self.assertNotIn('escaped_math_index_requires_source_review',cohort_flags(self.row(r'The result is \_\_\_.'),{}))
        self.assertNotIn('explicit_numbered_multipart_requires_full_target_review',cohort_flags(self.row('Given f(1)=2 and f(2)=3, find f(3).'),{}))
    def test_multiple_answer_slots_and_answer_filled_questions(self):
        self.assertIn('multiple_answer_slots_require_full_target_review',cohort_flags(self.row('The locus equation is _____. Its length is _____.'),{}))
        self.assertIn('multiple_answer_slots_require_full_target_review',cohort_flags(self.row(r'The equation is \_\_\_. The length is \_\_\_.'),{}))
        self.assertNotIn('multiple_answer_slots_require_full_target_review',cohort_flags(self.row(r'The result is \_\_\_.'),{}))
        self.assertIn('boxed_question_content_requires_answer_leakage_review',cohort_flags(self.row(r'The line has \boxed{0} common points.'),{}))
    def test_domains_and_set_parameter_uncertainty(self):
        self.assertIn('higher_derivative_from_tangent_requires_independent_review',cohort_flags(self.row("The tangent at x=1 is y=x+2. Find f(1)+f''(1)."),{}))
        self.assertNotIn('higher_derivative_from_tangent_requires_independent_review',cohort_flags(self.row("The tangent at x=1 is y=x+2. Find f(1)+f'(1)."),{}))
        self.assertIn('placeholder_function_domain_requires_source_review',cohort_flags(self.row(r'Given f: \square \rightarrow \square.'),{}))
        self.assertNotIn('placeholder_function_domain_requires_source_review',cohort_flags(self.row(r'Fill the square \square with a digit.'),{}))
        self.assertIn('set_parameter_alternatives_require_independent_review',cohort_flags(self.row(r'If A \cup B=A, find the value of $m$.'),{}))
    def test_exact_count_rounding_and_explicit_estimates_differ(self):
        solution='Rounding to the nearest whole number yields 12 levels.'
        self.assertIn('exact_discrete_rounding_requires_independent_review',cohort_flags(self.row('Find the number of levels using exactly 225 balls.',solution),{}))
        self.assertNotIn('exact_discrete_rounding_requires_independent_review',cohort_flags(self.row('Estimate the number of levels, rounding the result.',solution),{}))

if __name__=='__main__':unittest.main()

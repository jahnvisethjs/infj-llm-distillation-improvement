"""Reproduce the exact Sol failure at the real frozen validator seam."""

from copy import deepcopy
import unittest

from capstone_person4_replay_c1_15b import evaluation
from capstone_person4_replay_c1_15b.tests import test_pilot as fixtures
from capstone_replay15b_recovery import validation


class ParserStatusTests(unittest.TestCase):
    def test_empty_marker_is_a_false_answer_without_mutating_the_saved_record(self):
        expected, config, records, ids = fixtures.EvaluationTests.settings(None)
        records[0].update(parse_status="empty_marker", correct=False, parsed_answer=None)
        before = deepcopy(records)
        with validation.parser_status_support(evaluation):
            outcomes = evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)
        self.assertFalse(outcomes[records[0]["problem_id"]])
        self.assertEqual(records, before)

    def test_every_normal_parser_status_retains_the_false_score(self):
        for status in validation.NORMAL_STATUSES:
            with self.subTest(status=status):
                expected, config, records, ids = fixtures.EvaluationTests.settings(None)
                records[0].update(parse_status=status, correct=False, parsed_answer=None)
                with validation.parser_status_support(evaluation):
                    outcomes = evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)
                self.assertFalse(outcomes[records[0]["problem_id"]])

    def test_checker_errors_block_even_if_a_corrupt_record_claims_correct(self):
        for status in validation.ERROR_STATUSES:
            for correct in (False, True):
                expected, config, records, ids = fixtures.EvaluationTests.settings(None)
                records[0].update(parse_status=status, correct=correct)
                with validation.parser_status_support(evaluation):
                    with self.assertRaises(validation.CheckerExecutionError):
                        evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)

    def test_unknown_and_inconsistent_outcomes_are_not_reclassified_as_wrong(self):
        for changes in ({"parse_status": "new-unknown-status", "correct": False},
                        {"parse_status": "empty_marker", "correct": True},
                        {"parse_status": "empty_marker", "correct": False, "parsed_answer": "12"},
                        {"correct": "false"}, {"checkpoint_sha256": "other-checkpoint"}):
            expected, config, records, ids = fixtures.EvaluationTests.settings(None)
            records[0].update(changes)
            with validation.parser_status_support(evaluation):
                with self.assertRaises(ValueError):
                    evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)

    def test_context_restores_the_frozen_validator_after_failure(self):
        original = evaluation.validate_predictions
        with self.assertRaises(RuntimeError):
            with validation.parser_status_support(evaluation):
                raise RuntimeError("test")
        self.assertIs(evaluation.validate_predictions, original)


if __name__ == "__main__":
    unittest.main()

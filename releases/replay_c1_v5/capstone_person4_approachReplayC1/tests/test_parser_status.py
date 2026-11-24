"""Cover the frozen parser's complete output vocabulary in the 0.5B launcher."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from capstone_person4_approachReplayC1 import common, evaluation, validation
from capstone_person4_approachReplayC1.tests import test_pilot as fixtures


class ParserStatusTests(unittest.TestCase):
    def test_all_frozen_parser_outcomes_are_handled_without_changing_scores(self):
        path = Path(__file__).parent / "fixtures/frozen_answer_parser.py"
        self.assertEqual(common.sha(path), common.PARSER_SHA)
        spec = importlib.util.spec_from_file_location("_test_v5_frozen_parser", path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        results = [parser.score_gsm(text, "#### 2") for text in (
            "Final answer: 2", "", "Final answer:", "No final result was written.",
            r"\boxed{1} and \boxed{2}", "Final answer: apples")]
        with patch.dict("sys.modules", {"math_verify": SimpleNamespace(parse=lambda *a: [], verify=lambda *a: False)}):
            results.append(parser.score_math("Final answer: x", "y"))

        def timeout(*args):
            raise TimeoutError("SYNTHETIC-checker-error")

        with patch.dict("sys.modules", {"math_verify": SimpleNamespace(parse=lambda *a: ["SYNTHETIC"], verify=timeout)}):
            results.append(parser.score_math("Final answer: x", "y"))
        self.assertEqual({r["parse_status"] for r in results}, validation.NORMAL_STATUSES | {"verify_error"})
        for result in results:
            with self.subTest(status=result["parse_status"]):
                target, config, records, ids = fixtures.EvaluationTests.settings(None)
                records[0].update(result)
                before = deepcopy(records)
                if result["parse_status"] == "verify_error":
                    with self.assertRaises(validation.CheckerExecutionError):
                        evaluation.validate_predictions(records, config, target, "dev_gsm", ids)
                else:
                    actual = evaluation.validate_predictions(records, config, target, "dev_gsm", ids)
                    self.assertEqual(actual[records[0]["problem_id"]], result["correct"])
                self.assertEqual(records, before)

    def test_unknown_status_inconsistent_scores_and_wrong_row_lineage_block(self):
        for change in ({"parse_status": "unknown", "correct": False},
                       {"parse_status": "empty_marker", "correct": True},
                       {"parse_status": "empty", "correct": False, "parsed_answer": "3"},
                       {"checkpoint_sha256": "other-checkpoint"}):
            target, config, records, ids = fixtures.EvaluationTests.settings(None)
            records[0].update(change)
            with self.assertRaises(ValueError):
                evaluation.validate_predictions(records, config, target, "dev_gsm", ids)

    def test_parent123_uses_the_published_05b_parent_and_native_profile(self):
        self.assertEqual(common.MODEL, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(common.parent_sha({"model": "0.5B", "parent_seed": 123}),
                         "aa11429355543bcfe35e47856a3c6ada4e5e0eb7f4647318c1c68aee5bf1c3a2")
        self.assertEqual(common.PARENTS[123]["native_identity_prefix"], "50234d324ee09a25")


if __name__ == "__main__":
    unittest.main()

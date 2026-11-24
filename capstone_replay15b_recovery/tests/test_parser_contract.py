"""Exercise every outcome emitted by the exact hash-pinned parser, without a GPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from capstone_person4_replay_c1_15b import common, evaluation
from capstone_person4_replay_c1_15b.tests import test_pilot as fixtures
from capstone_replay15b_recovery import validation


class FrozenParserTests(unittest.TestCase):
    def test_every_frozen_parser_status_is_supported_or_explicitly_blocked(self):
        path = Path(__file__).parent / "fixtures/frozen_answer_parser.py"
        self.assertEqual(common.sha(path), common.PARSER_SHA)
        spec = importlib.util.spec_from_file_location("_recovery_test_frozen_parser", path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        outcomes = [parser.score_gsm(text, "#### 2") for text in (
            "Final answer: 2", "", "Final answer:", "No final result was written.",
            r"\boxed{1} and \boxed{2}", "Final answer: apples")]
        with patch.dict("sys.modules", {"math_verify": SimpleNamespace(parse=lambda *a: [], verify=lambda *a: False)}):
            outcomes.append(parser.score_math("Final answer: x", "y"))

        def comparison_timeout(*args):
            raise TimeoutError("SYNTHETIC-checker-timeout")

        with patch.dict("sys.modules", {"math_verify": SimpleNamespace(parse=lambda *a: ["SYNTHETIC"], verify=comparison_timeout)}):
            outcomes.append(parser.score_math("Final answer: x", "y"))
        self.assertEqual({r["parse_status"] for r in outcomes}, validation.NORMAL_STATUSES | {"verify_error"})
        for outcome in outcomes:
            with self.subTest(status=outcome["parse_status"]):
                expected, config, records, ids = fixtures.EvaluationTests.settings(None)
                records[0].update(outcome)
                with validation.parser_status_support(evaluation):
                    if outcome["parse_status"] == "verify_error":
                        with self.assertRaises(validation.CheckerExecutionError):
                            evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)
                    else:
                        result = evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)
                        self.assertEqual(result[records[0]["problem_id"]], outcome["correct"])

    def test_checker_error_does_not_hide_wrong_checkpoint_or_runtime(self):
        expected, config, records, ids = fixtures.EvaluationTests.settings(None)
        records[0].update(parse_status="verify_error", correct=False)
        config["torch"] = "2.14.1"
        with validation.parser_status_support(evaluation):
            with self.assertRaisesRegex(ValueError, "runtime/contract mismatch"):
                evaluation.validate_predictions(records, config, expected, "dev_gsm", ids)


if __name__ == "__main__":
    unittest.main()

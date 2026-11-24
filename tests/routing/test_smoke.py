"""Standalone generation-smoke checks with explicit mocked native/live backends."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from capstone_routing import common, inputs, live

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("routing_generation_smoke", REPO / "scripts/smoke_routing_generation.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_authorization_precedes_model_or_file_access(self):
        with patch.object(smoke, "gpu_preflight", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "authorize-gpu"):
                smoke.run(SimpleNamespace(authorize_gpu=False))

    def test_smoke_uses_training_questions_without_mutating_inputs(self):
        records = [{"benchmark": b, "problem_id": str(i), "question": f"{b} {i}", "split": "train", "gold_answer": "1"}
                   for b in ("gsm", "math") for i in range(12)]
        selected = smoke.smoke_questions(records, records)
        self.assertEqual(len(selected), 16)
        self.assertEqual(sum(r["benchmark"] == "gsm" for r in selected), 8)
        self.assertEqual(selected, smoke.smoke_questions(records[::-1], records))
        bad = [{**r, "split": "test"} for r in records]
        with self.assertRaises(common.RoutingError):
            smoke.smoke_questions(bad, records)

    def test_parity_requires_text_length_and_stopping_reason(self):
        native = {"raw_output": "Final answer: 1", "generated_tokens": 4, "finish_reason": "eos"}
        answer = {"answer_text": "Final answer: 1", "generated_tokens": 4, "finish_reason": "eos"}
        self.assertEqual(smoke.compare(native, answer), [])
        self.assertEqual(smoke.compare(native, {**answer, "generated_tokens": 5}), ["generated_tokens"])
        self.assertEqual(smoke.compare(native, {**answer, "answer_text": "Final answer: 2"}), ["raw_output"])

    def test_mocked_end_to_end_preserves_native_and_live_evidence(self):
        root = REPO / ".test_artifacts" / ("routing-smoke-" + uuid.uuid4().hex)
        pool_root, registry_root = root / "pool", root / "registry"
        records = [{"benchmark": b, "problem_id": str(i), "question": f"SYNTHETIC {b} {i}",
                    "family_id": f"{b}-{i}", "split": "train", "gold_answer": "1"} for b in ("gsm", "math") for i in range(8)]
        questions = [{k: v for k, v in r.items() if k != "gold_answer"} for r in records]
        common.write_rows(pool_root / "grading/train.jsonl", records)
        common.seal(pool_root, "pool", synthetic=True)
        candidates = {common.identity(17, m, 42): {"id": common.identity(17, m, 42), "checkpoint": "SYNTHETIC/" + m,
                      "model": common.MODEL, "revision": common.REVISION, "checkpoint_sha256": common.digest(m),
                      "train_manifest_sha256": "e" * 64, "continuation_seed": 42, "method": m, "recipe": {"SYNTHETIC": True}}
                      for m in common.METHODS}
        common.write(registry_root / "registry.json", {"SYNTHETIC": True})
        common.seal(registry_root, "registry", synthetic=True)
        fixture = REPO / "capstone_replay15b_recovery/tests/fixtures"
        native = root / "native"
        common.write_bytes(native / "evaluate.py", (fixture / "frozen_evaluate.py").read_bytes())
        common.write_bytes(native / "answer_parser.py", (fixture / "frozen_answer_parser.py").read_bytes())
        common.write(root / "config.json", {"version": "routing-v1", "evaluator": {
            "path": str(native / "evaluate.py"), "parser_path": str(native / "answer_parser.py")}})
        def native_runner(command, **kwargs):
            out = Path(command[command.index("--out") + 1])
            checkpoint = command[command.index("--adapter") + 1]
            candidate = next(c for c in candidates.values() if c["checkpoint"] == checkpoint)
            qp = Path(command[command.index("--eval_file") + 1])
            config = {**common.SETTINGS, "checkpoint_sha256": candidate["checkpoint_sha256"],
                      "train_manifest_sha256": candidate["train_manifest_sha256"], "train_seed": 42,
                      "train_arm": {"A": "REPEAT", "B": "RANDOM", "C1": "C1"}[candidate["method"]],
                      "eval_file_sha256": common.sha(qp), "eval_config_hash": "mocked"}
            common.write(out.with_suffix(".config.json"), config)
            common.write_rows(out, [{"benchmark": r["benchmark"], "problem_id": r["problem_id"],
                "raw_output": "Final answer: 1", "parsed_answer": "1", "correct": True, "parse_status": "ok",
                "generated_tokens": 4, "finish_reason": "eos", "checkpoint_sha256": candidate["checkpoint_sha256"],
                "train_manifest_sha256": candidate["train_manifest_sha256"], "model_revision": common.MODEL + "@" + common.REVISION,
                "parser_version": "p3-parser-v6", "eval_config_hash": "mocked"} for r in common.rows(qp)])
        class Generator:
            def __init__(self, config):
                pass
            def generate(self, text, candidate):
                return {"answer_text": "Final answer: 1", "generated_tokens": 4, "finish_reason": "eos"}
        args = SimpleNamespace(authorize_gpu=True, code_root=str(REPO), config=str(root / "config.json"),
                               registry=str(registry_root), pool=str(pool_root), output=str(root / "result"),
                               parent_seed=17, continuation_seed=42)
        with patch.object(inputs, "load_registry", return_value={"synthetic": False, "candidates": candidates}), \
             patch("capstone_routing.pool.pool_questions", return_value=questions), \
             patch.object(smoke, "gpu_preflight", return_value={"actual_gpu_execution": False, "mocked": True}), \
             patch.object(smoke.subprocess, "run", side_effect=native_runner), \
             patch.object(live, "VerifiedGenerator", Generator):
            result = smoke.run(args)
        self.assertTrue(result["native_live_match"])
        self.assertFalse(result["runtime"]["actual_gpu_execution"])
        self.assertEqual(result["native_responses"], 48)
        self.assertEqual(result["live_responses"], 48)
        self.assertFalse(result["test_questions_used"])
        common.bundle(root / "result", "generation_smoke")


if __name__ == "__main__":
    unittest.main()

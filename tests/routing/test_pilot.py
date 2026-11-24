"""Pilot checks: family isolation, native reuse, training-only tuning, and mocked generation."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from capstone_routing import common, pilot

REPO = Path(__file__).resolve().parents[2]


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("routing-pilot-" + uuid.uuid4().hex)
        self.registry = self.root / "registry"
        self.pool = self.root / "pool"
        self.candidates = {common.identity(17, method, seed): {
            "id": common.identity(17, method, seed), "method": method, "parent_seed": 17,
            "continuation_seed": seed, "checkpoint_sha256": common.digest([method, seed]),
            "parent_sha256": "a" * 64, "checkpoint": "SYNTHETIC/" + common.identity(17, method, seed),
            "train_manifest_sha256": "d" * 64, "model": common.MODEL, "revision": common.REVISION,
            "recipe": {"SYNTHETIC": True}, "construction": {"SYNTHETIC": True}}
            for method, seed in (("A", 42), ("B", 42), ("C1", 42), ("A", 137), ("A", 2026))}
        common.write(self.registry / "registry.json", {"candidates": self.candidates, "synthetic": True})
        common.seal(self.registry, "registry", synthetic=True)
        self.questions = []
        for split in ("train", "validation", "test"):
            qs = [{"benchmark": b, "problem_id": f"{split}_{b}_{i}", "question": f"SYNTHETIC {split} {b} {i}",
                   "family_id": f"{split}_family_{i // 2}", "split": split}
                  for b in ("gsm", "math") for i in range(12)]
            self.questions.extend(qs)
            common.write_rows(self.pool / (split + ".questions.jsonl"), qs)
            common.write_rows(self.pool / "grading" / (split + ".jsonl"), [{**q, "gold_answer": "1"} for q in qs])
        common.seal(self.pool, "pool", [self.registry], synthetic=True)
        self.plan = self.root / "plan"
        self.loaded_registry = {"candidates": self.candidates, "synthetic": True}

    def prepare(self):
        with patch.object(pilot, "load_registry", return_value=self.loaded_registry):
            return pilot.prepare(self.registry, self.pool, self.plan, train_size=16, validation_size=8)

    def native(self, directory, candidate, records, flip=False):
        common.write_rows(directory / "questions.jsonl", records)
        config = {**common.SETTINGS, "checkpoint_sha256": candidate["checkpoint_sha256"],
                  "train_manifest_sha256": candidate["train_manifest_sha256"],
                  "train_seed": candidate["continuation_seed"],
                  "train_arm": {"A": "REPEAT", "B": "RANDOM", "C1": "C1"}[candidate["method"]],
                  "eval_file_sha256": common.sha(directory / "questions.jsonl"), "eval_config_hash": "fixture"}
        common.write(directory / "predictions.config.json", config)
        common.write_rows(directory / "predictions.jsonl", [{"benchmark": q["benchmark"], "problem_id": q["problem_id"],
            "raw_output": "Final answer: 1" if not flip else "Final answer: 2", "parsed_answer": "1" if not flip else "2",
            "parse_status": "ok", "correct": (int(q["problem_id"].rsplit("_", 1)[1]) % 2 == 0) != flip,
            "generated_tokens": 4, "finish_reason": "eos", "checkpoint_sha256": candidate["checkpoint_sha256"],
            "train_manifest_sha256": candidate["train_manifest_sha256"],
            "model_revision": common.MODEL + "@" + common.REVISION, "parser_version": common.SETTINGS["parser_version"],
            "eval_config_hash": "fixture"} for q in records])

    def evaluator_config(self):
        fixture = REPO / "capstone_replay15b_recovery/tests/fixtures"
        runtime = self.root / "native"
        common.write_bytes(runtime / "evaluate.py", (fixture / "frozen_evaluate.py").read_bytes())
        common.write_bytes(runtime / "answer_parser.py", (fixture / "frozen_answer_parser.py").read_bytes())
        return {"evaluator": {"path": str(runtime / "evaluate.py"), "parser_path": str(runtime / "answer_parser.py")}}

    def reusable_fixture(self):
        self.prepare()
        _, _, grading, _ = pilot.load_plan(self.plan)
        source = self.root / "old_labels"
        for cid, candidate in self.candidates.items():
            directory = source / cid / "chunks" / "all"
            self.native(directory, candidate, list(grading.values()))
            common.seal(directory, "label_chunk", synthetic=True)
        return source, grading

    def test_prepare_is_balanced_keeps_whole_families_and_never_reads_test_gold(self):
        real_rows = common.rows
        def guarded(path):
            self.assertNotEqual(Path(path), self.pool / "grading/test.jsonl")
            return real_rows(path)
        with patch.object(pilot, "rows", side_effect=guarded):
            result = self.prepare()
        self.assertEqual(result["maximum_responses"], 120)
        plan, qs, _, _ = pilot.load_plan(self.plan)
        self.assertEqual(plan["actual_counts"], {"train": 16, "validation": 8})
        self.assertEqual(plan["benchmark_counts"]["train"], {"gsm": 8, "math": 8})
        self.assertFalse(plan["test_questions_used"])
        chosen = {common.qkey(q) for q in qs}
        for q in qs:
            family = [r for r in self.questions if r["family_id"] == q["family_id"]]
            self.assertTrue(all(common.qkey(r) in chosen for r in family))
        other = pilot.family_sample([q for q in self.questions if q["split"] == "train"][::-1], 16, [42, "train"])
        self.assertEqual(other, common.rows(self.plan / "train.questions.jsonl"))

    def test_reuse_preserves_native_contract_and_rejects_conflicting_attempts(self):
        source, grading = self.reusable_fixture()
        values, evidence = pilot.reusable([source], self.candidates, grading, True)
        self.assertEqual(len(values), 120)
        self.assertEqual(len(evidence), 5)
        self.assertEqual(pilot.reusable([source, source], self.candidates, grading, True)[0], values)
        cid, candidate = next(iter(self.candidates.items()))
        conflict = self.root / "conflict" / cid / "chunks" / "all"
        self.native(conflict, candidate, list(grading.values()), flip=True)
        common.seal(conflict, "label_chunk", synthetic=True)
        with self.assertRaisesRegex(common.RoutingError, "Conflicting"):
            pilot.reusable([source, self.root / "conflict"], self.candidates, grading, True)

    def test_unsealed_sources_are_skipped_and_test_sources_rejected(self):
        source, grading = self.reusable_fixture()
        cid, candidate = next(iter(self.candidates.items()))
        partial = self.root / "partial" / cid / "chunks" / "unfinished"
        self.native(partial, candidate, list(grading.values()))
        self.assertEqual(pilot.reusable([self.root / "partial"], self.candidates, grading, True), ({}, []))
        testdir = self.root / "test-source" / cid / "chunks" / "finished"
        self.native(testdir, candidate, common.rows(self.pool / "grading/test.jsonl"))
        common.seal(testdir, "label_chunk", synthetic=True)
        with self.assertRaisesRegex(common.RoutingError, "Final-test"):
            pilot.reusable([self.root / "test-source"], self.candidates, grading, True)

    def test_reused_questions_must_match_gold_and_runtime(self):
        source, grading = self.reusable_fixture()
        bad = deepcopy(grading)
        key = next(iter(bad))
        bad[key]["gold_answer"] = "WRONG"
        with self.assertRaisesRegex(common.RoutingError, "reference"):
            pilot.reusable([source], self.candidates, bad, True)
        with self.assertRaisesRegex(common.RoutingError, "Mixed"):
            pilot.reusable([source], self.candidates, grading, False)

    def test_all_reused_labeling_needs_no_gpu_and_report_uses_training_only_settings(self):
        source, grading = self.reusable_fixture()
        labels = self.root / "labels"
        with patch.object(pilot, "load_registry", return_value=self.loaded_registry), \
             patch.object(pilot, "gpu_preflight", side_effect=AssertionError("no GPU")), \
             patch.object(pilot.subprocess, "run", side_effect=AssertionError("no generation")):
            meta = pilot.label(self.plan, self.evaluator_config(), labels, [source])
        self.assertEqual((meta["reused_answers"], meta["generated_answers"]), (120, 0))
        class Encoder:
            def encode(self, text):
                number = int(text.rsplit(" ", 1)[1])
                return [float(number % 2 == 0)] + [0.] * 383
        original_selection = pilot.select_settings
        def guarded_selection(questions, vectors, y, plan):
            self.assertTrue(all(q["split"] == "train" for q in questions))
            self.assertTrue(all(len(values) == 16 for values in y.values()))
            return original_selection(questions, vectors, y, plan)
        with patch.object(pilot, "select_settings", side_effect=guarded_selection):
            result = pilot.fit(self.plan, labels, self.root / "report", Encoder(), resamples=30, permutations=20)
        self.assertTrue(result["complete"] and result["synthetic"])
        self.assertFalse(result["test_questions_used"] or result["full_experiment_started"])
        self.assertEqual(len(result["metrics"]), 2)
        selections = common.read(self.root / "report/selection.json")
        self.assertTrue(all(len(s["candidates"]) == 3 for s in selections.values()))
        self.assertIn("SYNTHETIC TEST", (self.root / "report/report.md").read_text())
        common.bundle(self.root / "report", "pilot_report")

    def test_generation_is_explicit_and_resume_preserves_completed_chunks(self):
        self.prepare()
        cfg = self.evaluator_config()
        with patch.object(pilot, "load_registry", return_value=self.loaded_registry), \
             patch.object(pilot, "gpu_preflight", side_effect=AssertionError("must not allocate")):
            with self.assertRaisesRegex(common.RoutingError, "authorize-gpu"):
                pilot.label(self.plan, cfg, self.root / "denied")
        def generate(command, **kwargs):
            qp = Path(command[command.index("--eval_file") + 1])
            candidate = next(cp for cp in self.candidates.values() if cp["checkpoint"] == command[command.index("--adapter") + 1])
            qs = common.rows(qp)
            self.assertTrue(all(q["split"] in ("train", "validation") for q in qs))
            self.native(qp.parent, candidate, qs)
        with patch.object(pilot, "load_registry", return_value=self.loaded_registry), \
             patch.object(pilot, "gpu_preflight"), patch("capstone_routing.live.verify_candidate"), \
             patch.object(pilot.subprocess, "run", side_effect=generate) as calls:
            result = pilot.label(self.plan, cfg, self.root / "generated", authorize_gpu=True)
        self.assertEqual(result["generated_answers"], 120)
        self.assertEqual(calls.call_count, 10)
        with patch.object(pilot, "load_registry", return_value=self.loaded_registry), \
             patch.object(pilot, "gpu_preflight", side_effect=AssertionError("no GPU")):
            resumed = pilot.label(self.plan, cfg, self.root / "resumed", [self.root / "generated"])
        self.assertEqual((resumed["generated_answers"], resumed["reused_answers"]), (0, 120))

    def test_cv_never_splits_families_and_has_no_empty_fold(self):
        self.prepare()
        qs = common.rows(self.plan / "train.questions.jsonl")
        folds = pilot.family_folds(qs)
        self.assertEqual(set(folds), {0, 1, 2})
        for family in {q["family_id"] for q in qs}:
            self.assertEqual(len({f for q, f in zip(qs, folds) if q["family_id"] == family}), 1)


if __name__ == "__main__":
    unittest.main()

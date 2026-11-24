"""Native-shaped evaluation-only integration tests; no model or GPU is loaded."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from capstone_person4_replay_c1_15b import common, evaluation, inputs, runtime
from capstone_person4_replay_c1_15b.tests import test_pilot as fixtures
from capstone_replay15b_recovery import PARENT17_IDENTITY, recovery, validation
from capstone_replay15b_recovery.__main__ import parser

REPO = Path(__file__).resolve().parents[2]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("replay15b-recovery-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.args = parser().parse_args(["run", "--repo", str(REPO), "--config", str(self.root / "config.json")])
        common.write(self.args.config, {"SYNTHETIC": True})
        self.bundle = SimpleNamespace(root=self.root / "pilot", identity=PARENT17_IDENTITY,
            cfg={"model": "1.5B", "parent_seed": 17, "training_python": "/SYNTHETIC/python",
                 "slurm": {"account": "class_account", "partition": "public", "qos": "class"}},
            binding={"SYNTHETIC": "original-training-binding"},
            context=SimpleNamespace(runtime=self.root / "native_runtime"))
        reference = Path(__file__).parent / "fixtures/frozen_evaluate.py"
        common.write_bytes(self.bundle.context.runtime / "vendor/evaluation/evaluate.py", reference.read_bytes())
        common.write_bytes(self.bundle.context.runtime / "vendor/evaluation/answer_parser.py",
                           (reference.parent / "frozen_answer_parser.py").read_bytes())
        self.targets, self.payloads = {}, {}
        self.questions = {}
        for benchmark in common.QUESTIONS:
            path = self.root / (benchmark + ".jsonl")
            common.write_rows(path, [{"problem_id": f"q{i}"} for i in range(500)])
            self.questions[benchmark] = path, {f"q{i}" for i in range(500)}
        for method in ("A", "C1", "REPLAY_C1_50_50"):
            for seed in common.SEEDS:
                checkpoint = self.root / "SYNTHETIC-adapters" / f"{method}-{seed}"
                common.write_bytes(checkpoint / "adapter_model.safetensors", f"SYNTHETIC-not-weights-{method}-{seed}".encode())
                target = {"method": method, "seed": seed, "checkpoint": str(checkpoint),
                          "checkpoint_sha256": common.sha(checkpoint / "adapter_model.safetensors"),
                          "manifest_sha256": "SYNTHETIC-manifest"}
                self.targets[method, seed] = target
                for benchmark in common.QUESTIONS:
                    _, config, records, _ = fixtures.EvaluationTests.settings(None)
                    config.update(checkpoint_sha256=target["checkpoint_sha256"], train_seed=seed,
                                  train_arm="REPEAT" if method == "A" else method,
                                  eval_file_sha256=common.QUESTIONS[benchmark])
                    self.payloads[method, seed, benchmark] = records, config
        self.stack.enter_context(patch.object(evaluation, "target", side_effect=lambda b, m, s, a: self.targets[m, s]))
        self.stack.enter_context(patch.object(evaluation, "question_file", side_effect=lambda b, name: self.questions[name]))
        self.stack.enter_context(patch.object(evaluation, "load_selection", return_value=({}, {"SYNTHETIC": True})))
        self.stack.enter_context(patch.object(evaluation, "validate_run", return_value={
            "checkpoint": "SYNTHETIC-trained-checkpoint", "adapter_sha256": "SYNTHETIC-hash",
            "portion_tokens": {"replay": 159000, "fresh": 159000}}))
        original_bootstrap = evaluation.paired_bootstrap
        self.stack.enter_context(patch.object(evaluation, "paired_bootstrap",
            side_effect=lambda left, right: original_bootstrap(left, right, resamples=10)))
        self.stack.enter_context(patch.object(evaluation, "native_c1_result", side_effect=self.native_c1))
        parent123 = self.root / "parent123/reports/eval01/accuracy.csv"
        common.write_bytes(parent123, b"SYNTHETIC-existing-parent123-results\n")

    def native_c1(self, bundle, expected, benchmark, ids):
        records, config = self.payloads["C1", expected["seed"], benchmark]
        path = self.root / "native_C1" / f"seed{expected['seed']}" / benchmark
        # Create native-shaped files up front in prepare(), never inside inspect.
        evaluation.validate_predictions(records, config, expected, benchmark, ids)
        return deepcopy(records), deepcopy(config), {"source": str(path / "predictions.jsonl"),
            "completion_sha256": common.sha(path / "completion.json"), "reused": True}

    def prepare(self, *, omit=(), raw=False):
        for key, (records, config) in self.payloads.items():
            method, seed, benchmark = key
            if method == "C1":
                dest = self.root / "native_C1" / f"seed{seed}" / benchmark
                common.write_rows(dest / "predictions.jsonl", records)
                common.write(dest / "predictions.config.json", config)
                common.write(dest / "completion.json", {"SYNTHETIC": True})
            elif key not in omit:
                dest = evaluation.result_root(self.bundle, method, seed, benchmark, "eval01")
                if raw:
                    common.write_rows(dest / "native/raw.jsonl", records)
                    common.write(dest / "native/raw.config.json", config)
                else:
                    common.write_rows(dest / "predictions.jsonl", records)
                    common.write(dest / "predictions.config.json", config)
                    common.write(dest / "completion.json", {"identity": self.bundle.identity,
                        "target": self.targets[method, seed], "benchmark": benchmark,
                        "evaluation_attempt": "eval01", "complete": True,
                        "predictions_sha256": common.sha(dest / "predictions.jsonl"),
                        "config_sha256": common.sha(dest / "predictions.config.json")})

    def snapshots(self):
        return {str(p.relative_to(self.root)): common.sha(p) for p in self.root.rglob("*") if p.is_file()}

    def change_payload(self, key, status):
        self.payloads[key][0][0].update(parse_status=status, correct=False, parsed_answer=None)

    def fake_generator(self, command, **kwargs):
        self.assertIn("--precision", command)
        self.assertEqual(command[command.index("--precision") + 1], "fp32")
        self.assertEqual(command[command.index("--max_new_tokens") + 1], "1024")
        checkpoint = command[command.index("--adapter") + 1]
        target_key = next(k for k, value in self.targets.items() if value["checkpoint"] == checkpoint)
        benchmark = Path(command[command.index("--eval_file") + 1]).stem
        records, config = self.payloads[*target_key, benchmark]
        raw = Path(command[command.index("--out") + 1])
        common.write_rows(raw, records)
        common.write(raw.with_suffix(".config.json"), config)
        return SimpleNamespace(returncode=0)

    def test_inspection_is_read_only_and_finds_18_results(self):
        self.change_payload(("A", 42, "dev_math"), "empty_marker")
        self.prepare(raw=True)
        before = self.snapshots()
        with patch.object(recovery.subprocess, "run", side_effect=AssertionError("no GPU or jobs")):
            _, result = recovery.inspect_sources(self.bundle, common, evaluation, self.args)
        self.assertTrue(result["ready"])
        self.assertEqual(result["reuse_benchmark_files"], 18)
        self.assertEqual(result["generate_benchmark_files"], 0)
        self.assertEqual(before, self.snapshots())

    def test_all_valid_raw_outputs_recover_and_finalize_without_gpu_or_training(self):
        self.change_payload(("A", 42, "dev_math"), "empty_marker")
        self.change_payload(("REPLAY_C1_50_50", 137, "dev_math"), "empty")
        self.prepare(raw=True)
        before = self.snapshots()
        with patch.object(recovery.subprocess, "run", side_effect=AssertionError("no GPU or training")):
            result = recovery.run(self.bundle, common, evaluation, self.args)
        self.assertTrue(result["complete"])
        self.assertFalse(result["training_restarted"])
        self.assertEqual(result["generated_benchmark_files"], 0)
        for name, expected in before.items():
            self.assertEqual(common.sha(self.root / name), expected)
        saved = evaluation.result_root(self.bundle, "A", 42, "dev_math", "eval02_recovery")
        first = common.rows(saved / "predictions.jsonl")[0]
        self.assertEqual(first["parse_status"], "empty_marker")
        self.assertIs(first["correct"], False)
        self.assertTrue((Path(result["report_directory"]) / "accuracy.csv").is_file())
        self.assertEqual(len(common.read(Path(result["report_directory"]) / "summary.json")["contrasts"]), 18)
        self.assertEqual(recovery.run(self.bundle, common, evaluation, self.args), result)
        (Path(result["report_directory"]) / "accuracy.csv").write_text("SYNTHETIC-tamper", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            recovery.run(self.bundle, common, evaluation, self.args)

    def test_missing_benchmark_requires_authorization_before_any_writes(self):
        self.prepare(omit=[("A", 42, "dev_math")])
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "authorize-gpu"):
            recovery.run(self.bundle, common, evaluation, self.args)
        self.assertEqual(before, self.snapshots())

    def test_only_missing_benchmark_is_generated_and_originals_are_preserved(self):
        self.prepare(omit=[("REPLAY_C1_50_50", 2026, "dev_math")])
        before = self.snapshots()
        self.args.authorize_gpu = True
        with patch.object(recovery.subprocess, "run", side_effect=self.fake_generator) as generator:
            result = recovery.run(self.bundle, common, evaluation, self.args)
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(result["generated_benchmark_files"], 1)
        self.assertEqual(result["reused_benchmark_files"], 17)
        for name, expected in before.items():
            self.assertEqual(common.sha(self.root / name), expected)

    def test_checker_exception_requires_a_clean_benchmark_not_a_false_score(self):
        key = ("A", 42, "dev_math")
        self.change_payload(key, "verify_error")
        self.prepare(raw=True)
        _, plan = recovery.inspect_sources(self.bundle, common, evaluation, self.args)
        self.assertTrue(plan["ready"])
        self.assertEqual(plan["generate_benchmark_files"], 1)
        self.payloads[key][0][0]["parse_status"] = "ok"
        self.args.authorize_gpu = True
        with patch.object(recovery.subprocess, "run", side_effect=self.fake_generator) as generator:
            result = recovery.run(self.bundle, common, evaluation, self.args)
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(result["generated_benchmark_files"], 1)
        destination = evaluation.result_root(self.bundle, *key, "eval02_recovery")
        receipt = common.read(destination / "completion.json")
        self.assertEqual(receipt["origin"]["prior_attempt"]["checker_errors"][0]["parse_status"], "verify_error")

    def test_a_fresh_checker_exception_leaves_failure_evidence_without_success(self):
        key = ("A", 42, "dev_math")
        self.change_payload(key, "verify_error")
        self.prepare(raw=True)
        self.args.authorize_gpu = True
        with patch.object(recovery.subprocess, "run", side_effect=self.fake_generator):
            with self.assertRaises(validation.CheckerExecutionError):
                recovery.run(self.bundle, common, evaluation, self.args)
        root = recovery.recovery_root(self.bundle, self.args)
        self.assertTrue((root / "failure.json").exists())
        self.assertFalse((root / "completion.json").exists())
        self.assertFalse((self.bundle.root / "reports/eval02_recovery/accuracy.csv").exists())

    def test_unknown_status_and_wrong_raw_runtime_block_recovery(self):
        self.change_payload(("A", 42, "dev_math"), "unknown")
        self.payloads["A", 137, "dev_math"][1]["torch"] = "2.14.1"
        self.prepare(raw=True)
        _, plan = recovery.inspect_sources(self.bundle, common, evaluation, self.args)
        self.assertFalse(plan["ready"])
        self.assertEqual(len(plan["blocked"]), 2)
        with self.assertRaisesRegex(ValueError, "invalid source"):
            recovery.run(self.bundle, common, evaluation, self.args)

    def test_corrupt_canonical_receipt_never_falls_back_to_raw_or_regeneration(self):
        self.prepare()
        source = evaluation.result_root(self.bundle, "A", 42, "dev_math", "eval01")
        (source / "predictions.jsonl").write_text("SYNTHETIC corruption", encoding="utf-8")
        _, plan = recovery.inspect_sources(self.bundle, common, evaluation, self.args)
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["generate_benchmark_files"], 0)
        self.assertIn("Hash mismatch", plan["blocked"][0]["error"])

    def test_partial_raw_coverage_can_be_regenerated_but_duplicates_block(self):
        self.payloads["A", 42, "dev_math"] = (self.payloads["A", 42, "dev_math"][0][:100],
                                              self.payloads["A", 42, "dev_math"][1])
        records, config = self.payloads["A", 137, "dev_math"]
        self.payloads["A", 137, "dev_math"] = (records[:-1] + [records[0]], config)
        self.prepare(raw=True)
        _, plan = recovery.inspect_sources(self.bundle, common, evaluation, self.args)
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["generate_benchmark_files"], 1)
        self.assertEqual(len(plan["blocked"]), 1)

    def test_original_attempt_and_invalid_names_cannot_be_recovery_destinations(self):
        for value in ("eval01", "../escape", ""):
            self.args.evaluation_attempt = value
            with self.assertRaises(ValueError):
                recovery.inspect_sources(self.bundle, common, evaluation, self.args)

    def test_submission_is_one_evaluation_job_and_is_idempotent(self):
        self.prepare(omit=[("A", 42, "dev_math")])
        self.args.authorize_gpu = True
        with patch.object(recovery.subprocess, "run", return_value=SimpleNamespace(stdout="123456;cluster\n")) as scheduler:
            result = recovery.submit(self.bundle, common, evaluation, self.args)
            repeated = recovery.submit(self.bundle, common, evaluation, self.args)
        self.assertEqual(scheduler.call_count, 1)
        self.assertEqual(result["job_id"], "123456")
        self.assertTrue(repeated["reused_submission"])
        self.assertFalse(result["training_restarted"])
        self.assertIn("--gpus=a100:1", result["command"])
        self.assertFalse(any("train42" in arg or "train137" in arg or "train2026" in arg for arg in result["command"]))

    def test_failed_submission_claim_prevents_a_second_automatic_submission(self):
        self.prepare()
        self.args.authorize_gpu = True
        with patch.object(recovery.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["sbatch"])):
            with self.assertRaises(subprocess.CalledProcessError):
                recovery.submit(self.bundle, common, evaluation, self.args)
        with patch.object(recovery.subprocess, "run", side_effect=AssertionError("must not submit again")):
            with self.assertRaisesRegex(ValueError, "Interrupted submission"):
                recovery.submit(self.bundle, common, evaluation, self.args)

    def test_parent123_is_rejected_before_reading_weights_or_native_context(self):
        cfg = common.read(REPO / "configs/replay15b.parent123.sol.example.json")
        path = self.root / "parent123_config.json"
        common.write(path, cfg)
        self.args.config = str(path)
        with patch.object(inputs, "check_environment", side_effect=AssertionError("must stop first")):
            with self.assertRaisesRegex(ValueError, "restricted to parent17"):
                recovery.load_original(self.args)

    def test_wrong_training_identity_is_rejected_before_run_validation(self):
        cfg = common.read(REPO / "configs/replay15b.parent17.sol.example.json")
        path = self.root / "parent17_config.json"
        common.write(path, cfg)
        self.args.config = str(path)
        with patch.object(inputs, "check_environment"), \
                patch.object(inputs, "load", return_value=SimpleNamespace(identity="wrong-training-identity")):
            with self.assertRaisesRegex(ValueError, "existing parent17 training identity"):
                recovery.load_original(self.args)

    def test_loading_the_existing_identity_validates_all_three_training_runs(self):
        cfg = common.read(REPO / "configs/replay15b.parent17.sol.example.json")
        path = self.root / "parent17_config.json"
        common.write(path, cfg)
        self.args.config = str(path)
        with patch.object(inputs, "check_environment"), patch.object(inputs, "load", return_value=self.bundle), \
                patch.object(runtime, "validate_run", return_value={"passed": True}) as verify_run:
            bundle, loaded_common, loaded_evaluation = recovery.load_original(self.args)
        self.assertIs(bundle, self.bundle)
        self.assertIs(loaded_common, common)
        self.assertIs(loaded_evaluation, evaluation)
        self.assertEqual([call.args[2] for call in verify_run.call_args_list], [42, 137, 2026])

    def test_the_original_source_fingerprints_stay_exactly_v1(self):
        self.assertEqual(recovery.verify_original(REPO), REPO)
        self.assertEqual(common.code_hashes(), recovery.FROZEN_CODE)

    def test_help_imports_without_torch_or_the_original_runtime(self):
        command = "import sys; from capstone_replay15b_recovery.__main__ import parser; parser(); assert 'torch' not in sys.modules; assert 'capstone_person4_replay_c1_15b' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", command], cwd=REPO, check=True)


if __name__ == "__main__":
    unittest.main()

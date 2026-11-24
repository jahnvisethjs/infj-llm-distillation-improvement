"""Parent/model routing checks for the isolated 0.5B deployment."""

from copy import deepcopy
import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from capstone_person4_approachReplayC1 import VERSION, common, evaluation, inputs, runtime, sol
from capstone_person4_approachReplayC1.__main__ import parser
from capstone_person4_approachReplayC1.tests import test_pilot as pilot


class ProfileTests(pilot.Fixture):
    def configured(self, seed):
        path = self.root / f"parent{seed}.json"
        args = parser().parse_args([
            "configure", "--parent-seed", str(seed), "--config-output", str(path),
            "--work", str(self.root / "work"), "--b-root", str(self.root / "b"),
            "--a-eval-root", str(self.root / "a"), "--output-root", str(self.root / "out")])
        result = sol.configure(args)
        self.assertFalse(result["jobs_submitted"])
        return common.load_config(path)

    def fake_native(self, cfg):
        context = SimpleNamespace(
            root=Path(cfg["b_root"]), bank_path=self.root / "candidate_bank.jsonl",
            entry={"expected_adapter_sha256": common.parent_sha(cfg),
                   "model": common.MODEL, "revision": common.REVISION})
        native = SimpleNamespace(load_context=Mock(return_value=context),
                                 artifact=Mock(return_value=Path(cfg["c1_profile"])),
                                 verify_proof=Mock(return_value={"verified": True}))
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        self.stack.enter_context(patch.object(inputs.importlib.util, "spec_from_file_location", return_value=spec))
        self.stack.enter_context(patch.object(inputs.importlib.util, "module_from_spec", return_value=native))
        self.stack.enter_context(patch.object(inputs, "checked", side_effect=lambda path, expected: Path(path)))
        return native, context

    def test_both_configurations_use_separate_verified_profiles(self):
        for seed in common.PARENTS:
            with self.subTest(parent=seed):
                cfg = self.configured(seed)
                self.assertEqual(cfg["model"], "0.5B")
                self.assertIn(f"0.5B_seed{seed}", cfg["c1_profile"])
                self.assertTrue(cfg["c1_profile"].endswith(common.PARENTS[seed]["native_identity_prefix"]))
                self.assertEqual(cfg["continuation_seeds"], [42, 137, 2026])
                self.assertEqual(common.profile(cfg), f"0.5B_parent{seed}")
        self.assertNotEqual(common.parent_sha({"model": "0.5B", "parent_seed": 17}),
                            common.parent_sha({"model": "0.5B", "parent_seed": 123}))

    def test_native_routes_model_parent_and_actual_label_proof_for_both(self):
        for seed in common.PARENTS:
            with self.subTest(parent=seed):
                cfg = self.configured(seed)
                native, context = self.fake_native(cfg)
                self.assertEqual(inputs.native_context(cfg), (native, context))
                args = native.load_context.call_args.args[0]
                self.assertEqual((args.model, args.parent_seed), ("0.5B", seed))
                native.verify_proof.assert_called_once_with(context)

    def test_wrong_parent_hash_model_revision_and_profile_are_rejected(self):
        cfg = self.configured(123)
        native, context = self.fake_native(cfg)
        for key, value, message in (
            ("expected_adapter_sha256", common.PARENTS[17]["sha256"], "Wrong original"),
            ("model", "Qwen/Qwen2.5-1.5B", "0.5B revision"),
            ("revision", "wrong-revision", "0.5B revision")):
            original = context.entry[key]
            context.entry[key] = value
            with self.assertRaisesRegex(common.PilotError, message):
                inputs.native_context(cfg)
            context.entry[key] = original
        native.artifact.return_value = self.root / "other-native-profile"
        with self.assertRaisesRegex(common.PilotError, "Native profile identity changed"):
            inputs.native_context(cfg)
        native.verify_proof.assert_not_called()

    def test_invalid_model_parent_version_or_fractional_seed_cannot_load(self):
        cfg = self.configured(17)
        for key, value in (("model", "1.5B"), ("parent_seed", 42), ("parent_seed", 17.0),
                           ("parent_seed", True), ("version", "replay-c1-screened-4955-v1")):
            changed = {**cfg, key: value}
            path = self.root / f"invalid-{key}-{value}.json"
            common.write(path, changed)
            with self.assertRaises(common.PilotError):
                common.load_config(path)

    def test_native_score_validation_failure_is_not_replaced_with_other_parent_cache(self):
        cfg = self.configured(123)
        native, context = self.fake_native(cfg)
        context.parent, context.bank = [], []
        native.load_scores = Mock(side_effect=common.PilotError("cached scores belong to another parent"))
        with patch.object(inputs, "validate_population"):
            with self.assertRaisesRegex(common.PilotError, "another parent"):
                inputs.load(cfg)
        self.assertFalse(Path(cfg["output_root"]).exists())

    def test_scheduler_names_and_worker_are_isolated_for_each_parent(self):
        for seed in common.PARENTS:
            cfg = self.configured(seed)
            bundle = SimpleNamespace(cfg=cfg, root=self.root / f"pilot{seed}", identity=f"SYNTHETIC-{seed}")
            counter = itertools.count(100)
            with patch.object(sol.subprocess, "run", side_effect=lambda *a, **k: SimpleNamespace(stdout=str(next(counter)))):
                report = sol.submit(bundle, "attempt01", "eval01")
            self.assertEqual(report["parent_seed"], seed)
            self.assertEqual(len(report["jobs"]), 7)
            for job in report["jobs"]:
                self.assertIn(f"--job-name=replay-c1-p{seed}-" + job["stage"], job["command"])
                self.assertIn(str(pilot.REPO / "scripts/sol/replay_c1_worker.sh"), job["command"])

    def test_existing_package_name_and_profile_scripts_are_fingerprinted(self):
        files = common.code_hashes()
        self.assertIn("capstone_person4_approachReplayC1/validation.py", files)
        self.assertIn("scripts/sol/replay_c1_worker.sh", files)
        self.assertFalse(any("replay15b" in name for name in files))



class Parent123RuntimeTests(pilot.RuntimeTests):
    """Exercise inherited runtime contracts again for parent123, with no GPU calls."""

    def setUp(self):
        super().setUp()
        self.bundle.cfg["parent_seed"] = 123

    def mock_native_training(self):
        torch = super().mock_native_training()
        self.bundle.cfg["parent_seed"] = 123
        return torch

    def test_completion_cannot_be_reused_under_other_parent(self):
        self.select()
        self.mock_native_training()
        runtime.train(self.bundle, 42, smoke=True)
        self.bundle.cfg["parent_seed"] = 17
        path = self.bundle.root / "smokes/seed42_attempt01"
        with self.assertRaisesRegex(common.PilotError, "Run identity"):
            runtime.validate_run(self.bundle, path, 42, smoke=True)


class Parent123EvaluationTests(pilot.EvaluationTests):
    def setUp(self):
        super().setUp()
        self.bundle.cfg["parent_seed"] = 123

    def test_native_and_published_123_paths_reuse_matching_checkpoints(self):
        import gzip
        expected, config, records, ids = self.settings()
        expected["method"] = config["train_arm"] = "C1"
        self.bundle.cfg.update(work=str(self.root / "work"), c1_eval_root=str(self.root / "native"))
        for layout in ("published", "native"):
            if layout == "native":
                dest = self.root / "native/results/C1_0.5B_p123_t42/dev_gsm"
            else:
                dest = self.root / "github_screened_4955_latest/C1/evaluation/0.5B_parent123_seed42/dev_gsm"
            common.write(dest / "predictions.config.json", config)
            raw = "".join(common.canonical(r) + "\n" for r in records).encode()
            common.write_bytes(dest / "predictions.jsonl.gz", gzip.compress(raw, mtime=0))
            common.write(dest / "completion.json", {
                "complete": True, "checkpoint_sha256": expected["checkpoint_sha256"],
                "eval_file_sha256": common.QUESTIONS["dev_gsm"], "benchmark": "dev_gsm",
                "reserved_final_tests_performed": False, "config_sha256": common.sha(dest / "predictions.config.json"),
                "predictions_sha256": hashlib.sha256(raw).hexdigest()})
            reused, _, origin = evaluation.native_c1_result(self.bundle, expected, "dev_gsm", ids)
            self.assertEqual(reused, records)
            self.assertEqual(origin["source_layout"], layout)
        changed = {**expected, "checkpoint_sha256": "other-parent-checkpoint"}
        with self.assertRaisesRegex(common.PilotError, "another experiment"):
            evaluation.native_c1_result(self.bundle, changed, "dev_gsm", ids)

    def test_other_model_or_revision_predictions_cannot_be_reused(self):
        expected, config, records, ids = self.settings()
        for key, value in (("model", "Qwen/Qwen2.5-1.5B"),
                           ("revision", "8faed761d45a263340a0528343f099c05c9a4323")):
            with self.assertRaisesRegex(common.PilotError, "runtime/contract mismatch"):
                evaluation.validate_predictions(records, {**config, key: value}, expected, "dev_gsm", ids)


if __name__ == "__main__":
    unittest.main()

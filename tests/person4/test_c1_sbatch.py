"""C1 dependency wiring and fail-closed scientific gates; no scheduler or GPU execution."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch
import uuid

from capstone_person4.provenance import PipelineError, digest, write_json

REPO = Path(__file__).resolve().parents[2]
SOL = REPO / "scripts/sol"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")
spec = importlib.util.spec_from_file_location("c1_batch_test_helper", SOL / "c1_stage.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@unittest.skipUnless(BASH, "Bash required")
class C1BatchTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("c1-batch-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.python = self.root / "python.sh"
        self.python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n', encoding="utf-8", newline="\n")
        self.python.chmod(0o755)
        self.log = self.root / "calls.log"
        stub = self.root / "bin"
        stub.mkdir()
        sbatch = stub / "sbatch"
        sbatch.write_text('#!/bin/bash\nprintf "%s " "$@" >> "$C1_TEST_LOG"\nprintf "\\n" >> "$C1_TEST_LOG"\n'
                          'echo "$(( $(wc -l < "$C1_TEST_LOG") + 100 ));sol"\n', encoding="utf-8", newline="\n")
        sbatch.chmod(0o755)
        self.env = {**os.environ, "C0_REPO": REPO.as_posix(), "C0_PYTHON": self.python.as_posix(),
                    "PATH": stub.as_posix() + os.pathsep + os.environ["PATH"], "C1_TEST_LOG": self.log.as_posix()}
        self.cfg = "configs/c0.v4.1.5b.seed123.json"
        self.predictions = self.root / "predictions with spaces.jsonl"
        self.report = self.root / "config.json"
        self.predictions.write_text("{}\n", encoding="utf-8")
        self.report.write_text("{}", encoding="utf-8")

    def pipeline(self, **env):
        return subprocess.run([BASH, (SOL / "submit_c1_pipeline.sh").as_posix(), self.cfg, "q15-s123-c1",
                               self.predictions.as_posix(), self.report.as_posix(), "c0full01", "c1full01", "c1smoke01"],
                              cwd=REPO, env={**self.env, **env}, capture_output=True, text=True)

    def test_default_chain_has_three_seeds_and_all_prerequisite_dependencies(self):
        result = self.pipeline(C1_SBATCH_ARGS="--mail-type=FAIL")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 7)
        self.assertIn("c1_ingest.sbatch", calls[0])
        self.assertNotIn("--dependency", calls[0])
        self.assertIn("afterok:101", calls[1])
        self.assertIn("c1_select.sbatch", calls[1])
        self.assertIn("afterok:102", calls[2])
        self.assertIn("c1_train_smoke.sbatch", calls[2])
        for call, seed in zip(calls[3:6], (42, 137, 2026)):
            self.assertIn("afterok:103", call)
            self.assertIn("c1_train.sbatch", call)
            self.assertIn(f"--training-seed {seed}", call)
            self.assertIn("--c0-attempt c0full01 --attempt c1full01", call)
        self.assertIn("afterok:104:105:106", calls[6])
        self.assertIn("--seeds 42 137 2026", calls[6])
        self.assertTrue(all("--mail-type=FAIL" in call for call in calls))
        self.assertIn("finalize=107", result.stdout)
        self.assertIn("preflight", result.stdout)
        self.assertFalse(any("score_reference.sbatch" in c or "c0-train" in c for c in calls))

    def test_failed_preflight_or_invalid_seeds_submit_nothing(self):
        for seeds in ("42 42", "42 wrong", "1 2 3 4 5 6 7"):
            result = self.pipeline(C1_SEEDS=seeds)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(self.log.exists())
        self.python.write_text("#!/bin/bash\nexit 2\n", encoding="utf-8", newline="\n")
        result = self.pipeline()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_spooled_wrappers_forward_profiles_seeds_and_smoke(self):
        for name, args, expected in (
            ("c1_export", ["--profile", "1.5B"], "export-generation-request"),
            ("c1_ingest", [self.predictions.as_posix(), self.report.as_posix()], "ingest-generation"),
            ("c1_select", [], "select"),
            ("c1_train_smoke", ["--attempt", "c1smoke", "--training-seed", "137"], "train"),
            ("c1_train", ["--attempt", "c1full", "--training-seed", "2026"], "train"),
            ("c1_finalize", ["--seeds", "42", "137", "2026"], "finalize"),
        ):
            with self.subTest(script=name):
                copied = self.root / (name + ".sbatch")
                copied.write_bytes((SOL / (name + ".sbatch")).read_bytes())
                result = subprocess.run([BASH, copied.as_posix(), self.cfg, *args], cwd=self.root,
                                        env=self.env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()[1:]
                self.assertIn(expected, lines)
                self.assertEqual("--smoke" in lines, name == "c1_train_smoke")
                for arg in args:
                    self.assertIn(arg, lines)


class C1GateTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("c1-gates-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.cfg = {"data": {"universe_mode": "explicit", "candidate_count": 6409}, "training": {"seed": 42}}

    def test_failed_diagnostics_preserve_evidence_without_selecting(self):
        diagnostics = {"proceed_to_training": False, "degeneracy_flags": {"mistakes_rare": True}}
        with patch.object(helper, "load_prepared", return_value=([], {})), \
                patch.object(helper, "load_scores", return_value=([], {})), \
                patch.object(helper.gs, "correctness_diagnostics", return_value=diagnostics), \
                patch.object(helper, "gate_bindings", return_value={"generation_key": "fixture"}), \
                patch.object(helper, "artifact_root", return_value=self.root), \
                patch.object(helper, "select") as select:
            with self.assertRaisesRegex(PipelineError, "gate failed"):
                helper.select_c1(self.cfg)
            select.assert_not_called()
            self.assertEqual(len(list((self.root / "diagnostics").glob("*.json"))), 1)
            self.assertFalse((self.root / "selection/C1/batch_gate.json").exists())

    def test_gate_rejects_changed_outcomes_and_changed_selection(self):
        path = self.root / "gate.json"
        selection = {"manifest_sha256": "manifest"}
        write_json(path, {"bindings": {"outcomes": "old"}, "selection_sha256": digest(selection),
                          "diagnostics": {"proceed_to_training": True, "degeneracy_flags": {"rare": False}}})
        with patch.object(helper, "gate_path", return_value=path), \
                patch.object(helper, "load_selection", return_value=([], selection)), \
                patch.object(helper, "gate_bindings", return_value={"outcomes": "new"}):
            with self.assertRaisesRegex(PipelineError, "stale"):
                helper.require_gate(self.cfg)
        with patch.object(helper, "gate_path", return_value=path), \
                patch.object(helper, "load_selection", return_value=([], {"manifest_sha256": "changed"})), \
                patch.object(helper, "gate_bindings", return_value={"outcomes": "old"}):
            with self.assertRaisesRegex(PipelineError, "stale"):
                helper.require_gate(self.cfg)

    def test_failed_gate_never_calls_trainer(self):
        with patch.object(helper, "load_config", return_value=self.cfg), \
                patch.object(helper, "require_gate", side_effect=PipelineError("gate failed")), \
                patch.object(helper, "train") as train, redirect_stderr(io.StringIO()):
            self.assertEqual(helper.main(["train", "--config", "fixture.json"]), 2)
            train.assert_not_called()

    def test_successful_selection_gate_is_required_for_c1_training(self):
        diagnostics = {"proceed_to_training": True, "degeneracy_flags": {"mistakes_rare": False}}
        bindings = {"generation_key": "fixture", "outcomes_sha256": "outcomes"}
        selection = {"method": "C1", "manifest_sha256": "selected"}
        path = self.root / "selection/C1/batch_gate.json"
        with patch.object(helper, "load_prepared", return_value=([], {})), \
                patch.object(helper, "load_scores", return_value=([], {})), \
                patch.object(helper.gs, "correctness_diagnostics", return_value=diagnostics), \
                patch.object(helper, "gate_bindings", return_value=bindings), \
                patch.object(helper, "artifact_root", return_value=self.root), \
                patch.object(helper, "select", return_value=selection) as select, \
                patch.object(helper, "load_selection", return_value=([], selection)), \
                patch.object(helper, "gate_path", return_value=path), \
                patch.object(helper, "load_config", return_value=self.cfg), \
                patch.object(helper, "c0_run"), \
                patch.object(helper, "train", return_value={"method": "C1"}) as train, \
                redirect_stdout(io.StringIO()):
            helper.select_c1(self.cfg)
            select.assert_called_once_with(self.cfg, "C1")
            self.assertEqual(helper.require_gate(self.cfg)["selection_sha256"], digest(selection))
            self.assertEqual(helper.main(["train", "--config", "fixture.json", "--attempt", "c1smoke", "--smoke"]), 0)
            train.assert_called_once_with(self.cfg, "C1", "c1smoke", smoke=True)

    def test_c0_precondition_rejects_a_different_seed_or_changed_checkpoint(self):
        cfg = {**self.cfg, "parent": {"adapter_sha256": "parent"}}
        selection = {"actual_tokens": 100}
        report = {"method": "C0", "valid_controlled_run": True, "training_smoke_only": False,
                  "synthetic": False, "parent_unchanged": True, "stage_fingerprint": "stage",
                  "parent_fingerprint": "parent", "selection_sha256": digest(selection), "continuation_seed": 42,
                  "run_config_sha256": "config", "training_summary_sha256": "summary",
                  "checkpoint": "fixture-final", "checkpoint_sha256": "adapter"}
        summary = {"complete": True, "processed_tokens": 100}
        config = {"seed": 42, "arm": "C0"}

        def read(path):
            return {"token_reconciliation.json": report, "training_summary.json": summary, "run_config.json": config}[Path(path).name]

        with patch.object(helper, "artifact_root", return_value=self.root), \
                patch.object(helper, "load_selection", return_value=([], selection)), \
                patch.object(helper, "stage_fingerprint", return_value="stage"), \
                patch.object(helper, "read_json", side_effect=read), \
                patch.object(helper, "verified") as verified:
            self.assertEqual(helper.c0_run(cfg, "c0full"), self.root / "runs/C0/c0full")
            self.assertEqual(verified.call_count, 3)
            report["continuation_seed"] = 137
            with self.assertRaisesRegex(PipelineError, "Matching completed C0"):
                helper.c0_run(cfg, "c0full")
            report["continuation_seed"] = 42
            verified.side_effect = PipelineError("checkpoint changed")
            with self.assertRaisesRegex(PipelineError, "checkpoint changed"):
                helper.c0_run(cfg, "c0full")

    def test_preflight_checks_every_c0_seed_and_writes_nothing(self):
        predictions, report = self.root / "predictions.jsonl", self.root / "config.json"
        predictions.write_text("{}\n", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")

        def config(path, profile=None, training_seed=None):
            return {**self.cfg, "training": {"seed": training_seed or 42}}

        with patch.object(helper, "load_config", side_effect=config), \
                patch.object(helper, "load_prepared", return_value=([], {})), \
                patch.object(helper, "load_scores", return_value=([], {})), \
                patch.object(helper, "load_selection", return_value=([], {})), \
                patch.object(helper.gs, "mistake_bonus", return_value=1), \
                patch.object(helper.gs, "generation_root", return_value=self.root), \
                patch.object(helper.gs, "request_name", return_value="parent17"), \
                patch.object(helper.gs, "generation_key", return_value="request"), \
                patch.object(helper, "read_json", return_value={"generation_key": "request"}), \
                patch.object(helper, "c0_run") as c0, \
                patch.object(helper, "write_json") as write, redirect_stdout(io.StringIO()):
            result = helper.main(["preflight", "--config", "fixture.json", "--predictions", str(predictions), "--report", str(report)])
            self.assertEqual(result, 0)
            self.assertEqual([call.args[0]["training"]["seed"] for call in c0.call_args_list], [42, 137, 2026])
            write.assert_not_called()

    def test_finalize_requires_c1_and_uses_each_matching_c0_seed(self):
        def config(path, profile=None, training_seed=None):
            return {**self.cfg, "training": {"seed": training_seed or 42}}

        def root(cfg):
            return self.root / str(cfg["training"]["seed"])

        with patch.object(helper, "load_config", side_effect=config), \
                patch.object(helper, "artifact_root", side_effect=root), \
                patch.object(helper, "require_gate"), \
                patch.object(helper, "c0_run", side_effect=lambda cfg, attempt: root(cfg) / "runs/C0" / attempt), \
                patch.object(helper, "finalize", return_value={"evaluation_status": "pending"}) as finalize, \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            arguments = ["finalize", "--config", "fixture.json", "--c0-attempt", "c0full", "--attempt", "c1full"]
            self.assertEqual(helper.main(arguments), 2)
            finalize.assert_not_called()
            for seed in (42, 137, 2026):
                write_json(self.root / str(seed) / "runs/C1/c1full/token_reconciliation.json", {})
            self.assertEqual(helper.main(arguments), 0)
            self.assertEqual([call.args[0]["training"]["seed"] for call in finalize.call_args_list], [42, 137, 2026])
            for call in finalize.call_args_list:
                cfg, b_run, c0, c1 = call.args
                self.assertIsNone(b_run)
                self.assertEqual(c0, root(cfg) / "runs/C0/c0full")
                self.assertEqual(c1, root(cfg) / "runs/C1/c1full")

    def test_attempt_paths_are_rejected_before_loading_inputs(self):
        with patch.object(helper, "load_config") as load, redirect_stderr(io.StringIO()):
            self.assertEqual(helper.main(["train", "--config", "fixture.json", "--attempt", "../other"]), 2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

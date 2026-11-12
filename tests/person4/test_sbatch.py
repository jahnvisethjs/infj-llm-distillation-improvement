"""Slurm resource caps, spool-safe dispatch and independent training routing."""

from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch
import uuid

REPO = Path(__file__).resolve().parents[2]
SOL = REPO / "scripts/sol"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")


def directives(path):
    return dict(line.removeprefix("#SBATCH --").split("=", 1)
                for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#SBATCH --"))


class ResourceRequestTests(unittest.TestCase):
    def test_class_account_resources_and_parallel_gpu_budget(self):
        files = list(SOL.glob("*.sbatch"))
        self.assertEqual({p.stem for p in files}, {
            "prepare", "score_smoke", "score_reference", "select", "train_smoke", "train", "finalize",
            "c1_export", "c1_ingest", "c1_select", "c1_train_smoke", "c1_train", "c1_finalize",
            "e_select", "e_train_smoke", "e_train", "e_finalize"})
        for path in files:
            with self.subTest(script=path.name):
                fields = directives(path)
                self.assertEqual(fields["account"], "class_fse57081354fall2026")
                # docs.rc.asu.edu/partitions-and-qos: the class QoS is accepted only on public.
                self.assertEqual(fields["partition"], "public")
                self.assertEqual(fields["qos"], "class")
                self.assertEqual(fields["cpus-per-task"], "4")
                self.assertEqual(fields["mem"], "32G")
                self.assertEqual(fields["nodes"], "1")
                self.assertEqual(fields["ntasks"], "1")
                self.assertNotIn("gres", fields)
                if "gpus" in fields:
                    self.assertEqual(fields["gpus"], "a100:1")
                    hours, minutes, seconds = map(int, fields["time"].split(":"))
                    requested_minutes = hours * 60 + minutes + seconds / 60
                    self.assertLessEqual(2 * requested_minutes, 960)
                self.assertNotIn("/scratch/", path.read_text(encoding="utf-8"))


@unittest.skipUnless(BASH, "Bash is required for batch script tests")
class BatchRoutingTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("slurm-spool-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.python = self.root / "python.sh"
        self.python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n', encoding="utf-8", newline="\n")
        self.python.chmod(0o755)
        self.environment = {**os.environ, "C0_PYTHON": self.python.as_posix(), "C0_REPO": REPO.as_posix(),
                            "SLURM_SUBMIT_DIR": REPO.as_posix(), "SLURM_JOB_ID": "fixture-job", "SLURM_CPUS_PER_TASK": "4"}
        self.cfg = "configs/c0.v4.1.5b.seed123.json"

    def invoke(self, script, *args):
        # Execute a copied script from a foreign working directory, as Slurm does.
        copied = self.root / script
        copied.write_bytes((SOL / script).read_bytes())
        return subprocess.run([BASH, copied.as_posix(), self.cfg, *args], cwd=self.root,
                              env=self.environment, capture_output=True, text=True)

    def test_scoring_smoke_and_full_pass_dispatch_from_slurm_spool(self):
        for name, smoke in (("score_smoke.sbatch", True), ("score_reference.sbatch", False)):
            with self.subTest(script=name):
                result = self.invoke(name, "1.5B")
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.splitlines()[1:]
                self.assertIn("score-reference", arguments)
                self.assertEqual("--smoke" in arguments, smoke)
                self.assertEqual(arguments[-2:], ["--profile", "1.5B"])

    def test_selection_runs_both_arms_in_cpu_job(self):
        result = self.invoke("select.sbatch", "1.5B")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertEqual(arguments.count("diagnose"), 1)
        self.assertEqual(arguments.count("select"), 2)
        self.assertEqual(arguments.count("B"), 1)
        self.assertEqual(arguments.count("C0"), 1)

    def test_preparation_audits_then_prepares_without_a_gpu_stage(self):
        result = self.invoke("prepare.sbatch", "1.5B")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertEqual(arguments.count("audit"), 1)
        self.assertEqual(arguments.count("prepare"), 1)
        self.assertNotIn("score-reference", arguments)

    def test_sol_configuration_generation_only_creates_the_requested_matrix(self):
        env = {**self.environment, "C0_DATA": "/fixture/v4", "C0_PARENTS": "/fixture/parents",
               "C0_INHERITED": "/fixture/inherited", "C0_OUTPUT": "/fixture/results"}
        result = subprocess.run([BASH, (SOL / "configure_sol.sh").as_posix()], cwd=self.root,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertEqual(arguments.count("capstone_person4.v4_config"), 4)
        for profile in ("0.5B", "1.5B"):
            self.assertEqual(arguments.count(profile), 2)
        for seed in ("17", "123"):
            self.assertEqual(arguments.count(seed), 2)
        self.assertNotIn("sbatch", result.stdout)

    def test_finalization_routes_to_same_attempt_helper(self):
        result = self.invoke("finalize.sbatch", "pilot01", "1.5B")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertIn("scripts/sol/finalize_attempt.py", arguments)
        self.assertEqual(arguments[arguments.index("--attempt") + 1], "pilot01")

    def test_training_smoke_is_distinct_from_full_independent_arms(self):
        smoke = self.invoke("train_smoke.sbatch", "smoke01", "1.5B")
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertIn("--smoke", smoke.stdout.splitlines())
        for method in ("B", "C0"):
            with self.subTest(method=method):
                result = self.invoke("train.sbatch", method, "pilot01", "1.5B")
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.splitlines()
                self.assertEqual(arguments[arguments.index("--method") + 1], method)
                self.assertNotIn("--smoke", arguments)
                self.assertNotIn("--resume", arguments)

    def test_resume_forwards_only_explicit_own_arm_checkpoint(self):
        result = self.invoke("train.sbatch", "C0", "resume01", "1.5B", "/fixture/C0/checkpoint-100")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-2:], ["--resume", "/fixture/C0/checkpoint-100"])

    def test_pipeline_chain_submits_eight_dependent_jobs_once(self):
        stub = self.root / "bin"
        stub.mkdir()
        # Record every sbatch call and return increasing fake job IDs (101, 102, ...).
        stub_lines = ["#!/bin/bash",
                      'printf "%s " "$@" >> "$C0_TEST_LOG"; printf "\\n" >> "$C0_TEST_LOG"',
                      'echo "$(( $(wc -l < "$C0_TEST_LOG") + 100 ))"', ""]
        (stub / "sbatch").write_text("\n".join(stub_lines), encoding="utf-8", newline="\n")
        (stub / "sbatch").chmod(0o755)
        log = self.root / "sbatch.log"
        env = {**self.environment, "PATH": stub.as_posix() + os.pathsep + os.environ["PATH"], "C0_TEST_LOG": log.as_posix(),
               "C0_SBATCH_ARGS": "--mail-user=fixture@example.edu"}
        result = subprocess.run([BASH, (SOL / "submit_pipeline.sh").as_posix(), self.cfg, "q15-s123"],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = log.read_text(encoding="utf-8").splitlines()
        # C0 only by default, one training job per continuation seed (42, 137, 2026)
        # after a single smoke, one finalize for all seeds: nine jobs, under the QoS limit of ten.
        self.assertEqual(len(calls), 9)
        self.assertIn("prepare.sbatch", calls[0])
        self.assertNotIn("--dependency", calls[0])
        self.assertIn("--dependency afterok:101", calls[1])
        c0_jobs = [c for c in calls if "train.sbatch" in c and " C0 " in c]
        self.assertEqual(len(c0_jobs), 3)
        self.assertEqual([c.split()[-1] for c in c0_jobs], ["42", "137", "2026"])
        self.assertTrue(all("afterok:105" in c for c in c0_jobs))
        self.assertFalse(any("train.sbatch" in c and " B " in c for c in calls))
        self.assertIn("afterok:106:107:108", calls[8])
        self.assertTrue(all("--mail-user=fixture@example.edu" in c and self.cfg in c for c in calls))
        self.assertIn("finalize=109", result.stdout)
        log.write_text("", encoding="utf-8")
        result = subprocess.run([BASH, (SOL / "submit_pipeline.sh").as_posix(), self.cfg, "q15-s123"],
                                cwd=self.root, env={**env, "C0_ARMS": "B C0", "C0_SEEDS": "42"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 8)
        self.assertEqual(sum("train.sbatch" in c for c in calls), 2)
        self.assertIn("afterok:106:107", calls[7])
        # Both arms at three seeds would be twelve jobs: refused before any submission.
        log.write_text("", encoding="utf-8")
        result = subprocess.run([BASH, (SOL / "submit_pipeline.sh").as_posix(), self.cfg, "q15-s123"],
                                cwd=self.root, env={**env, "C0_ARMS": "B C0"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("ten active jobs", result.stderr)
        self.assertEqual(log.read_text(encoding="utf-8"), "")

    def test_invalid_method_fails_before_python_runs(self):
        result = self.invoke("train.sbatch", "parent", "pilot01", "1.5B")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("capstone_person4", result.stdout)


class FinalizeAttemptTests(unittest.TestCase):
    def test_resolves_actual_training_stage_paths_and_rejects_path_traversal(self):
        spec = importlib.util.spec_from_file_location("c0_test_finalize_attempt", SOL / "finalize_attempt.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = {"fixture": "bound-configuration"}
        root = REPO / ".test_artifacts" / ("finalize-stage-" + uuid.uuid4().hex)
        (root / "runs/C0/pilot01").mkdir(parents=True)
        for b_exists, c1_exists in ((False, False), (True, False), (True, True)):  # directories accumulate
            if b_exists:
                (root / "runs/B/pilot01").mkdir(parents=True, exist_ok=True)
            if c1_exists:
                (root / "runs/C1/pilot01").mkdir(parents=True, exist_ok=True)
            with patch.object(module, "load_config", return_value=cfg), \
                    patch.object(module, "artifact_root", return_value=root), \
                    patch.object(module, "finalize", return_value={"evaluation_status": "pending"}) as finalize, \
                    patch("sys.argv", ["finalize_attempt.py", "--config", "fixture.json", "--attempt", "pilot01"]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 0)
                # B and C1 are passed only when same-named runs were trained here; otherwise C0-only.
                finalize.assert_called_once_with(cfg, root / "runs/B/pilot01" if b_exists else None, root / "runs/C0/pilot01",
                                                 root / "runs/C1/pilot01" if c1_exists else None)
        with patch.object(module, "load_config") as load, \
                patch("sys.argv", ["finalize_attempt.py", "--config", "fixture.json", "--attempt", "../other-arm"]), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(module.main(), 2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

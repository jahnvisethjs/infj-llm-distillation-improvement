"""Model/parent matrix binding and launcher routing without GPU execution."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from capstone_person4.config import artifact_root, load_config
from capstone_person4.provenance import PipelineError
from capstone_person4.v4_config import configuration

REPO = Path(__file__).resolve().parents[2]
PARENTS = REPO.parent / "R1000_parents_seeds17_123_20261101T023416637853Z"
COMBINATIONS = (("0.5B", 17), ("0.5B", 123), ("1.5B", 17), ("1.5B", 123))
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")


def config_path(profile, seed):
    return REPO / "configs" / f"c0.v4.{profile.lower()}.seed{seed}.json"


class GeneratorValidationTests(unittest.TestCase):
    def test_rejects_unknown_profile_and_parent_seed(self):
        arguments = ("release", "inherited", "parents", "outputs", 17, "notebook")
        with self.assertRaisesRegex(PipelineError, "Unknown model profile"):
            configuration(*arguments, profile="3B")
        with self.assertRaisesRegex(PipelineError, "parent training seed"):
            configuration(*arguments[:4], 42, arguments[-1])


@unittest.skipUnless(PARENTS.is_dir() and (REPO.parent / "data").is_dir(), "Optional supplied-parent integration")
class ModelProfileTests(unittest.TestCase):
    def test_all_four_explicit_configs_bind_correct_saved_parents_and_isolate_measurements(self):
        from capstone_person4.audit import check_weights
        from capstone_person4.reference_scoring import measurement_key
        roots, keys, weights = set(), set(), set()
        preparation = {"candidate_sha256": "common-candidate-content", "tokenizer_fingerprint": "same-tokenizer",
                       "evidence": {"release": {"format_sha256": "same-format"}}}
        for profile, seed in COMBINATIONS:
            with self.subTest(profile=profile, parent_seed=seed):
                cfg = load_config(config_path(profile, seed))
                self.assertEqual(cfg["active_profile"], profile)
                self.assertEqual(cfg["parent"]["model"], "Qwen/Qwen2.5-" + profile)
                self.assertEqual(cfg["parent"]["training_seed"], seed)
                self.assertIn(f"R1000_seed{seed}", cfg["parent"]["adapter"])
                self.assertEqual(cfg["data"]["candidate_count"], 5000)
                self.assertEqual(cfg["training"]["seed"], 42)
                self.assertEqual(check_weights(cfg)["processed_tokens"], 318845)
                roots.add(artifact_root(cfg))
                keys.add(measurement_key(cfg, preparation, runtime={"test": "CPU identity check"}))
                weights.add(cfg["parent"]["adapter_sha256"])
        self.assertEqual(len(roots), 4)
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(weights), 4)

    def test_generator_supports_every_requested_active_model_and_seed(self):
        for profile, seed in COMBINATIONS:
            with self.subTest(profile=profile, parent_seed=seed):
                cfg = configuration(REPO.parent / "data", REPO.parent / "capstone_person1/releases/person1_v3_20261012_ready",
                                    PARENTS, REPO / "outputs/person4/v4", seed,
                                    REPO / "Person2_SOL_Seeds17_123_UpTo4000.ipynb", profile=profile)
                self.assertEqual(cfg["active_profile"], profile)
                self.assertEqual(cfg["profiles"][profile]["training_seed"], seed)
                self.assertEqual(cfg["profiles"][profile]["adapter_sha256"], load_config(config_path(profile, seed))["parent"]["adapter_sha256"])

    def test_1_5b_encoder_rejects_different_reference_token_ids_even_when_counts_match(self):
        from capstone_person4.person2_adapter import Person2Adapter
        cfg = load_config(config_path("1.5B", 17))
        adapter = Person2Adapter(cfg)
        selected_tokenizer, reference_tokenizer = object(), object()
        def encoded(record, tokenizer, **kwargs):
            token = 22 if tokenizer is selected_tokenizer else 23
            return {"input_ids": [11, token, 151643], "labels": [-100, token, 151643],
                    "attention_mask": [1, 1, 1], "prompt_tokens": 1, "completion_tokens": 1,
                    "supervised_tokens": 2, "total_tokens": 3}
        def padded(record, width):
            padding = width - len(record["input_ids"])
            return {"input_ids": record["input_ids"] + [151643] * padding,
                    "labels": record["labels"] + [-100] * padding,
                    "attention_mask": record["attention_mask"] + [0] * padding}
        with patch.object(adapter.contract, "validate_manifest_record", side_effect=encoded), \
                patch.object(adapter.contract, "pad_encoded", side_effect=padded), \
                patch.object(adapter.wrapper, "model_tokenizer", return_value=reference_tokenizer):
            with self.assertRaisesRegex(ValueError, "1.5B tokenizer differs"):
                adapter.encode({"row_id": "fixture", "supervised_tokens": 2}, selected_tokenizer)


@unittest.skipUnless(BASH, "Bash is required for SOL launcher routing tests")
class LauncherRoutingTests(unittest.TestCase):
    def run_shell(self, source, *arguments, environment=None):
        return subprocess.run([BASH, "-c", source, "c0-launcher-test", *map(str, arguments)],
                              cwd=REPO, env={**os.environ, **(environment or {})}, capture_output=True, text=True)

    def test_gpu_launcher_routes_each_explicit_model_without_loading_a_model(self):
        source = '''
c0_record_python() { printf '%s\\n' "$@"; }
export -f c0_record_python
export C0_PYTHON=c0_record_python
bash scripts/sol/gpu_stage.sh "$@"
'''
        for profile, seed in COMBINATIONS:
            with self.subTest(profile=profile, parent_seed=seed):
                cfg = config_path(profile, seed).relative_to(REPO).as_posix()
                result = self.run_shell(source, cfg, "scoring-smoke", "", profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["-B", "-m", "capstone_person4", "score-reference", "--smoke",
                                                             "--config", cfg, "--profile", profile])

    def test_submit_forwards_model_and_dependency_with_distinct_job_name(self):
        source = '''
sbatch() { printf '%s\\n' "$@"; }
export -f sbatch
bash scripts/sol/submit.sh "$@"
'''
        env = {"C0_PYTHON": Path(sys.executable).as_posix(), "C0_ACCOUNT": "fixture-account",
               "C0_PARTITION": "fixture-partition", "C0_WALLTIME": "00:01:00", "C0_MEMORY": "1G", "C0_GRES": "gpu:1"}
        config = "configs/c0.v4.1.5b.seed123.json"
        result = self.run_shell(source, config, "train-B", "fixture-attempt", "12345", "1.5B", environment=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertEqual(arguments[arguments.index("--dependency") + 1], "afterok:12345")
        self.assertIn("c0.v4.1.5b.seed123", arguments[arguments.index("--job-name") + 1])
        self.assertEqual(arguments[-5:], ["scripts/sol/gpu_stage.sh", config, "train-B", "fixture-attempt", "1.5B"])

    def test_invalid_model_is_rejected_before_python_or_scheduler_runs(self):
        source = '''
c0_record_python() { printf 'unexpected-python'; }
export -f c0_record_python
export C0_PYTHON=c0_record_python
bash scripts/sol/gpu_stage.sh "$@"
'''
        result = self.run_shell(source, "configs/c0.v4.seed17.json", "scoring-smoke", "", "3B")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("unexpected-python", result.stdout)
        self.assertIn("Unsupported model profile", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""CPU scientific-integrity regressions; no claim of real Qwen execution."""
import ast
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import screened_c0 as s


def ordered(records):
    def key(r):
        value = hashlib.sha256(("42|training-order|" + r["row_id"]).encode()).digest()
        return int.from_bytes(value[:4], "big"), r["row_id"]
    return sorted(records, key=key)


def synthetic_rows():
    parent, selected = [], []
    for i in range(1000):
        common = {"dataset_id": "orca" if i < 500 else "numina", "length_bin": (i % 500) // 50,
                  "supervised_tokens": 318 + (i < 845), "problem": "SYNTHETIC question " + str(i),
                  "completion": "SYNTHETIC reference " + str(i)}
        parent.append(dict(common, row_id=f"parent_{i:04d}", problem_group_id=f"P_{i:04d}"))
        selected.append(dict(common, row_id=f"new_{i:04d}", problem_group_id=f"N_{i:04d}"))
    return parent, ordered(selected)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.selected = synthetic_rows()

    def test_conforming_counts_tokens_and_order_pass(self):
        self.assertEqual(s.validate_selection(self.selected, self.selected, self.parent), 318845)

    def test_reference_change_under_same_id_is_refused(self):
        changed = deepcopy(self.selected)
        changed[0]["completion"] += " altered"
        with self.assertRaisesRegex(ValueError, "membership/count/content"):
            s.validate_selection(changed, self.selected, self.parent)

    def test_parent_seen_group_is_refused(self):
        changed = deepcopy(self.selected)
        changed[0]["problem_group_id"] = self.parent[0]["problem_group_id"]
        with self.assertRaisesRegex(ValueError, "intersects parent"):
            s.validate_selection(changed, changed, self.parent)

    def test_global_budget_cannot_be_relaxed(self):
        changed = deepcopy(self.selected)
        for row in changed:
            row["supervised_tokens"] += 2
        with self.assertRaisesRegex(ValueError, "global tokens"):
            s.validate_selection(changed, changed, self.parent)

    def test_matching_total_does_not_hide_stratum_violation(self):
        changed = deepcopy(self.selected)
        for row in changed:
            if row["dataset_id"] == "orca" and row["length_bin"] == 0:
                row["supervised_tokens"] += 30
            elif row["dataset_id"] == "orca" and row["length_bin"] == 1:
                row["supervised_tokens"] -= 30
        self.assertEqual(sum(r["supervised_tokens"] for r in changed), s.TARGET)
        with self.assertRaisesRegex(ValueError, "stratum tokens"):
            s.validate_selection(changed, changed, self.parent)

    def test_quota_change_is_refused(self):
        changed = deepcopy(self.selected)
        changed[0]["length_bin"] = (changed[0]["length_bin"] + 1) % 10
        with self.assertRaisesRegex(ValueError, "50 rows"):
            s.validate_selection(changed, changed, self.parent)

    def test_changing_training_order_is_refused(self):
        changed = list(reversed(self.selected))
        with self.assertRaisesRegex(ValueError, "training order"):
            s.validate_selection(changed, changed, self.parent)


class ConsumptionTests(unittest.TestCase):
    def setUp(self):
        self.selected = synthetic_rows()[1][:32]
        tokens = sum(r["supervised_tokens"] for r in self.selected)
        self.summary = {"complete": True, "processed_tokens": tokens, "unique_examples": 32,
                        "unique_supervised_tokens": tokens, "optimizer_updates": 4,
                        "parent_optimizer_updates": 0,
                        "seen_ids": [r["row_id"] for r in self.selected],
                        "seen_groups": [r["problem_group_id"] for r in self.selected]}
        cumulative = 0
        self.logs = []
        for i in range(4):
            count = sum(r["supervised_tokens"] for r in self.selected[i*8:(i+1)*8])
            cumulative += count
            self.logs.append({"target_tokens": count, "optimizer_update": i+1,
                              "cumulative_processed_tokens": cumulative, "lr": 1e-4*(i+1)/10,
                              "loss": 1.2, "grad_norm": 0.3})

    def test_one_complete_pass_reconciles(self):
        self.assertEqual(s.validate_consumption(self.summary, self.selected, self.logs),
                         self.summary["processed_tokens"])

    def test_incomplete_training_cannot_pass(self):
        self.summary["complete"] = False
        with self.assertRaisesRegex(ValueError, "did not reconcile"):
            s.validate_consumption(self.summary, self.selected, self.logs)

    def test_wrong_ids_cannot_pass_with_correct_token_total(self):
        self.summary["seen_ids"][0] = "another_example"
        with self.assertRaisesRegex(ValueError, "did not reconcile"):
            s.validate_consumption(self.summary, self.selected, self.logs)

    def test_parent_optimizer_restore_is_refused(self):
        self.summary["parent_optimizer_updates"] = 125
        with self.assertRaisesRegex(ValueError, "optimizer isolation"):
            s.validate_consumption(self.summary, self.selected, self.logs)

    def test_skipped_warmup_is_refused(self):
        self.logs[0]["lr"] = 1e-4
        with self.assertRaisesRegex(ValueError, "warmup"):
            s.validate_consumption(self.summary, self.selected, self.logs)

    def test_nonfinite_loss_is_refused(self):
        self.logs[0]["loss"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite-loss"):
            s.validate_consumption(self.summary, self.selected, self.logs)

    def test_partial_log_is_refused(self):
        with self.assertRaisesRegex(ValueError, "complete token budget"):
            s.validate_consumption(self.summary, self.selected, self.logs[:-1])


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.questions = [{"problem_id": str(i), "benchmark": "gsm"} for i in range(500)]
        self.config = {"eval_config_hash": "CONFIG", "parser_version": "PARSER"}
        self.predictions = [{"problem_id": str(i), "benchmark": "gsm", "correct": i % 2 == 0,
                             "checkpoint_sha256": "CHECKPOINT", "train_manifest_sha256": "MANIFEST",
                             "eval_config_hash": "CONFIG", "parser_version": "PARSER", "parse_status": "ok"}
                            for i in range(500)]

    def validate(self):
        return s.validate_prediction_rows(self.predictions, self.questions, self.config, "CHECKPOINT", "MANIFEST")

    def test_complete_predictions_pass(self):
        self.assertEqual(sum(self.validate().values()), 250)

    def test_missing_or_duplicate_question_is_refused(self):
        self.predictions[-1] = deepcopy(self.predictions[0])
        with self.assertRaisesRegex(ValueError, "500 frozen questions"):
            self.validate()

    def test_mixed_checkpoint_is_refused(self):
        self.predictions[0]["checkpoint_sha256"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "mixed checkpoint"):
            self.validate()

    def test_checker_error_is_not_a_wrong_answer(self):
        self.predictions[0].update(parse_status="verify_error", correct=False)
        with self.assertRaisesRegex(ValueError, "execution errors"):
            self.validate()

    def test_boolean_correctness_is_required(self):
        self.predictions[0]["correct"] = 1
        with self.assertRaisesRegex(ValueError, "mixed checkpoint"):
            self.validate()

    def test_pairing_retains_improvements_and_regressions(self):
        before = dict(a=True, b=False, c=True, d=False)
        after = dict(a=True, b=True, c=False, d=False)
        report = s.paired_difference(after, before, resamples=100)
        self.assertEqual(report["difference_pp"], 0)
        self.assertEqual(report["paired_outcomes"], {"retained_success": 1, "improvement": 1,
                                                    "regression": 1, "persistent_failure": 1})

    def test_pairing_different_ids_is_refused(self):
        with self.assertRaisesRegex(ValueError, "identical nonempty IDs"):
            s.paired_difference({"a": True}, {"b": False})


class CompletionGateTests(unittest.TestCase):
    def test_smoke_receipt_cannot_pass_as_full_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            c = SimpleNamespace(out=Path(temporary), profile=s.PROFILE, identity="a"*64,
                                entry={"expected_adapter_sha256": "PARENT"})
            selection = {"example": "synthetic selection receipt"}
            selected = synthetic_rows()[1]
            directory = s.run_directory(c, "C0", "pilot01")
            receipt = {"passed": True, "identity": c.identity, "method": "C0", "seed": 42,
                       "smoke_only": True, "reload_probe_matched": True, "version": s.VERSION,
                       "parent_seed": 17, "bank_sha256": s.BANK_SHA,
                       "selection_sha256": s.digest(selection), "parent_sha256": "PARENT",
                       "screening_protocol_sha256": "PROTOCOL", "unique_examples": 1000,
                       "actual_tokens": 318845, "original_matcher_flags_resolved": False,
                       "changed_adapter_tensors": ["lora_A"],
                       "smoke_key": s.digest({"identity": c.identity, "selection": s.digest(selection),
                                              "method": "C0", "seed": 42})}
            s.save(directory / "completion.json", receipt)
            with patch.object(s, "require_prepared", return_value="PROTOCOL"), \
                    patch.object(s, "load_selection", return_value=(selected, selection)):
                with self.assertRaisesRegex(ValueError, "smoke status"):
                    s.validate_completion(c, directory, "C0", smoke=False)

    def test_training_source_change_is_reporting_only_for_both_methods(self):
        source = "def train():\n label = 'preliminary_document_aligned_B'\n lr = base * min(update / 10.0, 1.0)\n optimizer.step()\n"
        for method in ("B", "C0"):
            changed = s.training_source_for_method(source, method)
            restored = changed.replace(repr("independent_conservative_screen_" + method + "_continuation"),
                                       "'preliminary_document_aligned_B'")
            self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(restored)))

    def test_missing_sixth_evaluation_cannot_create_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            c = SimpleNamespace(out=Path(temporary), profile=s.PROFILE, identity="b"*64)
            outcomes = {str(i): i % 2 == 0 for i in range(500)}
            predictions = [{"parse_status": "ok", "finish_reason": "eos"} for _ in range(500)]
            def validate(context, run, benchmark, attempt):
                if run == "C0" and benchmark == "dev_math":
                    raise ValueError("Incomplete evaluation fixture")
                return outcomes, {}, predictions
            for run in ("parent", "B", "C0"):
                for benchmark in s.BENCHMARKS:
                    path = s.prediction_path(c, run, benchmark, "pilot01")
                    s.save_rows(path, [{"synthetic": True}])
                    s.save(path.with_suffix(".config.json"), {"synthetic": True})
            with patch.object(s, "require_prepared", return_value="PROTOCOL"), \
                    patch.object(s, "validate_completion", return_value={"synthetic": True}), \
                    patch.object(s, "validate_evaluation", side_effect=validate):
                with self.assertRaisesRegex(ValueError, "Incomplete evaluation fixture"):
                    s.summarize(c, "pilot01")
            self.assertFalse(s.artifact(c, "PILOT_COMPLETE_pilot01.json").exists())

    def test_mutating_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            s.save(path, {"run": "original"})
            s.save(path, {"run": "original"})
            with self.assertRaisesRegex(ValueError, "Different output preserved"):
                s.save(path, {"run": "replacement"})


class OrchestrationTests(unittest.TestCase):
    def test_pipeline_orders_smokes_before_full_training_then_six_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            c = SimpleNamespace(out=Path(temporary), profile=s.PROFILE, identity="a"*64,
                                work=Path(temporary), root=Path(temporary))
            with patch.object(s, "require_prepared"), patch.object(s.subprocess, "run") as run, redirect_stdout(io.StringIO()):
                s.pipeline(c, "pilot01")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 11)
            self.assertTrue(all("--smoke" in command for command in commands[:2]))
            self.assertTrue(all("--smoke" not in command and command[3] == "train" for command in commands[2:4]))
            self.assertEqual(sum(command[3] == "evaluate" for command in commands), 6)
            self.assertEqual(commands[-1][3], "summarize")

    def test_sbatch_script_works_when_copied_to_a_scheduler_spool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, spool = root / "uploaded_package", root / "scheduler_spool"
            package.mkdir()
            spool.mkdir()
            (package / "screened_c0.py").write_text("import sys,json\nprint(json.dumps(sys.argv[1:]))\n")
            script = (s.HERE / "gpu_pipeline.sbatch").read_text()
            script = script.replace("/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python", sys.executable)
            (spool / "slurm_script").write_text(script)
            result = subprocess.run(["bash", str(spool / "slurm_script"), str(package),
                                     "--attempt", "fixture"], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout.splitlines()[-1]), ["pipeline", "--attempt", "fixture"])

    def test_help_works_without_model_dependencies(self):
        result = subprocess.run([sys.executable, "-B", str(s.HERE / "screened_c0.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("summarize", result.stdout)


class RealBankEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = s.HERE.parent.parent
        cls.bank_path = cls.root / "protection_evidence/candidate_questions.json"
        if not cls.bank_path.is_file():
            raise unittest.SkipTest("Conversation-local question projection is not beside this upload")
        cls.original = s.read(cls.bank_path)
        # Full references exist only for the targeted sample. Do not represent
        # this projected question bank as the unavailable complete Sol manifest.
        references = {r["row_id"]: r for r in s.read(cls.root / "followup_evidence/targeted_full_candidate_rows.json")}
        for row in cls.original:
            if row["row_id"] in references:
                row["completion"] = references[row["row_id"]]["completion"]
        cls.screen = s.read(s.HERE / "screening/screening_proposal.json")
        cls.decisions = s.read(s.HERE / "screening/known_exclusions.json")["decisions"]
        held = set(cls.screen["held_out_ids"])
        cls.retained = [r for r in cls.original if r["row_id"] not in held]

    def test_question_screen_reproduces_all_4955_ids_and_source_counts(self):
        self.assertTrue(s.validate_population(self.original, self.retained, self.screen, self.decisions))
        projected = [{k: r.get(k) for k in self.screen["projection_fields"]} for r in self.original]
        self.assertEqual(s.digest(projected), self.screen["original_projection_sha256"])
        self.assertEqual(dict(Counter(r["dataset_id"] for r in self.retained)), {"orca": 3313, "numina": 1642})
        self.assertEqual(min(Counter(s.cell(r) for r in self.retained).values()), 148)

    def test_one_quarantined_row_cannot_be_reintroduced(self):
        held = set(self.screen["held_out_ids"])
        row = next(r for r in self.original if r["row_id"] in held)
        with self.assertRaisesRegex(ValueError, "group-level quarantine"):
            s.validate_population(self.original, self.retained + [row], self.screen, self.decisions)

    def test_actual_reference_change_cannot_reuse_known_defect_decision(self):
        altered = deepcopy(self.original)
        decision = next(d for d in self.decisions if d.get("completion_sha256"))
        next(r for r in altered if r["row_id"] == decision["row_id"])["completion"] += " changed"
        with self.assertRaisesRegex(ValueError, "reviewed reference content changed"):
            s.validate_population(altered, self.retained, self.screen, self.decisions)

    def test_inspected_b_update_loop_matches_original_ast_pin(self):
        runtime = self.root / "b_setup_inputs/b_workbench"
        if not (runtime / "continuation.py").is_file():
            self.skipTest("Conversation-local inspected B training source is unavailable")
        source = (runtime / "continuation.py").read_text().replace("'controlled_continuation'", "'preliminary_document_aligned_B'")
        namespace = {"ast": ast, "HERE": runtime}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in ("replace_once", "adapted_train_source"):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<inspected B source>", "exec"), namespace)
        original = namespace["adapted_train_source"]()
        expected = "180edf1c45f4a6654cb9fca6d7c198ea5dc2c7559cda68a48254fa73c02350c2"
        self.assertEqual(hashlib.sha256(ast.dump(ast.parse(original), include_attributes=False).encode()).hexdigest(), expected)
        for method in ("B", "C0"):
            modified = s.training_source_for_method(original, method)
            restored = modified.replace(repr("independent_conservative_screen_" + method + "_continuation"),
                                        "'preliminary_document_aligned_B'")
            self.assertEqual(original, restored)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU integrity/orchestration tests. These never claim real GPU training occurred."""
import ast
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nightly as n
import pilot_checks as checks


def args(root):
    return n.parser().parse_args(["worker", "--work", str(root), "--b-root", str(root / "b"),
        "--worker-seconds", "6000", "--allocations", "3"])


def synthetic():
    parent, bank = [], []
    for i in range(1000):
        common = {"dataset_id": "orca" if i < 500 else "numina", "length_bin": (i % 500) // 50,
            "supervised_tokens": 318 + (i < 845), "problem": "Synthetic question " + str(i),
            "completion": "Synthetic reference " + str(i)}
        parent.append({**common, "row_id": f"p{i}", "problem_group_id": f"pgroup{i}"})
        bank.append({**common, "row_id": f"b{i}", "problem_group_id": f"bgroup{i}"})
    return parent, n.training_order(bank)


class MatrixAndRecipe(unittest.TestCase):
    def test_exact_requested_matrix_no_B_no_benchmark_evaluation(self):
        plan = n.units()
        trained = [u for u in plan if u["stage"] == "train"]
        self.assertEqual(Counter(u["method"] for u in trained), {"C0": 11, "C1": 12})
        actual = {(u["profile"], u["method"], u["seed"]) for u in trained}
        expected = {(f"{m}_seed{p}", method, seed) for m, p in n.PROFILES
                    for method in ("C0", "C1") for seed in n.SEEDS}
        expected.remove(("0.5B_seed17", "C0", 42))
        self.assertEqual(actual, expected)
        self.assertEqual(len(plan), len({u["id"] for u in plan}))
        self.assertEqual(len([u for u in plan if u["stage"] == "generate"]), 40)
        self.assertFalse(any(u["stage"] in ("evaluate", "benchmark") for u in plan))

    def test_dependencies_prevent_training_before_complete_selection(self):
        done = set()
        by_id = {u["id"]: u for u in n.units()}
        for unit in n.units():
            self.assertTrue(set(unit["requires"]).issubset(done))
            if unit["stage"] == "select" and unit["method"] == "C1":
                self.assertEqual(len(unit["requires"]), 10)
                self.assertTrue(all(by_id[d]["stage"] == "generate" for d in unit["requires"]))
            done.add(unit["id"])

    def test_every_candidate_is_generated_once_per_parent(self):
        c = SimpleNamespace(bank=[{"row_id": str(i)} for i in range(4955)])
        chunks = [n.candidate_rows(c, i) for i in range(n.CHUNKS)]
        self.assertEqual([r for chunk in chunks for r in chunk], c.bank)
        self.assertEqual([len(c) for c in chunks], [512] * 9 + [347])

    def test_trainer_update_loop_changes_only_its_reporting_label(self):
        source = "def train():\n label='preliminary_document_aligned_B'\n optimizer.step()\n"
        for method in ("C0", "C1"):
            code = n.training_source_for_method(source, method)
            restored = code.replace(repr("screened_training_only_" + method), "'preliminary_document_aligned_B'")
            self.assertEqual(ast.dump(ast.parse(restored)), ast.dump(ast.parse(source)))
        with self.assertRaises(n.IntegrityError):
            n.training_source_for_method(source, "B")

    def test_exact_strata_tokens_parent_freshness_and_order(self):
        parent, bank = synthetic()
        self.assertEqual(checks.validate_selection(bank, bank, parent), n.TARGET)
        altered = deepcopy(bank)
        altered[0]["problem_group_id"] = parent[0]["problem_group_id"]
        with self.assertRaises(ValueError):
            checks.validate_selection(altered, altered, parent)
        altered = deepcopy(bank)
        for row in altered:
            row["supervised_tokens"] += 2
        with self.assertRaises(ValueError):
            checks.validate_selection(altered, altered, parent)
        with self.assertRaises(ValueError):
            checks.validate_selection(list(reversed(bank)), bank, parent)


class CandidateSemantics(unittest.TestCase):
    def setUp(self):
        # Execute the actual pinned classifier, without importing its unrelated GPU dependencies.
        source = (Path(__file__).parents[2] / "repo/capstone_person4_approachC1/generation_scoring.py")
        if not source.exists():
            source = Path(__file__).parent / "classifier_snapshot.py"
        code = source.read_text()
        node = next(x for x in ast.parse(code).body if isinstance(x, ast.FunctionDef) and x.name == "classify")
        scope = {"PipelineError": n.IntegrityError}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "classifier", "exec"), scope)
        module = SimpleNamespace(classify=scope["classify"])
        self.modules = patch.dict(sys.modules, {"capstone_person4_approachC1": SimpleNamespace(),
            "capstone_person4_approachC1.generation_scoring": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.c = SimpleNamespace(entry={"model": "Qwen/test", "revision": "rev", "expected_adapter_sha256": "parent"},
                                 legacy=SimpleNamespace(PARENT_SHA="training"))
        self.config = {"eval_config_hash": "gen"}

    def prediction(self, identifier, parse="ok", correct=False, finish="eos"):
        return {"problem_id": identifier, "parse_status": parse, "correct": correct, "finish_reason": finish,
            "checkpoint_sha256": "parent", "model_revision": "Qwen/test@rev", "parser_version": "p3-parser-v6",
            "eval_config_hash": "gen", "benchmark": "math", "train_manifest_sha256": "training"}

    def test_only_confirmed_wrong_answers_receive_mistake_bonus(self):
        predictions = [self.prediction("wrong"), self.prediction("right", correct=True),
            self.prediction("missing", "no_answer"), self.prediction("truncated", "no_answer", finish="length"),
            self.prediction("ambiguous", "ambiguous")]
        result = n.classify_predictions(self.c, predictions,
            [{"row_id": p["problem_id"]} for p in predictions], self.config)
        self.assertEqual([r["mistake"] for r in result], [True, False, False, False, False])

    def test_execution_errors_block_instead_of_becoming_mistakes(self):
        with self.assertRaises(RuntimeError):
            n.classify_predictions(self.c, [self.prediction("x", "error")], [{"row_id": "x"}], self.config)

    def test_duplicate_missing_or_wrong_parent_records_are_refused(self):
        p = self.prediction("x")
        for predictions, wanted in (([p, p], [{"row_id": "x"}]), ([p], [{"row_id": "x"}, {"row_id": "y"}])):
            with self.assertRaises(n.IntegrityError):
                n.classify_predictions(self.c, predictions, wanted, self.config)
        p["checkpoint_sha256"] = "different"
        with self.assertRaises(n.IntegrityError):
            n.classify_predictions(self.c, [p], [{"row_id": "x"}], self.config)

    def test_partial_generation_can_resume_without_claiming_complete_coverage(self):
        result = n.classify_predictions(self.c, [self.prediction("x")],
            [{"row_id": "x"}, {"row_id": "y"}], self.config, complete=False)
        self.assertEqual([r["row_id"] for r in result], ["x"])


class Consumption(unittest.TestCase):
    def fixture(self):
        selected = synthetic()[1]
        total = sum(r["supervised_tokens"] for r in selected)
        summary = {"complete": True, "processed_tokens": total, "unique_examples": 1000,
            "unique_supervised_tokens": total, "optimizer_updates": 125, "parent_optimizer_updates": 0,
            "seen_ids": [r["row_id"] for r in selected], "seen_groups": [r["problem_group_id"] for r in selected]}
        logs, cumulative = [], 0
        for i in range(125):
            tokens = sum(r["supervised_tokens"] for r in selected[i*8:(i+1)*8])
            cumulative += tokens
            logs.append({"target_tokens": tokens, "optimizer_update": i+1, "cumulative_processed_tokens": cumulative,
                         "lr": 1e-4 * min((i+1)/10, 1), "loss": 1.2, "grad_norm": 0.2})
        return selected, summary, logs

    def test_one_pass_125_updates_with_fresh_optimizer_passes(self):
        selected, summary, logs = self.fixture()
        self.assertEqual(n.validate_consumption(summary, selected, logs), n.TARGET)

    def test_wrong_seed_exposure_receipt_or_parent_optimizer_cannot_pass(self):
        selected, summary, logs = self.fixture()
        for key, bad in (("complete", False), ("optimizer_updates", 124), ("parent_optimizer_updates", 125),
                         ("processed_tokens", n.TARGET+1)):
            altered = {**summary, key: bad}
            with self.assertRaises(ValueError):
                n.validate_consumption(altered, selected, logs)
        logs[0]["lr"] = 1e-4
        with self.assertRaises(ValueError):
            n.validate_consumption(summary, selected, logs)


class TrainingReceipts(unittest.TestCase):
    def test_1p5B_parent123_C1_seed137_receipt_and_wrong_seed_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = args(root)
            a.model, a.parent_seed = "1.5B", 123
            parent, bank = synthetic()
            parent_adapter = root / "parent/adapter_model.safetensors"
            parent_adapter.parent.mkdir()
            parent_adapter.write_bytes(b"synthetic parent weight placeholder")
            c = SimpleNamespace(a=a, out=root / "outputs", root=root, profile="1.5B_seed123", identity="receipt-test",
                entry={"model":"Qwen/Qwen2.5-1.5B", "revision":"pinned", "expected_adapter_sha256":n.sha(parent_adapter),
                       "adapter_relative_path":"parent"}, train_source="label='preliminary_document_aligned_B'\n")
            method, seed, attempt = "C1", 137, "attempt01"
            selected, summary, logs = Consumption().fixture()
            selection = {"method":method,"manifest":"fixture"}
            source = n.training_source_for_method(c.train_source, method)
            directory = n.run_directory(c, method, seed, attempt)
            directory.mkdir(parents=True)
            checkpoint = directory / "final"
            (checkpoint / "adapter").mkdir(parents=True)
            adapter = checkpoint / "adapter/adapter_model.safetensors"
            adapter.write_bytes(b"synthetic changed weight placeholder")
            inputs = n.artifact(c,"training_inputs",method,n.digest(selection)[:12],"full")
            n.save_rows(inputs / "train.jsonl", selected)
            manifest = n.artifact(c,"selection",method,"train.jsonl")
            n.save_rows(manifest, selected)
            config = {"model":c.entry["model"],"revision":"pinned","seed":seed,"epochs":1,"accumulation":8,
                "learning_rate":1e-4,"warmup_updates":10,"microbatch_size":1,"packing":False,"arm":method,
                "shared_universe_sha256":n.BANK_SHA,"parent_adapter_sha256":n.sha(parent_adapter),
                "original_parent_adapter_sha256":n.sha(parent_adapter),"bridge_identity":c.identity,
                "protocol_version":n.VERSION,"parent_training_seed":123,"selection_sha256":n.digest(selection),
                "screening_protocol_sha256":"protocol","initial_optimizer_state":"empty","fresh_warmup_updates":10,
                "branch_optimizer_policy":"fresh_optimizer_and_scheduler","smoke_only":False,
                "wrapper_sha256":n.sha(n.__file__),"code_sha256":hashlib.sha256(source.encode()).hexdigest(),
                "manifest_sha256":n.sha(inputs/"train.jsonl")}
            n.save(directory/"run_config.json",config)
            n.save(checkpoint/"run_config.json",config)
            summary["checkpoint"]=str(checkpoint)
            n.save(directory/"training_summary.json",summary)
            n.save_rows(directory/"training_log.jsonl",logs)
            receipt={"passed":True,"identity":c.identity,"method":method,"seed":seed,"smoke_only":False,
                "reload_probe_matched":True,"version":n.VERSION,"parent_seed":123,"bank_sha256":n.BANK_SHA,
                "selection_sha256":n.digest(selection),"parent_sha256":n.sha(parent_adapter),
                "screening_protocol_sha256":"protocol","unique_examples":1000,"actual_tokens":n.TARGET,
                "original_matcher_flags_resolved":False,
                "smoke_key":n.digest({"identity":c.identity,"selection":n.digest(selection),"method":method,"seed":seed}),
                "changed_adapter_tensors":["synthetic_lora"],"run_config_sha256":n.sha(directory/"run_config.json"),
                "summary_sha256":n.sha(directory/"training_summary.json"),"checkpoint":str(checkpoint),
                "adapter_sha256":n.sha(adapter)}
            n.save(directory/"completion.json",receipt)
            with patch.object(n,"require_prepared",return_value="protocol"), \
                    patch.object(n,"load_selection",return_value=(selected,selection)):
                self.assertTrue(n.validate_completion(c,directory,method,seed)["passed"])
                with self.assertRaises(ValueError):
                    n.validate_completion(c,directory,method,42)
                c.a.parent_seed=17
                with self.assertRaises(ValueError):
                    n.validate_completion(c,directory,method,seed)


class PreservationAndRecovery(unittest.TestCase):
    def test_atomic_artifacts_are_never_overwritten_by_different_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            n.save(path, {"complete": True})
            before = path.read_bytes()
            n.save(path, {"complete": True})
            with self.assertRaises(n.IntegrityError):
                n.save(path, {"complete": False})
            self.assertEqual(path.read_bytes(), before)

    def test_incomplete_training_gets_new_attempt_and_original_parent_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = args(root)
            c = SimpleNamespace(a=a, out=root, profile="1.5B_seed123", identity="identity")
            first = n.run_directory(c, "C1", 137, "attempt01")
            first.mkdir(parents=True)
            (first / "partial-optimizer").write_text("preserve")
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0)
            with patch.object(n, "find_training", side_effect=[None, ({"passed":True}, "new")]), \
                    patch.object(n.subprocess, "run", side_effect=run):
                self.assertEqual(n.run_training(c, "C1", 137)[1], "new")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("attempt02" in call and "137" in call and "C1" in call for call in calls))
            self.assertTrue("--smoke" in calls[0] and "--smoke" not in calls[1])
            self.assertEqual((first / "partial-optimizer").read_text(), "preserve")

    def test_completed_run_reused_without_training_process(self):
        c = SimpleNamespace()
        with patch.object(n, "find_training", return_value=({"passed": True}, "existing")), \
                patch.object(n.subprocess, "run") as runner:
            self.assertEqual(n.run_training(c, "C0", 2026)[1], "existing")
            runner.assert_not_called()


class Scheduler(unittest.TestCase):
    def test_allocation_chain_is_held_recorded_and_released_serially(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = args(root)
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if command[0] == "sbatch" and "--test-only" not in command:
                    count = sum(c[0] == "sbatch" and "--test-only" not in c for c in calls)
                    return SimpleNamespace(returncode=0, stdout=str(7000+count)+"\n", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch.object(n, "preflight"), patch.object(n, "control_root", return_value=root), \
                    patch.object(n, "validate_unit_receipts", return_value=set()), \
                    patch.object(n, "package_check", return_value="package"), \
                    patch.object(n, "HERE", root / "package"), \
                    patch.object(n.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                n.submit(a)
            ledger = n.read(root / "submissions/submission_001.json")
            self.assertEqual([j["job_id"] for j in ledger["jobs"]], ["7001", "7002", "7003"])
            batches = [c for c in calls if c[0] == "sbatch" and "--test-only" not in c]
            self.assertTrue(all("--hold" in c and "04:00:00" in c and "a100:1" in c for c in batches))
            self.assertNotIn("--dependency", batches[0])
            self.assertIn("afterany:7001", batches[1])
            self.assertIn("afterany:7002", batches[2])
            releases = [c for c in calls if c[:2] == ["scontrol", "release"]]
            self.assertEqual([c[-1] for c in releases], ["7001", "7002", "7003"])
            self.assertTrue(ledger["submission_complete"])

    def test_existing_active_submission_does_not_duplicate_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = args(root)
            n.save(root / "submissions/submission_001.json", {"jobs": [{"job_id":"123", "slot":0}], "submission_complete":True})
            with patch.object(n, "preflight"), patch.object(n, "control_root", return_value=root), \
                    patch.object(n, "HERE", root / "package"), \
                    patch.object(n.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="123 RUNNING None", stderr="")) as runner, \
                    redirect_stdout(io.StringIO()):
                n.submit(a)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0][0], "squeue")

    def test_launch_recovers_a_recorded_held_chain_after_a_disconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=args(root)
            n.save(root/"submissions/submission_001.json",{"jobs":[{"job_id":"123","slot":0},{"job_id":"124","slot":1}],
                "submission_complete":True,"released_job_ids":["123"]})
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0,stdout="123 RUNNING None\n124 PENDING JobHeldUser\n" if command[0]=="squeue" else "",stderr="")
            with patch.object(n,"preflight"),patch.object(n,"control_root",return_value=root), \
                    patch.object(n,"HERE",root/"package"),patch.object(n.subprocess,"run",side_effect=run),redirect_stdout(io.StringIO()):
                n.submit(a)
            self.assertEqual([c for c in calls if c[0]=="scontrol"],[["scontrol","release","124"]])
            self.assertFalse(any(c[0]=="sbatch" for c in calls))


class WholeController(unittest.TestCase):
    """Execute real subprocesses and durable receipts, replacing only GPU compute with a tiny fixture."""
    def run_controller(self, interruption=False, blocked=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        a = args(root)
        ledger = {"package_sha256":"package", "control_directory":str(root),
                  "jobs":[{"slot":0,"job_id":"900"},{"slot":1,"job_id":"901"}]}
        n.save(root / "submission.json", ledger)
        a.submission = str(root / "submission.json")
        script = root / "fixture_unit.py"
        script.write_text(textwrap.dedent('''
            import hashlib,json,os,pathlib,signal,sys
            root=pathlib.Path(sys.argv[1]); unit=json.loads(sys.argv[2]); mode=sys.argv[3]
            if mode=='interrupt' and not (root/'interrupted_once').exists():
                (root/'interrupted_once').write_text('preserved')
                os.kill(os.getpid(),signal.SIGTERM)
            if mode=='block' and unit['id']=='0.5B_seed17__prepare': sys.exit(65)
            proof=root/'fixture_artifacts'/unit['id']/'completion.json'
            proof.parent.mkdir(parents=True,exist_ok=True)
            proof.write_text(json.dumps({'fixture':True,'unit':unit}))
            receipt={'unit':unit,'identity':unit['profile'],'version':'screened-c0-c1-training-only-v1',
                     'complete':True,'files':{str(proof):hashlib.sha256(proof.read_bytes()).hexdigest()}}
            output=root/'unit_receipts'/(unit['id']+'.json'); output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(receipt))
        '''))
        def child(a, action, **kwargs):
            unit = next(u for u in n.units() if u["id"] == kwargs["unit_id"])
            return [sys.executable, str(script), str(root), json.dumps(unit), "interrupt" if interruption else "block" if blocked else "normal"]
        def context(a):
            return SimpleNamespace(profile=f"{a.model}_seed{a.parent_seed}", identity=f"{a.model}_seed{a.parent_seed}")
        with patch.object(n, "control_root", return_value=root), patch.object(n, "package_check", return_value="package"), \
                patch.object(n, "load_context", side_effect=context), patch.object(n, "original_pilot", return_value=({},"original")), \
                patch.object(n, "child", side_effect=child), patch.object(n, "cancel_future") as cancel, \
                patch.dict(os.environ, {"SLURM_JOB_ID":"900"}), redirect_stdout(io.StringIO()):
            result = n.worker(a)
            if interruption:
                self.assertFalse((root / "TRAINING_COMPLETE.json").exists())
                self.assertEqual(n.read(root / "STATUS.json")["status"], "continuing_in_next_allocation")
                a.slot = 1
                with patch.dict(os.environ, {"SLURM_JOB_ID":"901"}):
                    result = n.worker(a)
            value = n.read(root / "STATUS.json")
            if blocked:
                self.assertEqual(value["status"], "incomplete_blocked")
                self.assertFalse(value["all_requested_training_complete"])
                self.assertFalse((root / "TRAINING_COMPLETE.json").exists())
                self.assertEqual(value["C0_complete_of_12"], 10)
                self.assertEqual(value["C1_complete_of_12"], 9)
                self.assertEqual(result, 2)
            else:
                self.assertEqual(value["C0_complete_of_12"], 12)
                self.assertEqual(value["C1_complete_of_12"], 12)
                self.assertEqual(value["new_training_runs_complete_of_23"], 23)
                self.assertFalse(value["benchmark_evaluation_performed"])
                self.assertEqual(result, 0)
                self.assertTrue((root / "TRAINING_COMPLETE.json").is_file())
                n.validate_final_marker(a)
                # A later queued allocation must not modify the hash-bound completed report.
                before = n.sha(root / "STATUS.json")
                with patch.dict(os.environ, {"SLURM_JOB_ID": "901" if a.slot == 1 else "900"}):
                    n.worker(a)
                self.assertEqual(before, n.sha(root / "STATUS.json"))
            self.assertTrue(cancel.called)
        return root, a

    def test_all_23_new_runs_reach_only_the_training_completion_marker(self):
        self.run_controller()

    def test_interrupted_allocation_continues_without_a_notebook(self):
        self.run_controller(interruption=True)

    def test_blocked_profile_does_not_hide_success_of_other_parents(self):
        self.run_controller(blocked=True)

    def test_changed_completed_artifact_is_rejected(self):
        root, a = self.run_controller()
        proof = next((root / "fixture_artifacts").glob("*/completion.json"))
        proof.write_text("tampered")
        def context(a):
            return SimpleNamespace(profile=f"{a.model}_seed{a.parent_seed}", identity=f"{a.model}_seed{a.parent_seed}")
        with patch.object(n, "control_root", return_value=root), patch.object(n, "load_context", side_effect=context):
            with self.assertRaises(n.IntegrityError):
                n.validate_unit_receipts(a)


if __name__ == "__main__":
    unittest.main(verbosity=2)

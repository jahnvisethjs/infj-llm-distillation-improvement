"""Exercise public pilot stages with tiny exhaustive selection and native-shaped records."""

from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import importlib.util
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from capstone_person4 import b_select
from capstone_person4_approachReplayC1 import METHOD, VERSION
from capstone_person4_approachReplayC1 import common, evaluation, inputs, runtime, selection, sol
from capstone_person4_approachReplayC1.__main__ import main

REPO = Path(__file__).resolve().parents[2]
HAS_SOLVER = importlib.util.find_spec("ortools") is not None


def row(name, bin_number, tokens=10):
    return {"row_id": "SYNTHETIC-" + name, "problem_group_id": "SYNTHETIC-group-" + name,
            "dataset_id": "orca", "length_bin": bin_number, "supervised_tokens": tokens,
            "completion_tokens": tokens - 1, "eos_tokens": 1, "prompt_tokens": 2,
            "total_tokens": tokens + 2, "boundary_crossing_token": False,
            "problem": "SYNTHETIC question " + name, "completion": "SYNTHETIC reference", "synthetic": True}


def exhaustive(reference, bank, multiplier=Fraction(1, 2), *, utilities=None, **unused):
    """Independent finite oracle, only for the tiny synthetic fixtures."""
    counts = selection.quotas(reference)
    groups = {key: [r for r in bank if selection.cell(r) == key] for key in counts}
    feasible = []
    for parts in itertools.product(*(list(itertools.combinations(groups[k], counts[k])) for k in sorted(counts))):
        chosen = list(itertools.chain.from_iterable(parts))
        try:
            selection.validate_half(reference, chosen, bank, "fresh")
        except common.PilotError:
            continue
        score = sum(utilities[r["row_id"]] if utilities else b_select.utility(r["row_id"], 42) for r in chosen)
        tie = sum(b_select.tie_utility(r["row_id"], 42) for r in chosen)
        feasible.append(((-score, tie), chosen))
    if not feasible:
        return None, {"status": "INFEASIBLE", "infeasibility_proven": True}
    objective, chosen = min(feasible, key=lambda p: p[0])
    return chosen, {"status": "OPTIMAL", "actual_tokens": sum(r["supervised_tokens"] for r in chosen),
                    "objective": -objective[0], "best_bound": -objective[0]}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("replay-c1-v5-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.parent = [row(f"p{b}{i}", b) for b in range(4) for i in range(4)]
        self.bank = [row(f"f{b}{i}", b) for b in range(4) for i in range(4)]
        self.bundle = SimpleNamespace(root=self.root / "pilot", identity="SYNTHETIC-identity",
            cfg={"model": "0.5B", "parent_seed": 17}, binding={"scores": "SYNTHETIC-cache", "parent": "SYNTHETIC-parent"}, parent=self.parent,
            rebinned=deepcopy(self.parent), bank=self.bank,
            scores=[{"row_id": r["row_id"], "status": "ok", "mean_nll": float(i), "supervised_tokens": 10}
                    for i, r in enumerate(self.bank)],
            outcomes=[{"row_id": r["row_id"], "mistake": i % 2 == 0} for i, r in enumerate(self.bank)])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(selection, "TARGET", 160))

    def select(self):
        with patch.object(selection, "solve_replay", side_effect=lambda r: exhaustive(r, r)), \
                patch.object(b_select, "solve", side_effect=exhaustive):
            return selection.select(self.bundle)


class SelectionTests(Fixture):
    def test_odd_production_budget_rounding(self):
        self.assertEqual(selection.half_bounds([{"supervised_tokens": common.TARGET}]), (158626, 160219))

    def test_two_portions_and_combined_manifest(self):
        report = self.select()
        data, loaded = selection.load_selection(self.bundle)
        self.assertEqual(loaded, report)
        self.assertEqual(report["portions"]["replay"]["tokens"], 80)
        self.assertEqual(report["portions"]["fresh"]["tokens"], 80)
        self.assertEqual(report["actual_tokens"], 160)
        self.assertEqual(report["replay_token_fraction"], 0.5)
        self.assertEqual(len(data["train"]), 16)
        self.assertEqual(set(r["portion"] for r in data["membership"]), {"replay", "fresh"})
        parent = {r["row_id"]: r for r in self.parent}
        self.assertTrue(all(parent[r["row_id"]] == r for r in data["replay"]))

    def test_replay_is_hash_selected_fresh_uses_confirmed_mistakes(self):
        self.select()
        data, _ = selection.load_selection(self.bundle)
        expected, _ = exhaustive(self.bundle.rebinned, self.bundle.rebinned)
        self.assertEqual({r["row_id"] for r in data["replay"]}, {r["row_id"] for r in expected})
        # This confirms that the fresh selector sees the complete bank and the half multiplier.
        with patch.object(selection, "solve_replay", side_effect=lambda r: exhaustive(r, r)), \
                patch.object(b_select, "solve", side_effect=exhaustive) as solver:
            self.bundle.root = self.root / "second"
            selection.select(self.bundle)
            self.assertEqual(solver.call_args.args[2], Fraction(1, 2))
            self.assertEqual(len(solver.call_args.kwargs["utilities"]), len(self.bank))

    def test_fresh_solver_report_describes_c1_utilities_not_default_random_labels(self):
        def solver_with_default_labels(*args, **kwargs):
            self.assertIsNotNone(kwargs["utilities"])
            chosen, report = exhaustive(*args, **kwargs)
            report.update(student_scores_used=False, random_utility="default B seeded hash")
            return chosen, report
        with patch.object(selection, "solve_replay", side_effect=lambda r: exhaustive(r, r)), \
                patch.object(b_select, "solve", side_effect=solver_with_default_labels):
            report = selection.select(self.bundle)
        self.assertIs(report["fresh_solver"]["student_scores_used"], True)
        self.assertIsNone(report["fresh_solver"]["random_utility"])
        self.assertEqual(report["fresh_solver"]["utility_definition"],
                         "round(1000000 * fit) * (1 + confirmed_mistake)")

    def test_identical_rerun_reuses_immutable_selection(self):
        report = self.select()
        with patch.object(selection, "solve_replay", side_effect=AssertionError("must reuse")):
            self.assertEqual(selection.select(self.bundle), report)

    def test_parent_historical_bins_are_preserved(self):
        for r in self.parent:
            r["length_bin"] = 99
        self.select()
        data, _ = selection.load_selection(self.bundle)
        self.assertTrue(all(r["length_bin"] == 99 for r in data["replay"]))
        self.assertEqual({r["length_bin"] for r in data["membership"]}, set(range(4)))

    def test_half_row_count_does_not_establish_half_token_budget(self):
        chosen = [r for r in self.bank if r["row_id"].endswith(("0", "1"))]
        for r in chosen:
            r["supervised_tokens"] = 11
        with self.assertRaisesRegex(common.PilotError, "half-budget"):
            selection.validate_half(self.bundle.rebinned, chosen, self.bank, "fresh")

    def test_wrong_strata_or_duplicate_rows_fail(self):
        with self.assertRaises(common.PilotError):
            selection.validate_half(self.parent, self.bank[:8], self.bank, "fresh")
        with self.assertRaises(common.PilotError):
            selection.validate_half(self.parent, [self.bank[0]] * 8, self.bank, "fresh")

    def test_parent_seen_fresh_examples_fail(self):
        chosen = self.parent[:2] + self.bank[4:6] + self.bank[8:10] + self.bank[12:14]
        replay = self.parent[2:4] + self.parent[6:8] + self.parent[10:12] + self.parent[14:16]
        with self.assertRaisesRegex(common.PilotError, "overlaps original parent"):
            selection.validate_mix(self.bundle, replay, chosen)

    def test_manifest_tamper_fails(self):
        self.select()
        path = self.bundle.root / "selection/train.jsonl"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(common.PilotError, "Hash mismatch"):
            selection.load_selection(self.bundle)

    def test_forged_membership_labels_fail_even_with_new_file_hash(self):
        self.select()
        dest = self.bundle.root / "selection"
        membership = common.rows(dest / "membership.jsonl")
        membership[0]["portion"] = "fresh" if membership[0]["portion"] == "replay" else "replay"
        (dest / "membership.jsonl").write_text("".join(common.canonical(r) + "\n" for r in membership), encoding="utf-8")
        report = common.read(dest / "selection.json")
        report["manifests"]["membership"] = common.sha(dest / "membership.jsonl")
        (dest / "selection.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(common.PilotError, "Membership ledger"):
            selection.load_selection(self.bundle)

    def test_changed_cache_binding_invalidates_selection(self):
        self.select()
        self.bundle.binding["scores"] = "CHANGED"
        with self.assertRaisesRegex(common.PilotError, "changed inputs"):
            selection.load_selection(self.bundle)

    def test_infeasible_selection_preserves_failure_and_bounds(self):
        with patch.object(selection, "solve_replay", return_value=(None, {"status": "UNKNOWN"})):
            with self.assertRaisesRegex(common.PilotError, "bounds were not relaxed"):
                selection.select(self.bundle)
        self.assertEqual(common.read(self.bundle.root / "selection/replay_failure.json")["status"], "UNKNOWN")

    @unittest.skipUnless(HAS_SOLVER, "Pinned OR-Tools is in the separate Sol solver environment")
    def test_real_replay_solver_matches_exhaustive_oracle(self):
        actual, report = selection.solve_replay(self.bundle.rebinned)
        expected, oracle_report = exhaustive(self.bundle.rebinned, self.bundle.rebinned)
        self.assertEqual({r["row_id"] for r in actual}, {r["row_id"] for r in expected})
        self.assertEqual(report["objective"], oracle_report["objective"])


class RuntimeTests(Fixture):
    def mock_native_training(self):
        """Native-shaped I/O contract, explicitly synthetic; no weights or GPU model loaded."""
        core = SimpleNamespace(MODEL="Qwen/Qwen2.5-0.5B", REVISION="SYNTHETIC-revision")
        root = self.root / "native"
        adapter = root / "parents/profile/final/adapter"
        common.write_bytes(adapter / "adapter_model.safetensors", b"SYNTHETIC-parent-not-weights")
        self.stack.enter_context(patch.object(runtime, "parent_sha", return_value=common.sha(adapter / "adapter_model.safetensors")))
        self.stack.enter_context(patch.object(runtime, "TARGET", 160))
        common.write(root / "runtime/records/run_config.json", {"environment": {"packages": {}}})
        for filename in ("__init__.py", "person1_data_contract.py", "pipeline_controls.py", "instance_matching.py"):
            common.write_bytes(root / "runtime/vendor/capstone_person1" / filename, b"# SYNTHETIC fixture\n")
        common.write(root / "condition/training_inputs/prompt_template.json", {"template": "SYNTHETIC"})
        common.write(root / "condition/training_inputs/tokenization_policy.json", {
            "add_special_tokens": False, "append_eos": True, "max_total_tokens": 2048})
        torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: True), bfloat16="SYNTHETIC-bf16",
                                load=lambda *a, **k: "SYNTHETIC-probe", allclose=lambda *a, **k: True)
        token_text = "SYNTHETIC-tokenizer"
        tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: token_text))
        encode = lambda r, *args: {"counts": {"supervised_tokens": r["supervised_tokens"]}}
        p2 = SimpleNamespace(core=core, select_model=lambda model: None, get_tokenizer=lambda: tokenizer, encode=encode)
        source = "def train(args):\n    label = 'preliminary_document_aligned_B'\n    return fake_native_train(args)\n"
        context = SimpleNamespace(root=root, runtime=root / "runtime", condition=root / "condition",
            entry={"records_relative_path": "records", "adapter_relative_path": "parents/profile/final/adapter",
                   "model": core.MODEL, "revision": core.REVISION},
            cfg={"training": {"checkpoint_every_updates": 100}}, train_source=source,
            legacy=SimpleNamespace(gpu=lambda: torch), common=SimpleNamespace(load_person2=lambda: p2),
            contract={"prompt_template": "SYNTHETIC", "max_total_tokens": 2048})
        def validate_consumption(summary, selected, logs):
            self.assertEqual(summary["processed_tokens"], sum(r["supervised_tokens"] for r in selected))
            self.assertEqual(sum(r["target_tokens"] for r in logs), summary["processed_tokens"])
            self.assertEqual(summary["parent_optimizer_updates"], 0)
        self.bundle.context = context
        self.bundle.cfg = {"model": "0.5B", "parent_seed": 17}
        self.bundle.native = SimpleNamespace(digest=common.digest,
            verify_proof=lambda c: {"tokenizer_sha256": common.digest(token_text)}, validate_consumption=validate_consumption)
        self.stack.enter_context(patch("capstone_person4.continuation.verify_adapter_update", return_value=["SYNTHETIC-lora"]))
        core.build_model = lambda *a, **k: "SYNTHETIC-model"
        core.probe_logits = lambda *a: "SYNTHETIC-probe"
        def fake_train(args):
            selected = common.rows(args.data)
            encoded = [encode(r) for r in selected]
            core.provenance_guard(args, selected, None)
            budget = core.parent_budget_check(args, selected, encoded, None, args.parent_metadata)
            directory = Path(args.out)
            config = {**core.fingerprint(args), **common.RECIPE, **args.continuation_metadata,
                      "arm": args.arm, "mode": "debug" if args.smoke else "branch", "seed": args.seed, "dtype": "torch.bfloat16",
                      "parent_adapter_sha256": runtime.parent_sha(self.bundle.cfg), "branch_optimizer_policy": "fresh_optimizer_and_scheduler",
                      "branch_budget": budget}
            common.write(directory / "run_config.json", config)
            common.write(directory / "final/run_config.json", config)
            common.write_bytes(directory / "final/adapter/adapter_model.safetensors", b"SYNTHETIC-updated-not-weights")
            tokens = sum(r["supervised_tokens"] for r in selected)
            summary = {"complete": True, "processed_tokens": tokens, "seen_ids": [r["row_id"] for r in selected],
                       "seen_groups": [r["problem_group_id"] for r in selected], "parent_optimizer_updates": 0,
                       "checkpoint": str(directory / "final"), "training_seconds": 0.01}
            common.write(directory / "training_summary.json", summary)
            common.write_rows(directory / "training_log.jsonl", [{"target_tokens": tokens}])
        core.fake_native_train = fake_train
        return torch

    def test_native_adapter_smoke_full_consumption_and_parent_immutability(self):
        self.select()
        torch = self.mock_native_training()
        before = common.sha(self.bundle.context.root / "parents/profile/final/adapter/adapter_model.safetensors")
        smoke = runtime.train(self.bundle, 42, smoke=True)
        full = runtime.train(self.bundle, 42)
        self.assertTrue(smoke["smoke_only"])
        self.assertFalse(full["smoke_only"])
        self.assertEqual(full["portion_tokens"], {"replay": 80, "fresh": 80})
        self.assertEqual(full["effective_train_source_sha256"],
                         common.sha(self.bundle.root / "training_inputs/full/effective_train.py"))
        self.assertEqual(before, common.sha(self.bundle.context.root / "parents/profile/final/adapter/adapter_model.safetensors"))
        self.assertEqual(runtime.train(self.bundle, 42), full)
        path = self.bundle.root / "runs/seed42_attempt01/training_summary.json"
        path.write_text(json.dumps({"complete": False}), encoding="utf-8")
        with self.assertRaisesRegex(common.PilotError, "Hash mismatch"):
            runtime.validate_run(self.bundle, path.parent, 42)

    def test_reload_failure_never_creates_success_receipt(self):
        self.select()
        torch = self.mock_native_training()
        torch.allclose = lambda *a, **k: False
        with self.assertRaisesRegex(common.PilotError, "reload probe differs"):
            runtime.train(self.bundle, 42, smoke=True)
        self.assertFalse((self.bundle.root / "smokes/seed42_attempt01/completion.json").exists())

    def test_final_validation_failure_never_publishes_passed_receipt(self):
        self.select()
        self.mock_native_training()
        with patch.object(runtime, "validate_run", side_effect=common.PilotError("final metadata validation failed")):
            with self.assertRaisesRegex(common.PilotError, "final metadata validation failed"):
                runtime.train(self.bundle, 42, smoke=True)
        self.assertFalse((self.bundle.root / "smokes/seed42_attempt01/completion.json").exists())

    def test_debug_mode_is_accepted_only_for_smoke_not_full_training(self):
        self.select()
        self.mock_native_training()
        runtime.train(self.bundle, 42, smoke=True)
        smoke_path = self.bundle.root / "smokes/seed42_attempt01"
        self.assertEqual(common.read(smoke_path / "run_config.json")["mode"], "debug")
        runtime.train(self.bundle, 42)
        full_path = self.bundle.root / "runs/seed42_attempt01"
        config = common.read(full_path / "run_config.json")
        self.assertEqual(config["mode"], "branch")
        config["mode"] = "debug"
        (full_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        receipt = common.read(full_path / "completion.json")
        receipt["run_config_sha256"] = common.sha(full_path / "run_config.json")
        (full_path / "completion.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(common.PilotError, "mode"):
            runtime.validate_run(self.bundle, full_path, 42)

    def test_upstream_synthetic_metadata_is_not_a_fixture_marker(self):
        parent = [row(f"real-parent-{i}", 0, 318) for i in range(1000)]
        parent[-1]["supervised_tokens"] += common.TARGET - sum(r["supervised_tokens"] for r in parent)
        bank = [row(f"real-bank-{i}", 0) for i in range(4955)]
        inputs.validate_population(parent, bank)
        bank[0]["problem_group_id"] = parent[0]["problem_group_id"]
        with self.assertRaisesRegex(common.PilotError, "overlap"):
            inputs.validate_population(parent, bank)

    def test_smoke_contains_eight_from_each_portion(self):
        self.select()
        data, _ = selection.load_selection(self.bundle)
        selected = runtime.training_rows(data, smoke=True)
        self.assertEqual(len(selected), 16)
        encoded = [{"counts": {"supervised_tokens": r["supervised_tokens"]}} for r in selected]
        self.assertEqual(runtime.reconcile_labels(selected, encoded, data["membership"]), {"replay": 80, "fresh": 80})
        encoded[0]["counts"]["supervised_tokens"] += 1
        with self.assertRaisesRegex(common.PilotError, "Actual trainer label"):
            runtime.reconcile_labels(selected, encoded, data["membership"])

    def test_native_source_changes_only_reporting_label(self):
        source = "def train():\n    x = 'preliminary_document_aligned_B'\n    update()\n"
        adapted = runtime.execution_source(source)
        self.assertEqual(adapted.replace("'screened_replay_c1_50_50_pilot'", "'preliminary_document_aligned_B'"), source)
        with self.assertRaises(common.PilotError):
            runtime.execution_source(source + source)

    def test_full_training_requires_own_smoke_before_gpu(self):
        self.select()
        with self.assertRaisesRegex(common.PilotError, "matching 16-example smoke"):
            runtime.train(self.bundle, 42)

    def test_partial_attempt_is_preserved(self):
        self.select()
        dest = runtime.run_directory(self.bundle, 42, "attempt01")
        dest.mkdir(parents=True)
        with self.assertRaisesRegex(common.PilotError, "Partial attempt preserved"):
            runtime.train(self.bundle, 42)
        self.assertTrue(dest.exists())

    def test_recipe_and_policy_mismatch_fail(self):
        config = {**common.RECIPE, "dtype": "torch.bfloat16", "branch_optimizer_policy": "fresh_optimizer_and_scheduler"}
        inputs.recipe(config)
        config["warmup_updates"] = 0
        with self.assertRaises(common.PilotError):
            inputs.recipe(config)
        with self.assertRaises(common.PilotError):
            inputs.token_policy({"add_special_tokens": False, "append_eos": False, "max_total_tokens": 2048})


class EvaluationTests(Fixture):
    def settings(self):
        expected = {"method": METHOD, "seed": 42, "checkpoint_sha256": "SYNTHETIC-checkpoint", "manifest_sha256": "SYNTHETIC-manifest"}
        config = {"precision": "fp32", "max_new_tokens": 1024, "batch_size": 16, "stop_at_question": False,
            "question_text": "p1_clean", "prompt_sha256": common.PROMPT_SHA, "add_special_tokens": False,
            "decoding": "greedy", "eos_token_id": 151643, "adapter_mode": "merged", "parser_version": "p3-parser-v6",
            "cut_at_next_question": True, "math_verify": "0.9.0", "torch": "2.6.0+cu124", "transformers": "4.57.1",
            "peft": "0.17.1", "cuda": "12.4", "gpu": "NVIDIA A100-SXM4-80GB",
            "eval_file_sha256": common.QUESTIONS["dev_gsm"], "evaluate_py_sha256": common.EVALUATOR_SHA,
            "answer_parser_sha256": common.PARSER_SHA, "checkpoint_sha256": expected["checkpoint_sha256"],
            "train_manifest_sha256": expected["manifest_sha256"], "model": "Qwen/Qwen2.5-0.5B",
            "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987", "train_seed": 42, "train_arm": METHOD}
        records = [{"problem_id": f"q{i}", "correct": i % 2 == 0, "parse_status": "ok", "parser_version": "p3-parser-v6",
                    "generated_tokens": 5} for i in range(500)]
        return expected, config, records, {r["problem_id"] for r in records}

    def test_common_contract_accepts_and_runtime_mismatch_rejects(self):
        target, config, records, ids = self.settings()
        self.assertEqual(len(evaluation.validate_predictions(records, config, target, "dev_gsm", ids)), 500)
        config["torch"] = "2.14.1"
        with self.assertRaisesRegex(common.PilotError, "runtime/contract mismatch"):
            evaluation.validate_predictions(records, config, target, "dev_gsm", ids)

    def test_duplicate_question_or_checker_error_rejects(self):
        target, config, records, ids = self.settings()
        bad = records[:-1] + [records[0]]
        with self.assertRaises(common.PilotError):
            evaluation.validate_predictions(bad, config, target, "dev_gsm", ids)
        records[0].update(correct=False, parse_status="error")
        with self.assertRaisesRegex(common.PilotError, "Checker execution errors"):
            evaluation.validate_predictions(records, config, target, "dev_gsm", ids)

    def test_question_or_checkpoint_mismatch_rejects(self):
        target, config, records, ids = self.settings()
        for key in ("checkpoint_sha256", "eval_file_sha256", "train_manifest_sha256"):
            changed = {**config, key: "CHANGED"}
            with self.assertRaises(common.PilotError):
                evaluation.validate_predictions(records, changed, target, "dev_gsm", ids)

    def test_bootstrap_is_paired_stratified_and_deterministic(self):
        left = {"gsm": {"a": True, "b": True}, "math": {"c": False, "d": False}}
        right = {"gsm": {"a": False, "b": False}, "math": {"c": True, "d": True}}
        result = evaluation.paired_bootstrap(left, right, resamples=100)
        self.assertEqual(result["difference_pp"], 0)
        self.assertEqual(result["ci95_low_pp"], 0)
        self.assertEqual(result, evaluation.paired_bootstrap(left, right, resamples=100))
        right["math"].pop("d")
        with self.assertRaises(common.PilotError):
            evaluation.paired_bootstrap(left, right, resamples=10)

    def test_native_c1_gzip_reuse_checks_receipt_hashes(self):
        expected, config, records, ids = self.settings()
        expected["method"] = "C1"
        config["train_arm"] = "C1"
        self.bundle.cfg = {"model": "0.5B", "parent_seed": 17, "c1_eval_root": str(self.root / "native_evaluation")}
        dest = Path(self.bundle.cfg["c1_eval_root"]) / "results/C1_0.5B_p17_t42/dev_gsm"
        common.write(dest / "predictions.config.json", config)
        raw = "".join(common.canonical(r) + "\n" for r in records).encode()
        common.write_bytes(dest / "predictions.jsonl.gz", gzip.compress(raw, mtime=0))
        import hashlib
        common.write(dest / "completion.json", {"complete": True, "checkpoint_sha256": expected["checkpoint_sha256"],
            "eval_file_sha256": common.QUESTIONS["dev_gsm"], "benchmark": "dev_gsm", "reserved_final_tests_performed": False,
            "config_sha256": common.sha(dest / "predictions.config.json"), "predictions_sha256": hashlib.sha256(raw).hexdigest()})
        reused, _, origin = evaluation.native_c1_result(self.bundle, expected, "dev_gsm", ids)
        self.assertEqual(reused, records)
        self.assertTrue(origin["reused"])
        (dest / "predictions.jsonl.gz").write_bytes(gzip.compress(raw + b"\n", mtime=0))
        with self.assertRaisesRegex(common.PilotError, "prediction bytes changed"):
            evaluation.native_c1_result(self.bundle, expected, "dev_gsm", ids)

    def test_published_c1_layout_is_used_when_native_results_are_absent(self):
        expected, config, records, ids = self.settings()
        expected["method"] = "C1"
        config["train_arm"] = "C1"
        self.bundle.cfg = {"model": "0.5B", "parent_seed": 17, "c1_eval_root": str(self.root / "work/outputs/native-evaluation"),
                           "work": str(self.root / "work")}
        dest = self.root / "github_screened_4955_latest/C1/evaluation/0.5B_parent17_seed42/dev_gsm"
        common.write(dest / "predictions.config.json", config)
        raw = "".join(common.canonical(r) + "\n" for r in records).encode()
        common.write_bytes(dest / "predictions.jsonl.gz", gzip.compress(raw, mtime=0))
        import hashlib
        common.write(dest / "completion.json", {"complete": True, "checkpoint_sha256": expected["checkpoint_sha256"],
            "eval_file_sha256": common.QUESTIONS["dev_gsm"], "benchmark": "dev_gsm", "reserved_final_tests_performed": False,
            "config_sha256": common.sha(dest / "predictions.config.json"), "predictions_sha256": hashlib.sha256(raw).hexdigest()})
        reused, _, origin = evaluation.native_c1_result(self.bundle, expected, "dev_gsm", ids)
        self.assertEqual(reused, records)
        self.assertEqual(origin["source"], str(dest / "predictions.jsonl.gz"))
        config["torch"] = "2.14.1"
        (dest / "predictions.config.json").write_text(json.dumps(config), encoding="utf-8")
        receipt = common.read(dest / "completion.json")
        receipt["config_sha256"] = common.sha(dest / "predictions.config.json")
        (dest / "completion.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(common.PilotError, "runtime/contract mismatch"):
            evaluation.native_c1_result(self.bundle, expected, "dev_gsm", ids)

    def test_finalization_requires_all_seeds_and_reports_matched_contrasts(self):
        self.select()
        def verified_run(bundle, path, seed):
            return {"checkpoint": "SYNTHETIC-checkpoint", "adapter_sha256": "SYNTHETIC-digest",
                    "portion_tokens": {"replay": 80, "fresh": 80}}
        def verified_result(bundle, method, seed, benchmark, *args):
            value = {"q1": True, "q2": method == METHOD}
            return value, {"method": method, "seed": seed, "benchmark": benchmark}
        with patch.object(evaluation, "validate_run", side_effect=verified_run), \
                patch.object(evaluation, "validate_result", side_effect=verified_result) as validator:
            result = evaluation.finalize(self.bundle)
        self.assertEqual(validator.call_count, 18)
        self.assertEqual(result["three_seed_summary"][METHOD]["mean_accuracy_percent"], 100)
        self.assertEqual(result["three_seed_summary"]["A"]["mean_accuracy_percent"], 50)
        report = common.read(Path(result["report_directory"]) / "summary.json")
        self.assertEqual(len(report["contrasts"]), 18)
        self.assertTrue(all(c["difference_pp"] == 50 for c in report["contrasts"]))
        with patch.object(evaluation, "validate_run", side_effect=verified_run), \
                patch.object(evaluation, "validate_result", side_effect=common.PilotError("missing matched evaluation")):
            with self.assertRaisesRegex(common.PilotError, "missing matched"):
                evaluation.finalize(self.bundle, evaluation_attempt="missing")
        self.assertFalse((self.bundle.root / "reports/missing/summary.json").exists())


class SchedulingAndCLITests(Fixture):
    def test_config_keeps_virtual_environment_executable_symlink_path(self):
        config = {"version": VERSION, "model": "0.5B", "parent_seed": 17, "selection_seed": 42,
                  "continuation_seeds": list(common.SEEDS)}
        for name in ("work", "b_root", "c1_profile", "output_root", "a_eval_root", "c1_eval_root"):
            config[name] = str(self.root / name)
        config.update(native_runner=str(self.root / "native_package/nightly.py"),
                      training_python=str(self.root / "venv_train/bin/python"),
                      solver_python=str(self.root / "venv_solver/bin/python"))
        path = self.root / "config.json"
        common.write(path, config)
        original_resolve = Path.resolve
        def resolve(current, *a, **kw):
            if current.name == "python":
                return self.root / "system/bin/python"
            return original_resolve(current, *a, **kw)
        with patch.object(Path, "resolve", autospec=True, side_effect=resolve):
            loaded = common.load_config(path)
        self.assertEqual(loaded["training_python"], config["training_python"])
        self.assertEqual(loaded["solver_python"], config["solver_python"])

    def test_scheduler_failure_preserves_partial_job_ledger(self):
        self.bundle.cfg = {"model": "0.5B", "parent_seed": 17, "training_python": "/SYNTHETIC/python", "config_path": "/SYNTHETIC/config.json",
                           "slurm": {"account": "class_account", "partition": "public", "qos": "class"}}
        with patch.object(sol.subprocess, "run", side_effect=[SimpleNamespace(stdout="100\n"),
                         subprocess.CalledProcessError(1, ["sbatch"])]) as scheduler:
            with self.assertRaises(subprocess.CalledProcessError):
                sol.submit(self.bundle, "attempt01", "eval01")
        self.assertEqual(scheduler.call_count, 2)
        root = self.bundle.root / "submissions/attempt01_eval01"
        self.assertTrue((root / "job_00_select.json").is_file())
        self.assertFalse((root / "submission.json").exists())
        with patch.object(sol.subprocess, "run", return_value=SimpleNamespace(stdout="")) as scheduler:
            with self.assertRaisesRegex(common.PilotError, "Interrupted submission preserved"):
                sol.submit(self.bundle, "attempt01", "eval01")
        self.assertEqual(scheduler.call_count, 1)

    def test_chain_is_serial_with_failure_dependencies(self):
        self.bundle.cfg = {"model": "0.5B", "parent_seed": 17, "training_python": "/SYNTHETIC/python", "config_path": "/SYNTHETIC/config.json",
                           "slurm": {"account": "class_account", "partition": "public", "qos": "class"}}
        counter = itertools.count(100)
        def fake_run(command, **kwargs):
            return SimpleNamespace(stdout=str(next(counter)) + "\n")
        with patch.object(sol.subprocess, "run", side_effect=fake_run) as scheduler:
            report = sol.submit(self.bundle, "attempt01", "eval01")
        self.assertEqual(len(report["jobs"]), 7)
        for i, job in enumerate(report["jobs"]):
            command = job["command"]
            self.assertEqual("--gpus=a100:1" in command, job["gpu"])
            if i:
                self.assertIn("--dependency=afterok:" + str(99 + i), command)
        self.assertEqual([r["stage"] for r in report["jobs"]], ["select", "smoke", "train42", "train137", "train2026", "evaluate", "finalize"])
        with patch.object(sol.subprocess, "run", return_value=SimpleNamespace(stdout="100\n")) as scheduler:
            with self.assertRaisesRegex(common.PilotError, "active submission"):
                sol.submit(self.bundle, "attempt01", "eval01")
        self.assertEqual(scheduler.call_count, 1)
        self.assertEqual(scheduler.call_args.args[0][0], "squeue")

    def test_cli_imports_do_not_import_torch_or_submit_jobs(self):
        command = [sys.executable, "-B", "-c", "import sys; from capstone_person4_approachReplayC1 import inputs, selection, runtime, evaluation; assert 'torch' not in sys.modules"]
        subprocess.run(command, cwd=REPO, check=True)

    def test_gpu_authorization_is_required_before_loading_native_inputs(self):
        config = {"version": VERSION, "model": "0.5B", "parent_seed": 17, "selection_seed": 42,
                  "continuation_seeds": list(common.SEEDS)}
        for name in ("work", "b_root", "native_runner", "c1_profile", "output_root", "training_python",
                     "solver_python", "a_eval_root", "c1_eval_root"):
            config[name] = str(self.root / name)
        config["native_runner"] = str(self.root / "native_package/nightly.py")
        path = self.root / "pilot.json"
        common.write(path, config)
        with patch.object(inputs, "load", side_effect=AssertionError("must not load or run GPU")):
            self.assertEqual(main(["train", "--config", str(path)]), 2)

    def test_immutable_writer_preserves_different_existing_bytes(self):
        path = self.root / "receipt.json"
        common.write(path, {"value": 1})
        common.write(path, {"value": 1})
        with self.assertRaises(common.PilotError):
            common.write(path, {"value": 2})
        self.assertEqual(common.read(path), {"value": 1})


if __name__ == "__main__":
    unittest.main()

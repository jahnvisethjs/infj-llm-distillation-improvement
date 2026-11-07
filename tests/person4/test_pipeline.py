"""Scientific-integrity tests. Fixtures are synthetic and never unlock real gates.

Test artifacts are retained under .test_artifacts; these tests do not delete files.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from capstone_person4.allocation import AllocationError, allocate
from capstone_person4.audit import audit, gate
from capstone_person4.config import load_config
from capstone_person4.evaluation import paired_difference, validate_predictions
from capstone_person4.person2_adapter import continuation_source, notebook_sources
from capstone_person4.provenance import PipelineError, digest, read_json, sha, write_bytes, write_json
from capstone_person4.reference_scoring import model_stamp, score_batch, token_losses, validate_score
from capstone_person4.selection import combined_priorities, fit_priorities, random_priorities

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / ".test_artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def artifact_dir():
    return Path(tempfile.mkdtemp(prefix="person4-", dir=ARTIFACTS))


def fixture_rows():
    return [{"row_id": f"r{i}", "problem_group_id": f"g{i}", "dataset_id": "orca" if i < 6 else "numina",
             "length_bin": 0, "supervised_tokens": 10, "synthetic": True} for i in range(12)]


def fixture_scores(rows):
    return [{"row_id": r["row_id"], "mean_nll": float(i // 2), "supervised_tokens": 10, "status": "ok"}
            for i, r in enumerate(rows)]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.scores = fixture_scores(self.rows)
        self.coverage = {"orca|0": 1, "numina|0": 1}

    def test_tied_losses_have_equal_fit(self):
        rows = fit_priorities(self.rows, self.scores, 42)
        by_id = {r["row_id"]: r for r in rows}
        self.assertEqual(by_id["r0"]["fit"], by_id["r1"]["fit"])
        self.assertAlmostEqual(by_id["r0"]["fit"], 1 - 1 / 6)

    def test_singleton_fit(self):
        result = fit_priorities(self.rows[:1], self.scores[:1], 42)
        self.assertEqual(result[0]["fit"], 0.5)

    def test_lower_loss_is_preferred_and_whole_rows_kept(self):
        rows = fit_priorities(self.rows, self.scores, 42)
        chosen, report = allocate(rows, 40, self.coverage, 0)
        self.assertEqual({r["row_id"] for r in chosen}, {"r0", "r1", "r6", "r7"})
        self.assertEqual(report["actual_tokens"], 40)
        self.assertEqual(report["quotas"], {"orca|0": 2, "numina|0": 2})

    def test_order_invariance_and_repeatability(self):
        a, ra = allocate(random_priorities(self.rows, 42), 40, self.coverage)
        b, rb = allocate(random_priorities(list(reversed(self.rows)), 42), 40, self.coverage)
        self.assertEqual(a, b)
        self.assertEqual(ra, rb)

    def test_bonus_zero_and_constant_mistakes_preserve_selection(self):
        fit = fit_priorities(self.rows, self.scores, 42)
        base, _ = allocate(fit, 40, self.coverage)
        for value, bonus in ((True, 0), (True, 1), (False, 1)):
            combined = combined_priorities(fit, {r["row_id"]: value for r in fit}, bonus)
            chosen, _ = allocate(combined, 40, self.coverage)
            self.assertEqual([r["row_id"] for r in base], [r["row_id"] for r in chosen])

    def test_partial_duplicate_and_nonfinite_scores_rejected(self):
        for scores in (self.scores[:-1], self.scores[:-1] + self.scores[:1],
                       [{**s, "mean_nll": float("nan")} for s in self.scores]):
            with self.assertRaises(PipelineError):
                fit_priorities(self.rows, scores, 42)

    def test_duplicate_group_rejected(self):
        rows = deepcopy(self.rows)
        rows[1]["problem_group_id"] = rows[0]["problem_group_id"]
        with self.assertRaises(PipelineError):
            random_priorities(rows, 42)

    def test_budget_infeasibility_is_explicit(self):
        with self.assertRaises(AllocationError) as caught:
            allocate(random_priorities(self.rows, 42), 9999, self.coverage)
        self.assertIn("proven total-capacity", caught.exception.report["reason"])
        self.assertIsNotNone(caught.exception.report["nearest"])
        with self.assertRaises(AllocationError) as caught:
            allocate(random_priorities(self.rows, 42), 41, self.coverage, 0)
        self.assertIn("bounded search", caught.exception.report["reason"])

    def test_boundary_swap_reaches_budget_and_records_sacrifice(self):
        rows = [{**r, "stratum": "s", "utility": 100 - i, "tie_key": str(i), "supervised_tokens": n}
                for i, (r, n) in enumerate(zip(self.rows[:3], [8, 9, 12]))]
        chosen, report = allocate(rows, 20, {"s": 1}, 0)
        self.assertEqual(sum(r["supervised_tokens"] for r in chosen), 20)
        self.assertTrue(report["swaps"])


class GateAndArtifactTests(unittest.TestCase):
    def test_cli_dry_run_is_visibly_synthetic(self):
        from capstone_person4.__main__ import dry_run
        root = artifact_dir()
        result = dry_run(root)
        self.assertTrue(result["synthetic"])
        self.assertTrue(read_json(root / "SYNTHETIC_DRY_RUN.json")["synthetic"])

    def test_group_and_context_exclusions_are_common_and_traceable(self):
        from capstone_person4.data import eligible_candidates
        rows = [{**r, "total_tokens": 12} for r in fixture_rows()[:4]]
        rows[-1]["total_tokens"] = 3000
        groups, exclusions = eligible_candidates(rows, {"g0"}, {"g1"}, 2048)
        self.assertEqual([r["row_id"] for part in groups.values() for r in part], ["r2"])
        self.assertEqual({r["reason"] for r in exclusions}, {"parent_seen_group", "development_or_final_group", "whole_example_context_limit"})

    def test_question_prompt_receives_no_reference(self):
        from capstone_person4.person2_adapter import Person2Adapter
        adapter = Person2Adapter.__new__(Person2Adapter)
        received = []
        adapter.contract = SimpleNamespace(render_prompt=lambda problem: received.append(problem) or "Question: " + problem)
        prompt = adapter.render_question({"problem": "QUESTION_ONLY", "completion": "SECRET_REFERENCE", "final_answer": "SECRET_ANSWER"})
        self.assertEqual(received, ["QUESTION_ONLY"])
        self.assertNotIn("SECRET", prompt)

    def test_example_unresolved_values_do_not_unlock_stages(self):
        cfg = load_config(REPO / "configs/c0.example.json")
        with self.assertRaisesRegex(PipelineError, "prepare blocked by release"):
            gate(cfg, "prepare")

    def test_model_independent_gate_does_not_check_gpu(self):
        def missing(unused):
            raise FileNotFoundError("missing actual adapter")
        checks = {name: (lambda unused: {"verified": True}) for name in ("release", "parent_exposure", "overlap", "interfaces")}
        checks["parent_weights"] = missing
        checks["runtime"] = lambda unused: self.fail("Pure data stage imported GPU runtime")
        with patch("capstone_person4.audit.CHECKS", checks):
            self.assertIn("release", gate({}, "prepare"))
            with self.assertRaisesRegex(PipelineError, "parent_weights"):
                gate({}, "score-reference")

    def test_synthetic_config_cannot_enter_real_stages(self):
        value = read_json(REPO / "configs/c0.example.json")
        value["synthetic"] = True
        path = artifact_dir() / "synthetic.json"
        write_json(path, value)
        with self.assertRaisesRegex(PipelineError, "synthetic=false"):
            load_config(path)

    def test_conflicting_existing_outputs_are_preserved(self):
        path = artifact_dir() / "result.json"
        write_json(path, {"value": 1})
        write_json(path, {"value": 1})
        with self.assertRaises(PipelineError):
            write_json(path, {"value": 2})
        self.assertEqual(read_json(path), {"value": 1})

    def test_cache_keys_change_with_dependencies(self):
        from capstone_person4.reference_scoring import measurement_key
        cfg = load_config(REPO / "configs/c0.example.json")
        prep = {"candidate_sha256": "candidate", "tokenizer_fingerprint": "tokenizer",
                "evidence": {"release": {"format_sha256": "format"}}}
        baseline = measurement_key(cfg, prep)
        replicated = deepcopy(cfg)
        replicated["training"]["seed"] = 17
        self.assertEqual(baseline, measurement_key(replicated, prep))
        for location, key in (("parent", "adapter_sha256"), ("scoring", "dtype")):
            changed = deepcopy(cfg)
            changed[location][key] = "changed"
            self.assertNotEqual(baseline, measurement_key(changed, prep))
        changed = deepcopy(prep)
        changed["tokenizer_fingerprint"] = "different"
        self.assertNotEqual(baseline, measurement_key(cfg, changed))

    def test_notebook_adaptation_changes_initialization_not_training_loop(self):
        sources = notebook_sources(REPO / "Person2_SOL_Seeds17_123_UpTo4000.ipynb")
        original = ast.parse(sources["trainer_core.py"])
        adapted = ast.parse(continuation_source(sources["trainer_core.py"]))
        def update_loop(tree):
            train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
            loop = next(n for n in ast.walk(train) if isinstance(n, ast.For) and ast.unparse(n.target) == "i" and ast.unparse(n.iter) == "indices")
            return ast.dump(loop)
        self.assertEqual(update_loop(original), update_loop(adapted))
        self.assertIn("fresh_from_verified_parent_weights", ast.unparse(adapted))


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch unavailable; CPU tensor checks require torch")
        cls.torch = torch

    def test_manual_causal_alignment_prompt_mask_and_eos(self):
        t = self.torch
        probs = t.tensor([[[0.2, 0.3, 0.5], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2], [0.7, 0.1, 0.2]]])
        ids = t.tensor([[1, 1, 2, 0]])
        labels = t.tensor([[-100, -100, 2, 0]])
        record = token_losses(probs.log(), ids, labels, t.ones_like(ids))[0]
        self.assertAlmostEqual(record["mean_nll"], (-math.log(0.6) - math.log(0.5)) / 2, places=6)
        self.assertEqual([r["position"] for r in record["tokens"]], [2, 3])
        self.assertEqual(record["tokens"][-1]["token_id"], 0)
        validate_score(record, 2)

    def test_pad_id_equal_to_eos_does_not_remove_genuine_eos(self):
        t = self.torch
        ids = t.tensor([[1, 2, 0, 0]])
        labels = t.tensor([[-100, 2, 0, -100]])
        mask = t.tensor([[1, 1, 1, 0]])
        record = token_losses(t.zeros((1, 4, 3)), ids, labels, mask)[0]
        self.assertEqual(record["supervised_tokens"], 2)
        self.assertAlmostEqual(record["total_nll"], 2 * math.log(3), places=6)

    def test_supervised_padding_and_zero_count_fail(self):
        t = self.torch
        ids = t.tensor([[1, 0]])
        for labels, mask in ((t.tensor([[-100, 0]]), t.tensor([[1, 0]])),
                             (t.tensor([[-100, -100]]), t.tensor([[1, 1]]))):
            with self.assertRaises(PipelineError):
                token_losses(t.zeros((1, 2, 3)), ids, labels, mask)

    def test_padded_batches_equal_single_scoring_and_weights_unchanged(self):
        t = self.torch
        class Tiny(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_weight = t.nn.Parameter(t.tensor([[0.2, -0.4, 0.7], [0.8, 0.1, -0.2], [-0.3, 0.5, 0.2]]))
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.lora_weight[input_ids])
        model = Tiny()
        encoded = [{"input_ids": [1, 2, 0], "labels": [-100, 2, 0]},
                   {"input_ids": [1, 2, 1, 0], "labels": [-100, 2, 1, 0]}]
        before = model_stamp(model)
        singles = [score_batch(model, [e], 0)[0] for e in encoded]
        batched = score_batch(model, encoded, 0)
        self.assertEqual(singles, batched)
        self.assertEqual(before, model_stamp(model))
        self.assertIsNone(model.lora_weight.grad)

    def test_corrupt_token_reconstruction_fails(self):
        record = {"tokens": [{"position": 2, "token_id": 0, "nll": 1.0}],
                  "supervised_tokens": 1, "mean_nll": 2.0, "total_nll": 2.0}
        with self.assertRaises(PipelineError):
            validate_score(record)


class EvaluationTests(unittest.TestCase):
    def test_comparison_matches_seeds_and_rejects_budget_mismatch(self):
        from capstone_person4.evaluation import compare, method_definition
        from capstone_person4.provenance import write_rows
        root = artifact_dir()
        cfg = load_config(REPO / "configs/c0.example.json")
        cfg["parent"]["adapter_sha256"] = "SYNTHETIC_NO_REAL_WEIGHTS"
        cfg["output_root"] = str(root / "outputs")
        questions = [{"problem_id": "gsm1", "benchmark": "gsm", "problem": "SYNTHETIC question 1", "answer": "1"},
                     {"problem_id": "math1", "benchmark": "math", "problem": "SYNTHETIC question 2", "answer": "1"}]
        write_rows(root / "questions.jsonl", questions)
        cfg["evaluation"]["development_manifest"] = str(root / "questions.jsonl")
        cfg["evaluation"]["development_sha256"] = sha(root / "questions.jsonl")
        paths = []
        for method, seed, values in (("parent", None, [False, False]), ("B", 42, [True, False]),
                                     ("C0", 42, [True, True]), ("B", 17, [True, False]), ("C0", 17, [False, True])):
            predictions = [{"problem_id": q["problem_id"], "benchmark": q["benchmark"],
                            "status": "correct" if correct else "incorrect", "correct": correct,
                            "generated_text": "SYNTHETIC", "extracted_answer": "1", "generated_tokens": 1,
                            "stop_reason": "eos", "flags": []} for q, correct in zip(questions, values)]
            predpath = root / f"{method}_{seed}_predictions.jsonl"
            write_rows(predpath, predictions)
            checkpoint = digest([method, seed])
            config_id = f"SYNTHETIC_CONFIG_{seed}"
            rec = None if method == "parent" else {
                "valid_controlled_run": True, "training_smoke_only": False, "method": method,
                "config_fingerprint": config_id, "parent_fingerprint": cfg["parent"]["adapter_sha256"],
                "checkpoint_sha256": checkpoint, "actual_tokens": 100, "expected_tokens": 100,
                "target_tokens": 100, "budget_tolerance": 0.005, "candidate_sha256": "SYNTHETIC_POOL"}
            report = {"synthetic": False, "fixture_notice": "SYNTHETIC TEST; gate explicitly mocked; never real evidence",
                      "method": method, "split": "development", "training_seed": seed,
                      "checkpoint_sha256": checkpoint, "parent_fingerprint": cfg["parent"]["adapter_sha256"],
                      "config_fingerprint": config_id, "method_definition": method_definition(cfg),
                      "evaluation_contract_sha256": cfg["evaluation"]["contract_sha256"],
                      "questions_sha256": cfg["evaluation"]["development_sha256"], "predictions": str(predpath),
                      "predictions_sha256": sha(predpath), "training_reconciliation": rec}
            path = root / f"{method}_{seed}_report.json"
            write_json(path, report)
            paths.append(path)
        with patch("capstone_person4.evaluation.gate", return_value={}):
            result = compare(cfg, paths, "SYNTHETIC_GOOD")
            self.assertEqual(result["training_seed_count"], 2)
            self.assertIsNotNone(result["training_seed_variability"]["gsm"]["training_seed_sd_pp"])
            changed = read_json(paths[-1])
            changed["training_reconciliation"]["target_tokens"] = 200
            write_json(root / "SYNTHETIC_BAD_REPORT.json", changed)
            with self.assertRaisesRegex(PipelineError, "consumed budgets"):
                compare(cfg, paths[:-1] + [root / "SYNTHETIC_BAD_REPORT.json"], "SYNTHETIC_BAD")

    def test_method_definition_allows_only_training_seed_replication(self):
        from capstone_person4.evaluation import method_definition
        cfg = load_config(REPO / "configs/c0.example.json")
        changed = deepcopy(cfg)
        changed["training"]["seed"] = 17
        self.assertEqual(method_definition(cfg), method_definition(changed))
        changed["allocation"]["seed"] = 17
        self.assertNotEqual(method_definition(cfg), method_definition(changed))

    def test_paired_difference_uses_same_ids(self):
        report = paired_difference({"a": True, "b": True}, {"a": True, "b": False}, resamples=100)
        self.assertEqual(report["difference_pp"], 50)
        self.assertEqual(report, paired_difference({"a": True, "b": True}, {"a": True, "b": False}, resamples=100))
        with self.assertRaises(PipelineError):
            paired_difference({"a": True}, {"b": True})

    def test_execution_errors_do_not_become_mathematical_errors(self):
        prediction = {"problem_id": "a", "status": "execution_error", "correct": False,
                      "generated_text": "", "extracted_answer": None, "generated_tokens": 0, "stop_reason": "error", "flags": []}
        with self.assertRaisesRegex(PipelineError, "Retry"):
            validate_predictions([prediction], [{"problem_id": "a"}])
        prediction["status"] = "unresolved_equivalence"
        self.assertEqual(validate_predictions([prediction], [{"problem_id": "a"}]), [prediction])


if __name__ == "__main__":
    unittest.main()

"""Phase 1: question-only generation request, ingest, mistake indicator, C1 utilities, comparison."""

from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from capstone_person4_approachC1 import generation_scoring as gs
from capstone_person4.config import load_config
from capstone_person4.provenance import PipelineError, digest, read_json, read_rows, sha, write_json, write_rows

HAS_SOLVER = importlib.util.find_spec("ortools") is not None


def candidate(identity, source="orca", tokens=20, bin_=0):
    return {"row_id": identity, "problem_group_id": "group-" + identity, "dataset_id": source, "original_source": "fixture",
            "problem": f"SYNTHETIC question {identity}", "completion": f"SYNTHETIC solution text {identity}",
            "solution": f"SYNTHETIC solution text {identity}", "final_answer": "7",
            "prompt_tokens": 5, "completion_tokens": tokens - 1, "eos_tokens": 1, "supervised_tokens": tokens,
            "total_tokens": tokens + 5, "boundary_crossing_token": False, "length_bin": bin_}


def prediction(identity, parse="ok", correct=False, finish="eos", **extra):
    return {"problem_id": identity, "parse_status": parse, "correct": correct, "finish_reason": finish,
            "parsed_answer": "7" if parse == "ok" else None, "generated_tokens": 12, "parser_version": "p3-parser-v6", **extra}


class ClassifierTests(unittest.TestCase):
    def test_only_checked_wrong_answers_are_mistakes(self):
        cases = {("ok", True, "eos"): ("correct", []), ("ok", False, "eos"): ("incorrect", []),
                 ("no_answer", False, "eos"): ("no_answer", []), ("unparseable_number", False, "eos"): ("no_answer", []),
                 ("no_answer", False, "length"): ("truncated", ["hit_length_limit"]),
                 ("ok", False, "length"): ("incorrect", ["hit_length_limit"]), ("ok", "True", "length"): ("correct", ["hit_length_limit"]),
                 ("ambiguous", False, "eos"): ("unresolved_equivalence", []), ("error", False, "eos"): ("execution_error", [])}
        for (parse, correct, finish), (status, flags) in cases.items():
            with self.subTest(parse=parse, correct=correct, finish=finish):
                got_status, got_correct, got_flags = gs.classify(prediction("x", parse, correct, finish))
                self.assertEqual((got_status, got_flags), (status, flags))
                self.assertEqual(got_correct, status == "correct")
        with self.assertRaisesRegex(PipelineError, "Unknown checker outcome"):
            gs.classify(prediction("x", "something_new", False))


class UtilityTests(unittest.TestCase):
    def setUp(self):
        self.fit = [{"row_id": f"r{i}", "fit": f} for i, f in enumerate((1.0, 0.75, 0.5, 0.25, 0.0))]
        self.ids = [r["row_id"] for r in self.fit]

    def test_lambda_zero_and_constant_mistakes_reproduce_c0_up_to_uniform_scaling(self):
        base = gs.c1_utilities(self.fit, dict.fromkeys(self.ids, False), 1_000_000, Fraction(0))
        self.assertEqual(base, {"r0": 1_000_000, "r1": 750_000, "r2": 500_000, "r3": 250_000, "r4": 0})
        mixed_zero = gs.c1_utilities(self.fit, {"r0": True, "r1": False, "r2": True, "r3": False, "r4": True}, 1_000_000, Fraction(0))
        self.assertEqual(mixed_zero, base)
        for constant, factor in ((False, 1), (True, 2)):
            scaled = gs.c1_utilities(self.fit, dict.fromkeys(self.ids, constant), 1_000_000, Fraction(1))
            self.assertEqual(scaled, {k: v * factor for k, v in base.items()})
        half = gs.c1_utilities(self.fit, dict.fromkeys(self.ids, True), 1_000_000, Fraction(1, 2))
        self.assertEqual(half, {k: v * 3 for k, v in base.items()})

    def test_mixed_mistakes_double_only_mistaken_rows_and_coverage_is_enforced(self):
        mixed = gs.c1_utilities(self.fit, {"r0": False, "r1": True, "r2": False, "r3": True, "r4": False}, 1_000_000, Fraction(1))
        self.assertEqual(mixed, {"r0": 1_000_000, "r1": 1_500_000, "r2": 500_000, "r3": 500_000, "r4": 0})
        with self.assertRaisesRegex(PipelineError, "entire universe"):
            gs.c1_utilities(self.fit, {"r0": True}, 1_000_000, Fraction(1))
        with self.assertRaisesRegex(PipelineError, "entire universe"):
            gs.c1_utilities(self.fit, {k: 1 for k in self.ids}, 1_000_000, Fraction(1))


class RequestAndIngestTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("gen-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.cfg = load_config(REPO / "configs/c0.example.json")
        template = read_json(REPO / "configs/c0.v4.0.5b.seed17.json")
        self.cfg["generation"] = dict(template["generation"])
        self.cfg["selection"] = dict(template["selection"])
        self.cfg["parent"] = dict(template["profiles"]["0.5B"])
        contract = self.root / "synthetic_contract.json"
        write_json(contract, {"synthetic": True, "purpose": "unit-test checksum fixture; no real evaluator"})
        self.cfg["generation"]["generation_contract"] = str(contract)
        self.cfg["generation"]["contract_sha256"] = sha(contract)
        self.cfg["output_root"] = str(self.root / "outputs")
        self.rows = [candidate("a"), candidate("b", "numina", 30, 1), candidate("c", "orca", 25, 1)]
        self.preparation = {"candidate_sha256": digest(self.rows)}
        self.patcher = patch("capstone_person4_approachC1.generation_scoring.load_prepared", return_value=(self.rows, self.preparation))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def export(self):
        return gs.export_request(self.cfg)

    def good_report(self, request):
        g = self.cfg["generation"]
        return {"model": request["model"], "revision": request["revision"], "checkpoint_sha256": request["adapter_sha256"],
                "questions_sha256": request["questions_sha256"], "contract_sha256": g["contract_sha256"],
                "prompt_sha256": g["prompt_sha256"], "parser_version": g["parser_version"], "precision": g["precision"],
                "decoding": g["decoding"], "max_new_tokens": g["max_new_tokens"], "eos_token_id": g["eos_token_id"],
                "add_special_tokens": g["add_special_tokens"], "adapter_mode": g["adapter_mode"]}

    def test_request_carries_questions_and_gold_only_and_binds_the_policy(self):
        request = self.export()
        questions = read_rows(Path(request["request_directory"]) / "questions.jsonl")
        self.assertEqual([q["problem_id"] for q in questions], ["a", "b", "c"])
        self.assertEqual(set(questions[0]), set(gs.REQUEST_FIELDS))
        text = (Path(request["request_directory"]) / "questions.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("solution text", text)
        self.assertIn("SYNTHETIC question a", text)
        self.assertEqual(request["prompt_sha256"], "22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db")
        self.assertEqual(request["decoding"]["precision"], "fp32")
        self.assertEqual(request["adapter_sha256"], self.cfg["parent"]["adapter_sha256"])
        # The key binds parent, universe and policy, never code or paths.
        other = dict(self.cfg, generation={**self.cfg["generation"], "parser_version": "p3-parser-v7"})
        self.assertNotEqual(gs.generation_key(other, self.preparation), request["generation_key"])
        self.assertEqual(gs.generation_key(dict(self.cfg, output_root="/elsewhere"), self.preparation), request["generation_key"])

    def test_ingest_freezes_outcomes_and_only_incorrect_counts_as_mistake(self):
        request = self.export()
        preds = self.root / "preds.jsonl"
        write_rows(preds, [prediction("a", "ok", False), prediction("b", "ok", True), prediction("c", "no_answer", False, "length")])
        report = self.root / "config.json"
        write_json(report, self.good_report(request))
        coverage = gs.ingest(self.cfg, preds, report)
        self.assertEqual(coverage["status_counts"], {"incorrect": 1, "correct": 1, "truncated": 1})
        self.assertEqual(coverage["mistakes"], 1)
        outcomes, loaded = gs.load_generation(self.cfg)
        self.assertEqual({o["row_id"]: o["mistake"] for o in outcomes}, {"a": True, "b": False, "c": False})
        self.assertEqual(loaded["generation_key"], request["generation_key"])
        # A changed policy makes the stored outcomes stale rather than silently reused.
        stale = dict(self.cfg, generation={**self.cfg["generation"], "max_new_tokens": 512})
        with self.assertRaisesRegex(PipelineError, "stale"):
            gs.load_generation(stale)

    def test_ingest_rejects_partial_coverage_wrong_run_and_execution_errors(self):
        request = self.export()
        report = self.root / "config.json"
        write_json(report, self.good_report(request))
        preds = self.root / "partial.jsonl"
        write_rows(preds, [prediction("a"), prediction("b", "ok", True)])
        with self.assertRaisesRegex(PipelineError, "exactly once"):
            gs.ingest(self.cfg, preds, report)
        wrong = self.root / "wrong_config.json"
        write_json(wrong, {**self.good_report(request), "checkpoint_sha256": "0" * 64})
        full = self.root / "full.jsonl"
        write_rows(full, [prediction("a"), prediction("b", "ok", True), prediction("c")])
        with self.assertRaisesRegex(PipelineError, "differs from the request"):
            gs.ingest(self.cfg, full, wrong)
        errors = self.root / "errors.jsonl"
        write_rows(errors, [prediction("a"), prediction("b", "ok", True), prediction("c", "error")])
        with self.assertRaisesRegex(PipelineError, "execution errors must be retried"):
            gs.ingest(self.cfg, errors, report)
        other_parser = self.root / "parser.jsonl"
        write_rows(other_parser, [prediction("a", parser_version="p3-parser-v5"), prediction("b", "ok", True), prediction("c")])
        with self.assertRaisesRegex(PipelineError, "another parser version"):
            gs.ingest(self.cfg, other_parser, report)

    def test_ingest_rejects_same_ids_with_wrong_question_or_contract_identity(self):
        request = self.export()
        preds = self.root / "predictions.jsonl"
        write_rows(preds, [prediction("a"), prediction("b"), prediction("c")])
        for field in ("questions_sha256", "contract_sha256"):
            wrong = self.root / (field + ".json")
            write_json(wrong, {**self.good_report(request), field: "0" * 64})
            with self.assertRaisesRegex(PipelineError, "differs from the request"):
                gs.ingest(self.cfg, preds, wrong)


class ComparisonWithC1Tests(unittest.TestCase):
    def test_primary_contrast_is_c1_minus_c0_when_c1_present(self):
        from capstone_person4.evaluation import compare, method_definition
        root = REPO / ".test_artifacts" / ("cmp-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        cfg = load_config(REPO / "configs/c0.example.json")
        cfg["parent"]["adapter_sha256"] = "SYNTHETIC_NO_REAL_WEIGHTS"
        cfg["output_root"] = str(root / "outputs")
        questions = [{"problem_id": "gsm1", "benchmark": "gsm", "problem": "SYNTHETIC question 1", "answer": "1"},
                     {"problem_id": "gsm2", "benchmark": "gsm", "problem": "SYNTHETIC question 2", "answer": "1"}]
        write_rows(root / "questions.jsonl", questions)
        cfg["evaluation"]["development_manifest"] = str(root / "questions.jsonl")
        cfg["evaluation"]["development_sha256"] = sha(root / "questions.jsonl")
        paths = []
        for method, seed, values in (("parent", None, [False, False]), ("C0", 42, [True, False]), ("C1", 42, [True, True]),
                                     ("C0", 137, [False, False]), ("C1", 137, [True, False])):
            predictions = [{"problem_id": q["problem_id"], "benchmark": q["benchmark"], "status": "correct" if c else "incorrect",
                            "correct": c, "generated_text": "SYNTHETIC", "extracted_answer": "1", "generated_tokens": 1,
                            "stop_reason": "eos", "flags": []} for q, c in zip(questions, values)]
            predpath = root / f"{method}_{seed}.jsonl"
            write_rows(predpath, predictions)
            rec = None if method == "parent" else {
                "valid_controlled_run": True, "training_smoke_only": False, "method": method, "config_fingerprint": f"CFG{seed}",
                "parent_fingerprint": cfg["parent"]["adapter_sha256"], "checkpoint_sha256": digest([method, seed]),
                "actual_tokens": 100, "expected_tokens": 100, "target_tokens": 100, "budget_tolerance": 0.005, "candidate_sha256": "POOL"}
            path = root / f"{method}_{seed}_report.json"
            write_json(path, {"synthetic": False, "method": method, "split": "development", "training_seed": seed,
                              "checkpoint_sha256": digest([method, seed]), "parent_fingerprint": cfg["parent"]["adapter_sha256"],
                              "config_fingerprint": f"CFG{seed}", "method_definition": method_definition(cfg),
                              "evaluation_contract_sha256": cfg["evaluation"]["contract_sha256"],
                              "questions_sha256": cfg["evaluation"]["development_sha256"], "predictions": str(predpath),
                              "predictions_sha256": sha(predpath), "training_reconciliation": rec})
            paths.append(path)
        with patch("capstone_person4.evaluation.gate", return_value={}):
            result = compare(cfg, paths, "SYNTHETIC_C1")
        self.assertEqual(result["primary_contrast"], "C1_minus_C0")
        self.assertEqual(result["arms"], ["C0", "C1"])
        by_seed = {r["training_seed"]: r for r in result["results"]}
        self.assertEqual(by_seed[42]["C1_minus_C0"]["difference_pp"], 50.0)
        self.assertEqual(by_seed[137]["C1_minus_C0"]["difference_pp"], 50.0)
        self.assertEqual(result["training_seed_variability"]["gsm"]["mean_C1_minus_C0_pp"], 50.0)
        text = (Path(cfg["output_root"]) / "0.5B").rglob("comparison.md")
        self.assertIn("C1 versus C0", next(text).read_text(encoding="utf-8"))
        with patch("capstone_person4.evaluation.gate", return_value={}), self.assertRaisesRegex(PipelineError, "at least one other"):
            compare(cfg, [paths[0], paths[1], paths[3]], "SYNTHETIC_C0_ONLY")


if __name__ == "__main__":
    unittest.main()

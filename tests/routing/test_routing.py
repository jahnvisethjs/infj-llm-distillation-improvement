"""Synthetic scientific-integrity tests. No downloads, Qwen weights, or GPU jobs."""

from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from capstone_routing import analysis, common, evaluation, inputs, labeling, live, pool, router
from capstone_routing.__main__ import parser

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "capstone_replay15b_recovery/tests/fixtures"


class FixtureCase(unittest.TestCase):
    def setUp(self):
        # Retained evidence, rather than automatic recursive deletion.
        self.root = REPO / ".test_artifacts" / ("routing-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)


class Artifacts(FixtureCase):
    def test_create_only_and_integrity(self):
        common.write(self.root / "item.json", {"a": 1})
        common.seal(self.root, "fixture", synthetic=True)
        common.bundle(self.root, "fixture")
        with self.assertRaises(common.RoutingError):
            common.write(self.root / "item.json", {"a": 2})
        (self.root / "item.json").write_text("{}")
        with self.assertRaises(common.RoutingError):
            common.bundle(self.root)

    def test_destination_cannot_overlap_inputs(self):
        for output in (self.root, self.root / "input" / "child", self.root / "input"):
            with self.assertRaises(common.RoutingError):
                common.destination(output, [self.root / "input"])

    def test_all_commands_are_explicit(self):
        commands = parser()._subparsers._group_actions[0].choices
        self.assertEqual(set(commands), {"inspect", "opportunity", "patterns", "prepare-pool", "label", "embed",
                                         "train", "freeze", "evaluate-test", "report", "route", "answer"})
        args = parser().parse_args(["label", "--config", "c", "--registry", "r", "--pool", "p", "--split", "test", "--output", "o"])
        self.assertFalse(args.authorize_gpu)
        self.assertIsNone(args.frozen)


class OpportunityTests(FixtureCase):
    def test_opportunity_pattern_and_report_artifacts_from_recorded_shape(self):
        questions = {(b, f"q{i}"): {"benchmark": b, "problem_id": f"q{i}",
                     "question": f"SYNTHETIC {b} question {i}: compute a fraction 1/2."}
                     for b in ("gsm", "math") for i in range(4)}
        candidates, predictions = {}, {}
        for method, seed in (("A", 42), ("B", 42), ("C1", 42), ("A", 137), ("A", 2026)):
            cid = common.identity(17, method, seed)
            candidate = {"id": cid, "parent_seed": 17, "continuation_seed": seed, "method": method,
                         "checkpoint_sha256": common.digest(cid), "train_manifest_sha256": "d" * 64,
                         "parent_sha256": "p" * 64, "actual_tokens": 318845}
            candidates[cid] = candidate
            for bench in ("gsm", "math"):
                config = {**common.SETTINGS, "checkpoint_sha256": candidate["checkpoint_sha256"],
                          "train_manifest_sha256": candidate["train_manifest_sha256"], "train_seed": seed,
                          "train_arm": {"A": "REPEAT", "B": "RANDOM", "C1": "C1"}[method],
                          "eval_file_sha256": common.QUESTIONS[bench], "eval_config_hash": "fixture"}
                path = self.root / "inputs" / cid / bench
                common.write(path / "predictions.config.json", config)
                common.write_rows(path / "predictions.jsonl", [{"benchmark": bench, "problem_id": f"q{i}",
                    "raw_output": "Final answer: 1", "parsed_answer": "1", "parse_status": "ok",
                    "correct": bool((i % 2) == (method == "A")), "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "train_manifest_sha256": candidate["train_manifest_sha256"], "model_revision": common.MODEL + "@" + common.REVISION,
                    "parser_version": "p3-parser-v6", "eval_config_hash": "fixture"} for i in range(4)])
                predictions[cid + "/" + bench] = [{"candidate": cid, "benchmark": bench,
                    "predictions": str(path / "predictions.jsonl"), "config": str(path / "predictions.config.json")}]
        registry = {"candidates": candidates, "predictions": predictions, "rejected": [], "synthetic": True}
        root = self.root / "opportunity"
        with patch.object(analysis, "inspect", return_value=registry), patch.object(analysis, "development_questions", return_value=(questions, {})):
            summary = analysis.opportunity({}, root)
        self.assertEqual(summary["complete_block_count"], 4)
        self.assertFalse(summary["complete"])
        import csv
        with (root / "opportunity.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        joint = next(r for r in metrics if r["experiment"] == "policy" and r["benchmark"] == "combined" and r["alternative"] == "joint")
        self.assertEqual(float(joint["oracle_accuracy"]), 1.)
        self.assertEqual(float(joint["available_improvement_pp"]), 50.)
        analysis.patterns(root, self.root / "patterns")
        evaluation.report(root, self.root / "report")
        self.assertIn("SYNTHETIC", (self.root / "report/report.md").read_text())

    def prediction_fixture(self):
        questions = {("gsm", "q" + str(i)): {"benchmark": "gsm", "problem_id": "q" + str(i)} for i in range(4)}
        candidate = {"checkpoint_sha256": "c" * 64, "train_manifest_sha256": "d" * 64,
                     "continuation_seed": 42, "method": "A"}
        config = {**common.SETTINGS, "checkpoint_sha256": candidate["checkpoint_sha256"],
                  "train_manifest_sha256": candidate["train_manifest_sha256"], "train_seed": 42,
                  "train_arm": "REPEAT", "eval_file_sha256": "e" * 64, "eval_config_hash": "hash"}
        records = [{"benchmark": "gsm", "problem_id": "q" + str(i), "correct": bool(i % 2),
                    "parse_status": "ok", "parsed_answer": "1", "raw_output": "Final answer: 1",
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "train_manifest_sha256": candidate["train_manifest_sha256"],
                    "model_revision": common.MODEL + "@" + common.REVISION,
                    "parser_version": "p3-parser-v6", "eval_config_hash": "hash"} for i in reversed(range(4))]
        spec = {"predictions": str(self.root / "predictions.jsonl"), "config": str(self.root / "predictions.config.json"),
                "questions_sha256": "e" * 64}
        return questions, candidate, config, records, spec

    def validate(self, config, records, spec, candidate, questions):
        Path(spec["config"]).write_text(json.dumps(config))
        Path(spec["predictions"]).write_text("".join(json.dumps(r) + "\n" for r in records))
        return inputs.prediction_records(spec, candidate, questions)

    def test_four_quadrants_and_joint_rescues_are_not_added(self):
        matrix = [{"A": 0, "B": 1, "C1": 1}, {"A": 1, "B": 0, "C1": 0},
                  {"A": 1, "B": 1, "C1": 0}, {"A": 0, "B": 0, "C1": 0}]
        b, _, joint = analysis.opportunity_metrics(matrix, ("A", "B", "C1"), "A")
        self.assertEqual([b[k] for k in ("rescues", "risks", "both_correct", "both_wrong")], [1, 1, 1, 1])
        self.assertEqual(joint["rescues"], 1)
        self.assertEqual(joint["oracle_accuracy"], .75)
        self.assertEqual(b["accuracy_difference_pp"], 0)

    def test_shuffled_and_missing_rows_keep_full_denominator(self):
        questions, cp, config, records, spec = self.prediction_fixture()
        records = records[1:]
        values, counts = self.validate(config, records, spec, cp, questions)
        self.assertEqual(len(values), 4)
        self.assertEqual(values["gsm", "q3"], 0)
        self.assertEqual(counts["missing_rows"], 1)

    def test_empty_and_unparseable_answers_are_zero(self):
        questions, cp, config, records, spec = self.prediction_fixture()
        for status in ("empty", "empty_marker", "no_answer", "ambiguous", "unparseable_number", "unparseable_math"):
            records[0].update(correct=False, parse_status=status, parsed_answer=None)
            values, _ = self.validate(config, records, spec, cp, questions)
            self.assertEqual(values["gsm", "q3"], 0)

    def test_corruption_unknown_duplicate_and_checker_failure_are_rejected(self):
        questions, cp, config, records, spec = self.prediction_fixture()
        for field, value in (("parse_status", "verify_error"), ("problem_id", "unknown"),
                             ("checkpoint_sha256", "wrong"), ("parser_version", "wrong"),
                             ("parsed_answer", None)):
            changed = deepcopy(records)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(common.RoutingError):
                self.validate(config, changed, spec, cp, questions)
        with self.assertRaises(common.RoutingError):
            self.validate(config, records + records[:1], spec, cp, questions)

    def test_mixed_runtime_seed_arm_or_evaluator_is_rejected(self):
        questions, cp, config, records, spec = self.prediction_fixture()
        for field, value in (("torch", "other"), ("train_seed", 137), ("train_arm", "RANDOM"),
                             ("evaluate_py_sha256", "wrong"), ("eval_file_sha256", "other")):
            with self.subTest(field=field), self.assertRaises(common.RoutingError):
                self.validate({**config, field: value}, records, spec, cp, questions)

    def test_benchmark_is_part_of_question_identity(self):
        values = common.keyed([{"benchmark": b, "problem_id": "same"} for b in ("gsm", "math")])
        self.assertEqual(len(values), 2)

    def test_features_only_use_question(self):
        f = analysis.features("A 1/2 meter rope is 20 percent longer if x=2 and y<3. Find the ratio.")
        self.assertTrue(f["fractions"] and f["percentages"] and f["units"] and f["multiple_constraints"])
        self.assertIn("rates_ratios", f["topics"])


class PoolTests(FixtureCase):
    def test_exposure_propagates_through_related_family_components(self):
        source = REPO.parent / "data/inputs/overlap_evidence/overlap_check.py"
        rules = pool.matcher({"path": str(source), "sha256": common.sha(source)})
        a = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
        b = a.replace("echo", "oscar")
        c = b.replace("juliet", "papa")
        index = pool.QuestionIndex(rules)
        self.assertTrue(pool.related(index.describe({"question": a}), index.describe({"question": b}), rules))
        self.assertFalse(pool.related(index.describe({"question": a}), index.describe({"question": c}), rules))
        population = [{"benchmark": "gsm", "problem_id": str(i), "question": text,
                       "gold_answer": "1", "exposure_matches": int(i == 0)} for i, text in enumerate((a, b, c, "An entirely unrelated geometric task."))]
        eligible, excluded = pool.screen_families(population, rules)
        self.assertEqual([r["problem_id"] for r in eligible], ["3"])
        self.assertEqual({r["problem_id"] for r in excluded}, {"0", "1", "2"})

    def test_frozen_matcher_and_short_numeric_families(self):
        source = REPO.parent / "data/inputs/overlap_evidence/overlap_check.py"
        rules = pool.matcher({"path": str(source), "sha256": common.sha(source)})
        index = pool.QuestionIndex(rules)
        index.add(index.describe({"question": "Compute 12 + 34."}))
        self.assertEqual(index.matches(index.describe({"question": "Compute 56 + 78."})), [0])
        self.assertEqual(index.matches(index.describe({"question": "Name the capital city of Peru."})), [])

    def test_deterministic_splits_keep_family_together(self):
        records = [{"benchmark": bench, "problem_id": f"{bench}-{i}", "family_id": f"f{i // 2}", "question": str(i)}
                   for bench in ("gsm", "math") for i in range(40)]
        first = pool.split_families(records)
        self.assertEqual(first, pool.split_families(list(reversed(records)))[::-1])
        memberships = {}
        for row in first:
            memberships.setdefault(row["family_id"], set()).add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in memberships.values()))
        self.assertEqual({r["split"] for r in first}, {"train", "validation", "test"})

    def test_missing_exposure_inventory_blocks(self):
        with self.assertRaises(common.RoutingError):
            pool.audit_exposures([], {"candidates": {}})


class CheckpointTests(FixtureCase):
    def fixture(self):
        parent = self.root / "parent/final/adapter"
        cp = self.root / "run/final/adapter"
        common.write_bytes(parent / "adapter_model.safetensors", b"SYNTHETIC parent, not real weights")
        common.write(parent.parent / "run_config.json", {"model": common.MODEL, "revision": common.REVISION,
                     "seed": 17, "manifest_sha256": "e" * 64})
        common.write_bytes(cp / "adapter_model.safetensors", b"SYNTHETIC continuation, not real weights")
        common.write(cp / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": common.MODEL})
        training = {"model": common.MODEL, "revision": common.REVISION, "parent_adapter_sha256": common.sha(parent / "adapter_model.safetensors"),
                    "parent": str(parent.parent), "arm": "REPEAT", "seed": 42, "epochs": 1, "manifest_sha256": "c" * 64,
                    "mode": "branch", "branch_budget": {"target_supervised_tokens": 318845}}
        common.write(cp.parent / "run_config.json", training)
        common.write(cp.parent.parent / "training_summary.json", {"complete": True, "processed_tokens": 318845})
        return {"checkpoint": str(cp)}, training

    def test_lineage_seed_budget_and_screened_c1_checks(self):
        spec, training = self.fixture()
        path = Path(spec["checkpoint"]).parent / "run_config.json"
        with patch.dict(inputs.PARENT_HASHES, {17: training["parent_adapter_sha256"]}):
            self.assertEqual(inputs.checkpoint_record(spec)["id"], "p17_A_t42")
            with self.assertRaises(common.RoutingError):
                inputs.checkpoint_record({**spec, "continuation_seed": 137})
            path.write_text(json.dumps({**training, "parent_adapter_sha256": "f" * 64}))
            with self.assertRaises(common.RoutingError):
                inputs.checkpoint_record(spec)
            path.write_text(json.dumps({**training, "arm": "C1", "shared_universe_sha256": "wrong"}))
            with self.assertRaises(common.RoutingError):
                inputs.checkpoint_record(spec)
            path.write_text(json.dumps(training))
            (path.parent.parent / "training_summary.json").write_text(json.dumps({"complete": True, "processed_tokens": 1}))
            with self.assertRaises(common.RoutingError):
                inputs.checkpoint_record(spec)

    def test_inspect_is_read_only_and_duplicate_id_does_not_pick_first(self):
        spec, training = self.fixture()
        cfg = {"checkpoints": [spec, spec], "roots": {k: str(self.root / "absent") for k in ("a", "b", "c1")}}
        before = {str(p): common.sha(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.dict(inputs.PARENT_HASHES, {17: training["parent_adapter_sha256"]}):
            result = inputs.inspect(cfg)
        self.assertNotIn("p17_A_t42", result["candidates"])
        self.assertTrue(any("Duplicate" in r["reason"] for r in result["rejected"]))
        self.assertEqual(before, {str(p): common.sha(p) for p in self.root.rglob("*") if p.is_file()})


class RouterTests(FixtureCase):
    def test_long_question_embedding_keeps_every_token_and_batch_axis(self):
        import numpy as np
        from types import ModuleType
        encoder = router.QuestionEncoder.__new__(router.QuestionEncoder)
        encoder.limit = 3
        chunks = []
        def prepare(ids, **kwargs):
            self.assertTrue(kwargs["prepend_batch_axis"])
            self.assertFalse(kwargs["truncation"])
            chunks.append(list(ids))
            return {"input_ids": ids}
        encoder.tokenizer = SimpleNamespace(encode=lambda *a, **kw: list(range(7)), prepare_for_model=prepare)
        class Vector:
            def float(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return np.array([1., 2.] + [0.] * 382)
        class Model:
            def eval(self):
                pass
            def __call__(self, features):
                return {"sentence_embedding": [Vector()]}
        encoder.model = Model()
        fake_torch = ModuleType("torch")
        fake_torch.inference_mode = nullcontext
        with patch.dict("sys.modules", {"torch": fake_torch}):
            vector = encoder.encode("long question")
        self.assertEqual(chunks, [[0, 1, 2], [3, 4, 5], [6]])
        self.assertAlmostEqual(sum(v * v for v in vector), 1.)

    def test_threshold_is_strict_and_ties_are_stable(self):
        scores = {"A": .5, "B": .75, "C1": .75}
        self.assertEqual(router.choose(scores, "A", ["A", "B", "C1"], .25), "A")
        self.assertEqual(router.choose(scores, "A", ["A", "C1", "B"], .2), "B")
        self.assertEqual(router.choose(scores, "A", ["A"], 0), "A")

    def test_constant_and_fitted_classifier(self):
        self.assertEqual(router.fit_binary([[1], [2]], [0, 0], 1)["constant"], .25)
        model = router.fit_binary([[-2], [-1], [1], [2]], [0, 0, 1, 1], 1)
        self.assertGreater(router.probability(model, [2]), router.probability(model, [-2]))

    def test_diversity_comparison_is_three_candidates_same_parent(self):
        scenarios = router.scenarios()
        self.assertEqual(len(scenarios), 18)
        for row in scenarios:
            if row["experiment"] != "primary":
                self.assertEqual(len(row["pools"]), 1)
                self.assertEqual(len(row["candidates"]), 3)
                self.assertIn(row["reference"], row["candidates"])
            if row["experiment"] == "seed_three":
                self.assertEqual(row["candidates"], [common.identity(row["parent_seed"], "A", s) for s in common.SEEDS])

    def test_family_bootstrap_uses_same_draw_for_all_contrasts(self):
        questions = [{"family_id": "f1"}, {"family_id": "f1"}, {"family_id": "f2"}]
        result = evaluation.paired_bootstrap(questions, {"a": [1, 1, -1], "b": [2, 2, -2]}, 100)
        self.assertAlmostEqual(result["a"]["difference_pp"] * 2, result["b"]["difference_pp"])
        self.assertAlmostEqual(result["a"]["ci95_low_pp"] * 2, result["b"]["ci95_low_pp"])
        self.assertEqual(result["a"]["family_count"], 2)

    def test_random_control_matches_selection_counts(self):
        expected, report = evaluation.random_control({"A": [1, 1, 0, 0], "B": [0, 0, 1, 1]}, ["A", "A", "A", "B"], 100)
        self.assertEqual(report["selection_counts"], {"A": 3, "B": 1})
        self.assertEqual(expected, [.75, .75, .25, .25])
        self.assertEqual(report["expected_accuracy"], .5)

    def test_switch_identity(self):
        result = evaluation.switch_counts([0, 1, 1, 0], [1, 0, 1, 0], ["B", "B", "A", "B"], "A")
        self.assertEqual(result["successful_switches"], 1)
        self.assertEqual(result["harmful_switches"], 1)
        self.assertEqual(result["difference_pp"], 0)


class NativeBridgeTests(FixtureCase):
    def test_live_completion_uses_frozen_decoding_and_strips_eos(self):
        generator = live.VerifiedGenerator.__new__(live.VerifiedGenerator)
        class Batch(dict):
            def __init__(self):
                self.input_ids = SimpleNamespace(shape=(1, 5))
                super().__init__(input_ids=self.input_ids)
            def to(self, device):
                return self
        class Generated:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [9, 151643, 151643])
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return Generated()
        class Tokenizer:
            pad_token_id = 151643
            def __call__(self, prompts, **kwargs):
                self.prompts = prompts
                return Batch()
            def decode(self, ids, **kwargs):
                self.ids = ids
                return "Final answer: 4"
        tok = Tokenizer()
        generator.state = {"tok": tok, "model": SimpleNamespace(device="cpu", generate=generate),
                           "torch": SimpleNamespace(inference_mode=nullcontext), "ctx_limit": 32768}
        generator.native = {"render_prompt": lambda q, mode: "Question: " + q}
        generator.load = lambda cp: (.01, False)
        generator.sync = lambda: None
        candidate = {k: k for k in ("id", "checkpoint", "checkpoint_sha256", "model", "revision", "parent_seed", "parent_checkpoint", "parent_sha256",
                                    "continuation_seed", "method", "train_manifest_sha256", "recipe")}
        value = generator.generate("2+2?", candidate)
        self.assertEqual(tok.ids, [9])
        self.assertEqual(value["generated_tokens"], 1)
        self.assertEqual(value["answer_text"], "Final answer: 4")
        self.assertEqual(calls[0]["max_new_tokens"], 1024)
        self.assertFalse(calls[0]["do_sample"])
        self.assertEqual(calls[0]["eos_token_id"], 151643)
        self.assertEqual(value["lineage"]["parent_sha256"], "parent_sha256")
        generator.state["ctx_limit"] = 100
        with self.assertRaises(common.RoutingError):
            generator.generate("2+2?", candidate)
        self.assertEqual(len(calls), 1)

    def test_checkpoint_cache_switch_reconstructs_a_fresh_model(self):
        generator = live.VerifiedGenerator.__new__(live.VerifiedGenerator)
        events = []
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        def parse_args(arguments):
            return SimpleNamespace(adapter=arguments[arguments.index("--adapter") + 1])
        generator.native = {"build_parser": lambda: SimpleNamespace(parse_args=parse_args),
                            "resolve_checkpoint": lambda args: {"checkpoint_sha256": args.adapter},
                            "generation_config": lambda args: {}, "P": SimpleNamespace(PARSER_VERSION="p3-parser-v6", CUT_AT_NEXT_QUESTION=True),
                            "version": lambda name: "0.9.0", "events": events, "fake_torch": fake_torch}
        generator.code = compile("events.append('fresh'); model=object(); tok=object(); device='cpu'; torch=fake_torch", "synthetic", "exec")
        generator.cache_key = generator.state = None
        def cp(cid):
            return {"checkpoint": cid, "checkpoint_sha256": cid, "model": common.MODEL,
                    "revision": common.REVISION, "train_manifest_sha256": "manifest"}
        with patch.object(live, "verify_candidate"), patch.object(live, "settings"):
            first, hit1 = generator.load(cp("first"))
            model1 = generator.state["model"]
            second, hit2 = generator.load(cp("first"))
            third, hit3 = generator.load(cp("second"))
        self.assertEqual((hit1, hit2, hit3), (False, True, False))
        self.assertEqual(events, ["fresh", "fresh"])
        self.assertIsNot(generator.state["model"], model1)

    def evaluator(self):
        return {"path": str(FROZEN / "frozen_evaluate.py"), "parser_path": str(FROZEN / "frozen_answer_parser.py")}

    def test_hash_pinned_native_prompt_and_loader_can_be_reused(self):
        native = live.evaluator_module(self.evaluator())
        self.assertEqual(native["render_prompt"]("  x  + y  ", "p1_clean"),
                         "Question: x + y\nSolve the problem step by step. End with: Final answer: <answer>.\nSolution:\n")
        self.assertIsInstance(live.loader_code(self.evaluator()), type(compile("pass", "", "exec")))
        self.assertEqual(native["P"].score_gsm("Final answer: 2", "#### 2")["correct"], True)

    def test_unverified_save_representation_is_rejected(self):
        with self.assertRaises(common.RoutingError):
            live.verify_candidate({"recipe": None})

    def test_audited_cumulative_save_and_incremental_rejection(self):
        source = self.root / "trainer.py"
        source.write_text("def model_loader(base, adapter):\n    model = PeftModel.from_pretrained(base, adapter)\n    return model\n\ndef save_checkpoint(model, path):\n    model.save_pretrained(path)\n")
        spec = {"path": str(source), "sha256": common.sha(source), "loader_function": "model_loader",
                "save_function": "save_checkpoint", "format": "cumulative_lora_on_base", "parent_loaded_for_continuation": True}
        self.assertEqual(inputs.load_recipe(spec)["format"], "cumulative_lora_on_base")
        with self.assertRaises(common.RoutingError):
            inputs.load_recipe({**spec, "format": "incremental_delta"})

    def test_answer_requires_gpu_authorization_before_loading(self):
        with patch.object(live, "VerifiedGenerator", side_effect=AssertionError("must not load")):
            with self.assertRaises(common.RoutingError):
                live.Answerer("missing", self.evaluator(), False)

    def test_answer_returns_scores_lineage_and_separate_timings(self):
        answerer = live.Answerer.__new__(live.Answerer)
        answerer.encoder_initialization_seconds = .1
        answerer.router = SimpleNamespace(candidates={"cp": {"id": "cp"}}, route=lambda q: {
            "selected_checkpoint": "cp", "routing_scores": {"cp": .8}})
        answerer.generator = SimpleNamespace(generate=lambda q, cp: {"answer_text": "Final answer: 4", "selected_checkpoint": cp["id"],
            "lineage": {"parent_sha256": "parent"}, "timing": {"loading_seconds": .2, "generation_seconds": .3}})
        value = answerer.answer("2+2?")
        self.assertEqual(value["answer_text"], "Final answer: 4")
        self.assertEqual(value["routing_scores"], {"cp": .8})
        self.assertEqual(value["lineage"]["parent_sha256"], "parent")
        self.assertTrue({"routing_seconds", "loading_seconds", "generation_seconds", "end_to_end_seconds"} <= value["timing"].keys())


class SyntheticPipelineTests(FixtureCase):
    def fixtures(self):
        registry_root, pool_root, embedding_root = [self.root / n for n in ("registry", "pool", "embeddings")]
        candidates = {common.identity(p, m, s): {"id": common.identity(p, m, s), "method": m, "parent_seed": p,
                      "continuation_seed": s, "checkpoint_sha256": common.digest([p, m, s]), "parent_sha256": common.digest(p),
                      "actual_tokens": 318845, "construction": {"SYNTHETIC": True}, "checkpoint": f"SYNTHETIC/{p}/{m}/{s}",
                      "train_manifest_sha256": common.digest([p, m]), "model": common.MODEL, "revision": common.REVISION}
                      for p in common.PARENTS for m in common.METHODS for s in common.SEEDS}
        common.write(registry_root / "registry.json", {"candidates": candidates, "synthetic": True})
        common.seal(registry_root, "opportunity", synthetic=True)
        questions = []
        for split, n in (("train", 12), ("validation", 8), ("test", 8)):
            selected = [{"benchmark": "gsm" if i % 2 else "math", "problem_id": f"{split}{i}",
                         "question": f"SYNTHETIC question {split} {i}", "family_id": f"{split}-f{i // 2}", "split": split} for i in range(n)]
            questions.extend(selected)
            common.write_rows(pool_root / (split + ".questions.jsonl"), selected)
            common.write_rows(pool_root / "grading" / (split + ".jsonl"), [dict(q, gold_answer="1") for q in selected])
        common.seal(pool_root, "pool", [registry_root], synthetic=True)
        common.write_rows(embedding_root / "embeddings.jsonl", [{"benchmark": q["benchmark"], "problem_id": q["problem_id"],
                          "question_sha256": common.digest(q["question"]), "vector": [1. if i % 4 < 2 else -1.] + [0.] * 383}
                          for i, q in enumerate(questions)])
        common.write(embedding_root / "encoder.json", {"model": common.ENCODER, "revision": common.ENCODER_REVISION,
                     "policy": "token-chunks-weighted-l2-v1", "dimensions": 384})
        common.seal(embedding_root, "embeddings", [pool_root], synthetic=True)
        return registry_root, pool_root, embedding_root, candidates, questions

    def labels(self, root, pool_root, registry_root, split, candidates, questions, frozen=None):
        for cid, cp in candidates.items():
            subset = [q for q in questions if q["split"] == split]
            values = [{"benchmark": q["benchmark"], "problem_id": q["problem_id"], "question_sha256": common.digest(q["question"]),
                       "correct": int((i % 4 < 2) == (cp["method"] != "A"))} for i, q in enumerate(subset)]
            common.write_rows(root / cid / "outcomes.jsonl", values)
        common.write(root / "labels.json", {"split": split, "checkpoint_hashes": {cid: cp["checkpoint_sha256"] for cid, cp in candidates.items()},
                     "pool_receipt_sha256": common.sha(pool_root / "receipt.json"),
                     "frozen_receipt_sha256": common.sha(frozen / "receipt.json") if frozen else None})
        common.seal(root, "labels", [pool_root, registry_root] + ([frozen] if frozen else []), synthetic=True)

    def test_end_to_end_training_controls_freezing_test_and_no_test_label_read_during_training(self):
        registry_root, pool_root, embedding_root, candidates, questions = self.fixtures()
        train_labels, val_labels, test_labels = [self.root / n for n in ("train-labels", "val-labels", "test-labels")]
        self.labels(train_labels, pool_root, registry_root, "train", candidates, questions)
        self.labels(val_labels, pool_root, registry_root, "validation", candidates, questions)
        training_root, frozen_root, result_root = [self.root / n for n in ("training", "frozen", "test-report")]
        real_read = router.read
        def no_test(path):
            self.assertNotIn("test-labels", str(path))
            return real_read(path)
        with patch.object(router, "load_registry", return_value={"candidates": candidates, "synthetic": True}), patch.object(router, "read", side_effect=no_test):
            trained = router.train(registry_root, pool_root, embedding_root, train_labels, val_labels, training_root)
        self.assertEqual(trained["routers"], 18)
        router.freeze(training_root, frozen_root)
        class Encoder:
            def encode(self, text):
                return [1.] + [0.] * 383
        routed = router.FrozenRouter(frozen_root, encoder=Encoder()).route("SYNTHETIC new question")
        self.assertIn(routed["selected_checkpoint"], candidates)
        self.assertTrue(routed["routing_scores"])
        self.labels(test_labels, pool_root, registry_root, "test", candidates, questions, frozen_root)
        result = evaluation.evaluate_test(frozen_root, pool_root, embedding_root, test_labels, result_root, resamples=30, permutations=20)
        self.assertTrue(result["synthetic"] and result["complete"])
        self.assertEqual(result["test_questions"], 8)
        self.assertGreaterEqual(result["primary"]["difference_pp"], 0)
        self.assertEqual(len(common.rows(result_root / "switches.jsonl")), 18 * 8)
        common.bundle(result_root, "routing_test")
        with self.assertRaises(common.RoutingError):
            evaluation.evaluate_test(frozen_root, pool_root, embedding_root, test_labels, result_root)

    def test_cross_split_family_is_rejected(self):
        registry_root, pool_root, embedding_root, candidates, questions = self.fixtures()
        test = common.rows(pool_root / "test.questions.jsonl")
        test[0]["family_id"] = "train-f0"
        (pool_root / "test.questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test))
        # Restore only the synthetic fixture receipt to exercise semantic checks.
        receipt = common.read(pool_root / "receipt.json")
        receipt["files"]["test.questions.jsonl"] = common.sha(pool_root / "test.questions.jsonl")
        (pool_root / "receipt.json").write_text(json.dumps(receipt))
        with self.assertRaises(common.RoutingError):
            pool.pool_questions(pool_root)

    def test_labeling_cannot_access_test_before_freeze(self):
        registry_root, pool_root, embedding_root, candidates, questions = self.fixtures()
        with patch.object(labeling, "load_registry", return_value={"candidates": candidates, "synthetic": True}):
            with self.assertRaisesRegex(common.RoutingError, "frozen"):
                labeling.label({}, registry_root, pool_root, "test", self.root / "labels")

    def test_native_shaped_label_import_keeps_missing_predictions_as_zero(self):
        registry_root, pool_root, embedding_root, candidates, questions = self.fixtures()
        runtime = self.root / "native"
        common.write_bytes(runtime / "evaluate.py", (FROZEN / "frozen_evaluate.py").read_bytes())
        common.write_bytes(runtime / "answer_parser.py", (FROZEN / "frozen_answer_parser.py").read_bytes())
        index = []
        grading = common.rows(pool_root / "grading/train.jsonl")
        for cid, cp in candidates.items():
            for bench in ("gsm", "math"):
                subset = [r for r in grading if r["benchmark"] == bench]
                directory = self.root / "imports" / cid / bench
                common.write_rows(directory / "questions.jsonl", subset)
                config = {**common.SETTINGS, "eval_file_sha256": common.sha(directory / "questions.jsonl"),
                          "checkpoint_sha256": cp["checkpoint_sha256"], "train_manifest_sha256": cp["train_manifest_sha256"],
                          "train_seed": cp["continuation_seed"], "train_arm": {"A": "REPEAT", "B": "RANDOM", "C1": "C1"}[cp["method"]],
                          "eval_config_hash": "synthetic"}
                common.write(directory / "predictions.config.json", config)
                common.write_rows(directory / "predictions.jsonl", [{"benchmark": r["benchmark"], "problem_id": r["problem_id"],
                                  "correct": True, "raw_output": "Final answer: 1", "parsed_answer": "1", "parse_status": "ok",
                                  "checkpoint_sha256": cp["checkpoint_sha256"], "train_manifest_sha256": cp["train_manifest_sha256"],
                                  "model_revision": common.MODEL + "@" + common.REVISION, "parser_version": "p3-parser-v6",
                                  "eval_config_hash": "synthetic"} for r in subset[:-1]])
                index.append({"candidate": cid, "benchmark": bench, "split": "train",
                              "predictions": str(directory / "predictions.jsonl"), "config": str(directory / "predictions.config.json"),
                              "questions": {"path": str(directory / "questions.jsonl"), "sha256": common.sha(directory / "questions.jsonl")}})
        common.write(self.root / "index.json", index)
        output = self.root / "imported-labels"
        with patch.object(labeling, "load_registry", return_value={"candidates": candidates, "synthetic": True}), \
             patch.object(labeling.subprocess, "run", side_effect=AssertionError("imports must not generate")):
            metadata = labeling.label({"evaluator": {"path": str(runtime / "evaluate.py"), "parser_path": str(runtime / "answer_parser.py")}},
                                      registry_root, pool_root, "train", output, import_index=self.root / "index.json")
        self.assertEqual(sum(r["missing_rows"] for r in metadata["coverage"]), 36)
        values = common.rows(output / "p17_A_t42/outcomes.jsonl")
        self.assertEqual(len(values), 12)
        self.assertEqual(sum(r["correct"] == 0 for r in values), 2)
        common.bundle(output, "labels")


if __name__ == "__main__":
    unittest.main()

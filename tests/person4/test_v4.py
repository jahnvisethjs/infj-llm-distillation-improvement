"""V4 boundary policy, inheritance, stage identity, and actual handoff regressions."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from capstone_person4.config import artifact_root, load_config, stage_fingerprint
from capstone_person4.data import freeze_stratified
from capstone_person4.protection import canonical_question, dispositions, matching_rules
from capstone_person4.provenance import PipelineError, digest, read_json, sha, write_json

REPO = Path(__file__).resolve().parents[2]


def row(identity, problem="A fresh question", group=None, source="orca", length=0):
    return {"row_id": identity, "problem_group_id": group or "group-" + identity,
            "problem": problem, "completion": "Worked solution", "eligible": True,
            "dataset_id": source, "length_bin": length, "duplicate_cluster_id": digest(problem)}


class PolicyTests(unittest.TestCase):
    def run_policy(self, candidates, hits=(), references=(), reviews=()):
        class Matcher:
            def matches(self, record, position):
                return [{"eval_set": h["boundary"], "eval_id": h["counterpart_id"], "eval_text": h.get("text", "reference"),
                         "kind": h["kind"], "jaccard": 0.7, "containment": 0.7, "jaccard_numbers_masked": 0.7}
                        for h in hits if h["row_id"] == record["row_id"]]
        rules = SimpleNamespace(normalize=lambda s: " ".join(s.casefold().split()), fingerprint="frozen-rules")
        with patch("capstone_person4.protection.make_matcher", return_value=Matcher()):
            return dispositions(candidates, references, rules, reviews)

    def test_all_boundary_reasons_are_unioned_without_double_counting(self):
        candidate = row("a")
        references = [{**row("p"), "boundary": "parent", "identity": "p"},
                      {**row("e"), "boundary": "protected", "identity": "e"}]
        hits = [{"row_id": "a", "boundary": b, "counterpart_id": i, "kind": "EXACT"}
                for b, i in (("parent", "p"), ("protected", "e"))]
        result = self.run_policy([candidate], hits, references)
        self.assertEqual(len(result["exclusions"]), 1)
        self.assertEqual(len(result["exclusions"][0]["reasons"]), 2)
        self.assertFalse(result["quarantine"])

    def test_unreviewed_similarity_is_quarantined_and_unrelated_review_cannot_admit(self):
        reference = {**row("p"), "boundary": "reserved", "identity": "p"}
        hit = {"row_id": "a", "boundary": "reserved", "counterpart_id": "p", "kind": "SIMILAR"}
        result = self.run_policy([row("a")], [hit], [reference])
        self.assertEqual(len(result["quarantine"]), 1)
        evidence = result["flagged_pairs"][0]
        review = {**evidence, "decision": "distinct", "rationale": "Different mathematical constraint"}
        admitted = self.run_policy([row("a")], [hit], [reference], [review])
        self.assertEqual(len(admitted["eligible"]), 1)
        review["counterpart_text_sha256"] = "unrelated-review"
        rejected = self.run_policy([row("a")], [hit], [reference], [review])
        self.assertEqual(len(rejected["quarantine"]), 1)
        self.assertTrue(rejected["unused_review_keys"])

    def test_duplicate_cluster_uses_row_order_and_propagates_protection(self):
        candidates = [row("b", "SAME   Question"), row("a", "same question")]
        result = self.run_policy(candidates)
        self.assertEqual([r["row_id"] for r in result["eligible"]], ["a"])
        self.assertEqual(result["duplicate_clusters"][0]["representative"], "a")
        hit = {"row_id": "b", "boundary": "parent", "counterpart_id": "p", "kind": "EXACT"}
        result = self.run_policy(candidates, [hit], [{**row("p"), "boundary": "parent", "identity": "p"}])
        self.assertFalse(result["eligible"])
        self.assertEqual(len(result["exclusions"]) + len(result["quarantine"]), 2)

    def test_internal_near_match_quarantines_both_endpoints(self):
        candidates = [row("a", "First"), row("b", "Second")]
        hit = {"row_id": "a", "boundary": "internal", "counterpart_id": "b", "kind": "NUMBERS_CHANGED"}
        result = self.run_policy(candidates, [hit], [{**candidates[1], "boundary": "internal", "identity": "b"}])
        self.assertEqual({r["row_id"] for r in result["quarantine"]}, {"a", "b"})

    def test_original_group_boundary_and_quality_are_enforced(self):
        candidate = {**row("a"), "quality_flags": ["unresolved"]}
        reference = {**row("p", group=candidate["problem_group_id"]), "boundary": "parent", "identity": "p"}
        result = self.run_policy([candidate], references=[reference])
        self.assertEqual({r["reason"] for r in result["exclusions"][0]["reasons"]},
                         {"reserved_or_seen_group", "quality_or_reference"})

    def test_upstream_synthetic_metadata_is_preserved_and_does_not_mean_fixture(self):
        candidate = {**row("orca"), "synthetic": True}
        result = self.run_policy([candidate])
        self.assertEqual(len(result["eligible"]), 1)
        self.assertIs(result["eligible"][0]["synthetic"], True)

    def test_stratified_freeze_keeps_parent_proportions_and_never_silently_shrinks(self):
        parents = [row("p0"), row("p1", source="numina")]
        candidates = [row("a"), row("b", "Other"), row("c", "Third", source="numina"), row("d", "Fourth", source="numina")]
        selected, wanted = freeze_stratified(candidates, parents, 4, 42)
        again, _ = freeze_stratified(list(reversed(candidates)), parents, 4, 42)
        self.assertEqual(selected, again)
        self.assertEqual(wanted, {"numina|0": 2, "orca|0": 2})
        with self.assertRaisesRegex(PipelineError, "Insufficient eligible candidate strata"):
            freeze_stratified(candidates[:3], parents, 4, 42)

    def test_question_schema_preserves_original_text_and_keeps_gold_separate(self):
        from capstone_person4.person2_adapter import Person2Adapter
        source = {"problem_id": "q1", "benchmark": "gsm", "question": "  ORIGINAL $x+1$?  ",
                  "gold_answer": "SECRET", "gold_solution": "SECRET SOLUTION", "source": "upstream", "split": "test"}
        canonical = canonical_question(source)
        self.assertEqual(canonical["problem"], source["question"])
        adapter = Person2Adapter.__new__(Person2Adapter)
        adapter.contract = SimpleNamespace(render_prompt=lambda question: "Question: " + question)
        self.assertNotIn("SECRET", adapter.render_question(canonical))
        self.assertEqual(canonical["source"], "upstream")


class DependencyTests(unittest.TestCase):
    def test_scoring_smoke_spans_sources_and_lengths(self):
        from capstone_person4.reference_scoring import diagnostic_subset
        rows = [{**row(f"{source}-{i}", source=source, length=i), "supervised_tokens": i + 10, "total_tokens": i + 20}
                for source in ("orca", "numina") for i in range(10)]
        selected = diagnostic_subset(rows, 8)
        self.assertEqual(len(selected), 8)
        for source in ("orca", "numina"):
            subset = [r for r in selected if r["dataset_id"] == source]
            self.assertEqual(len(subset), 4)
            self.assertIn(0, [r["length_bin"] for r in subset])
            self.assertIn(9, [r["length_bin"] for r in subset])

    def test_later_evaluator_settings_leave_earlier_stages_valid(self):
        cfg = load_config(REPO / "configs/c0.example.json")
        future = deepcopy(cfg)
        future["evaluation"].update(evaluator_script="later.py", script_sha256="later", evaluation_contract="later.json")
        for stage in ("prepare", "score-reference", "select", "train"):
            self.assertEqual(stage_fingerprint(cfg, stage), stage_fingerprint(future, stage))
            self.assertEqual(artifact_root(cfg, stage), artifact_root(future, stage))
        self.assertNotEqual(stage_fingerprint(cfg, "evaluate"), stage_fingerprint(future, "evaluate"))

    def test_continuation_seed_does_not_change_preparation_or_scoring(self):
        cfg = load_config(REPO / "configs/c0.example.json")
        other = deepcopy(cfg)
        other["training"]["seed"] = 123
        for stage in ("prepare", "score-reference", "select"):
            self.assertEqual(stage_fingerprint(cfg, stage), stage_fingerprint(other, stage))
        self.assertNotEqual(stage_fingerprint(cfg, "train"), stage_fingerprint(other, "train"))

    def test_fingerprints_bind_content_hashes_not_input_locations(self):
        from capstone_person4.evaluation import method_definition
        from capstone_person4.reference_scoring import measurement_key, smoke_proof
        cfg = load_config(REPO / "configs/c0.example.json")
        moved = deepcopy(cfg)
        for section, key in (("parent", "adapter"), ("data", "release"), ("data", "candidate_manifest"), ("integration", "notebook")):
            moved[section][key] = "/another/mount/" + key
        moved["output_root"] = "/another/mount/results"
        prep = {"candidate_sha256": "c", "tokenizer_fingerprint": "t", "evidence": {"release": {"format_sha256": "f"}}}
        for stage in ("prepare", "score-reference", "select", "train"):
            self.assertEqual(stage_fingerprint(cfg, stage), stage_fingerprint(moved, stage))
        self.assertEqual(measurement_key(cfg, prep, {}), measurement_key(moved, prep, {}))
        self.assertEqual(method_definition(cfg), method_definition(moved))
        moved["parent"]["adapter_sha256"] = "0" * 64
        self.assertNotEqual(stage_fingerprint(cfg, "score-reference"), stage_fingerprint(moved, "score-reference"))
        self.assertIsNone(smoke_proof(cfg, "no-such-key"))

    def test_audit_target_defers_evaluator_but_requires_protection(self):
        from capstone_person4.audit import audit, CHECKS
        checks = {name: (lambda unused: {"verified": True}) for name in CHECKS}
        def missing(unused):
            raise PipelineError("Missing evaluator")
        checks["evaluation"] = missing
        cfg = load_config(REPO / "configs/c0.example.json")
        with patch("capstone_person4.audit.CHECKS", checks), patch("capstone_person4.audit.discover_inputs", return_value={}):
            self.assertTrue(audit(cfg, target="approach")["ready"])
            self.assertFalse(audit(cfg, target="evaluation")["ready"])
            checks["overlap"] = missing
            self.assertFalse(audit(cfg, target="inputs")["ready"])


class HandoffTests(unittest.TestCase):
    def test_serialization_changes_alone_do_not_prove_training_updates(self):
        from capstone_person4.continuation import verify_adapter_update
        from safetensors.torch import save_file
        import torch
        root = REPO / ".test_artifacts" / ("v4-adapter-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        name = "model.layer.lora_A.weight"
        save_file({name: torch.zeros(2, 2)}, root / "parent.safetensors", metadata={"run": "parent"})
        save_file({name: torch.zeros(2, 2)}, root / "metadata_only.safetensors", metadata={"run": "continued"})
        self.assertNotEqual(sha(root / "parent.safetensors"), sha(root / "metadata_only.safetensors"))
        with self.assertRaisesRegex(PipelineError, "did not change any"):
            verify_adapter_update(root / "parent.safetensors", root / "metadata_only.safetensors")
        save_file({name: torch.ones(2, 2)}, root / "updated.safetensors")
        self.assertEqual(verify_adapter_update(root / "parent.safetensors", root / "updated.safetensors"), [name])

    def test_only_independent_reconciled_full_arms_can_complete_approach(self):
        from capstone_person4.handoff import finalize
        root = REPO / ".test_artifacts" / ("v4-handoff-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        cfg = load_config(REPO / "configs/c0.example.json")
        cfg["parent"]["adapter"] = str(root / "parent")
        Path(cfg["parent"]["adapter"]).mkdir()
        parent_weights = Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors"
        parent_weights.write_bytes(b"fixture-parent")
        cfg["parent"]["adapter_sha256"] = sha(parent_weights)
        selections, runs = {}, {}
        for method in ("B", "C0"):
            run = root / method
            adapter = run / "final/adapter"
            adapter.mkdir(parents=True)
            checkpoint = adapter / "adapter_model.safetensors"
            checkpoint.write_bytes(("fixture-continued-" + method).encode())
            selected = [{**row(method), "supervised_tokens": 10}]
            selection = {"manifest_sha256": "selection-" + method, "target_tokens": 10, "tolerance": 0.005}
            selections[method] = selected, selection
            write_json(run / "run_config.json", {"seed": cfg["training"]["seed"], "arm": method, "mode": "continuation",
                       "optimizer_initialization": "fresh_from_verified_parent_weights", "parent_adapter_sha256": cfg["parent"]["adapter_sha256"]})
            write_json(run / "training_summary.json", {"complete": True, "processed_tokens": 10,
                       "seen_ids": [method], "seen_groups": ["group-" + method]})
            write_json(run / "token_reconciliation.json", {
                "valid_controlled_run": True, "training_smoke_only": False, "synthetic": False, "method": method,
                "stage_fingerprint": stage_fingerprint(cfg, "train"), "selection_sha256": digest(selection),
                "parent_unchanged": True, "parent_fingerprint": cfg["parent"]["adapter_sha256"], "actual_tokens": 10,
                "run_config_sha256": sha(run / "run_config.json"), "training_summary_sha256": sha(run / "training_summary.json"),
                "checkpoint": str(adapter.parent), "checkpoint_sha256": sha(checkpoint)})
            runs[method] = run
        with patch("capstone_person4.handoff.gate"), \
                patch("capstone_person4.handoff.load_prepared", return_value=([], {"candidate_sha256": "candidate"})), \
                patch("capstone_person4.handoff.load_scores", return_value=([], {"measurement_key": "scores"})), \
                patch("capstone_person4.handoff.load_selection", side_effect=lambda unused, method: selections[method]), \
                patch("capstone_person4.handoff.artifact_root", return_value=root / "outputs"):
            report = finalize(cfg, runs["B"], runs["C0"])
            self.assertTrue(report["approach_complete"])
            self.assertEqual(report["evaluation_status"], "pending")
            self.assertEqual(report["pairwise_token_difference"], 0)
            self.assertIsNone(report["external_B"])
            # B trained elsewhere: the C0-only handoff states the budget B must match.
            solo = finalize(cfg, None, runs["C0"])
            self.assertEqual(set(solo["arms"]), {"C0"})
            self.assertIsNone(solo["pairwise_token_difference"])
            self.assertIn("manifest.jsonl", solo["external_B"]["matched_budget_B_manifest"])
            # A copied smoke record cannot be relabeled as a complete experimental arm.
            path = runs["C0"] / "token_reconciliation.json"
            altered = read_json(path)
            altered["training_smoke_only"] = True
            path.write_text(__import__("json").dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(PipelineError, "not a valid reconciled full arm"):
                finalize(cfg, runs["B"], runs["C0"])
            altered["training_smoke_only"] = False
            altered["actual_tokens"] = 11
            path.write_text(__import__("json").dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(PipelineError, "not a valid reconciled full arm"):
                finalize(cfg, runs["B"], runs["C0"])


@unittest.skipUnless((REPO.parent / "data/metadata/release_manifest.json").exists(), "Optional actual V4 input integration")
class SuppliedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(REPO / "configs/c0.v4.seed17.json")

    def test_inherited_release_and_exact_parent_exposure_bind(self):
        from capstone_person4.bindings import verify_release, parent_exposure
        release = verify_release(self.cfg)
        rows, parent = parent_exposure(self.cfg)
        self.assertEqual(release["rows"], 10000)
        self.assertEqual(len(rows), 1000)
        self.assertEqual(parent["processed_tokens"], 318845)
        changed = deepcopy(self.cfg)
        changed["data"]["release_sha256"] = "0" * 64
        with self.assertRaisesRegex(PipelineError, "SHA256 mismatch"):
            verify_release(changed)

    def test_seed_selection_cannot_silently_reuse_seed17(self):
        from capstone_person4.bindings import resolve_parent
        other = load_config(REPO / "configs/c0.v4.seed123.json")
        self.assertNotEqual(other["parent"]["adapter_sha256"], self.cfg["parent"]["adapter_sha256"])
        changed = deepcopy(self.cfg)
        changed["parent"]["training_seed"] = 123
        with self.assertRaisesRegex(PipelineError, "differs"):
            resolve_parent(changed)

    def test_supplied_matcher_keeps_math_operators_and_finds_cross_id_repeats(self):
        from capstone_person4.protection import make_matcher
        rules = matching_rules(self.cfg)
        self.assertNotEqual(rules.normalize("Find 2 + 3"), rules.normalize("Find 2 - 3"))
        reference = {"identity": "parent-id", "problem": "What is $2+3$?", "boundary": "parent"}
        matcher = make_matcher([reference], rules)
        hits = matcher.matches({**row("different-id", "What is \\(2+3\\)?"), "source": "orca"}, 1)
        self.assertEqual(hits[0]["kind"], "EXACT")
        changed = deepcopy(self.cfg)
        changed["protection"]["matcher_sha256"] = "0" * 64
        with self.assertRaisesRegex(PipelineError, "SHA256 mismatch"):
            matching_rules(changed)

    def test_nested_fast_questions_are_counted_once(self):
        from capstone_person4.protection import check_inputs
        report = check_inputs(self.cfg)
        self.assertEqual(report["source_rows"], 3019)
        self.assertEqual(report["unique_questions"], 2819)

    def test_wrong_historical_label_count_blocks_parent_binding(self):
        from capstone_person4.bindings import parent_exposure
        from capstone_person4.provenance import read_rows
        def changed_labels(path):
            records = read_rows(path)
            if Path(path).name == "actual_labels.csv":
                records[0]["actual_supervised_tokens"] = str(int(records[0]["actual_supervised_tokens"]) + 1)
            return records
        with patch("capstone_person4.bindings.read_rows", side_effect=changed_labels):
            with self.assertRaisesRegex(PipelineError, "actual-label counts"):
                parent_exposure(self.cfg)

    def test_parent_serialization_policy_must_match_v4(self):
        from capstone_person4.bindings import parent_exposure
        from capstone_person4.provenance import read_json
        def changed_policy(path):
            metadata = read_json(path)
            if Path(path).name == "tokenization_policy.json":
                metadata["append_eos"] = False
            return metadata
        with patch("capstone_person4.bindings.read_json", side_effect=changed_policy):
            with self.assertRaisesRegex(PipelineError, "serialization/tokenization policy"):
                parent_exposure(self.cfg)

    def test_wrong_inherited_manifest_binding_is_rejected(self):
        from capstone_person4.bindings import verify_release
        from capstone_person4.provenance import read_json
        def changed_binding(path):
            metadata = read_json(path)
            if Path(path) == Path(self.cfg["data"]["release"]) / "metadata/release_manifest.json":
                metadata["original_manifest_sha256"] = "0" * 64
            return metadata
        with patch("capstone_person4.bindings.read_json", side_effect=changed_binding):
            with self.assertRaisesRegex(PipelineError, "SHA256 mismatch"):
                verify_release(self.cfg)


if __name__ == "__main__":
    unittest.main()

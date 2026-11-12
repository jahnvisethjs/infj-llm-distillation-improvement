"""E scientific controls, trainer boundary, additive handoff and scheduler wiring."""

from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import importlib.util
import io
import itertools
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import zipfile

from capstone_person4 import b_select, continuation
from capstone_person4.config import load_config, stage_fingerprint
from capstone_person4.provenance import PipelineError, digest, read_json, read_rows, sha, write_bytes, write_json, write_rows
from capstone_person4.selection import random_priorities
from capstone_person4_approachC1.generation_scoring import classify
from capstone_person4_approachE import selection as e, runtime, handoff

REPO = Path(__file__).resolve().parents[2]
HAS_SOLVER = importlib.util.find_spec("ortools") is not None


def row(i):
    return {"row_id": "SYNTHETIC-" + i, "problem_group_id": "SYNTHETIC-group-" + i,
            "problem": "SYNTHETIC question " + i, "completion": "SYNTHETIC correct reference", "final_answer": "1",
            "dataset_id": "orca", "length_bin": 0, "prompt_tokens": 2, "completion_tokens": 9,
            "eos_tokens": 1, "supervised_tokens": 10, "total_tokens": 12, "boundary_crossing_token": False,
            "synthetic": True}


def oracle(parent, rows, multiplier=1, *, seed=42, utilities=None, **settings):
    """Exhaustive tiny-fixture oracle; never used by production selection."""
    feasible = []
    for chosen in itertools.combinations(rows, len(parent)):
        try:
            e.validate_budget(parent, chosen, {"tolerance": settings.get("global_tolerance", "0.005"),
                                              "stratum_tolerance": settings.get("stratum_tolerance", "0.05")})
        except PipelineError:
            continue
        score = sum((utilities[r["row_id"]] if utilities else b_select.utility(r["row_id"], seed)) for r in chosen)
        tie = sum(b_select.tie_utility(r["row_id"], seed) for r in chosen)
        feasible.append(((-score, tie), chosen))
    if not feasible:
        return None, {"status": "INFEASIBLE"}
    objective, chosen = min(feasible, key=lambda pair: pair[0])
    return list(chosen), {"status": "OPTIMAL", "objective": -objective[0], "best_bound": -objective[0],
                          "relative_gap": 0, "wall_seconds": 0, "approximate_optimization": False}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.directory = REPO / ".test_artifacts" / ("E-SYNTHETIC-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.cfg = load_config(REPO / "configs/c0.v4.0.5b.seed17.json")
        self.cfg["output_root"] = str(self.directory / "outputs")
        self.cfg["allocation"]["expected_replay_tokens"] = 20
        parent = self.directory / "SYNTHETIC_PARENT"
        write_bytes(parent / "adapter_model.safetensors", b"SYNTHETIC_NOT_REAL_WEIGHTS")
        self.cfg["parent"]["adapter"] = str(parent)
        self.cfg["parent"]["adapter_sha256"] = sha(parent / "adapter_model.safetensors")
        self.rows = [row(i) for i in "abcdef"]
        self.parent = [row(i) for i in "pq"]
        self.outcomes = [{"row_id": r["row_id"], "mistake": i < 3, "status": "incorrect" if i < 3 else "correct",
                          "correct": i >= 3, "flags": [], "generated_tokens": 5} for i, r in enumerate(self.rows)]
        self.prep = {"candidate_sha256": digest(self.rows), "parent_replay_tokens": 20,
                     "tokenizer_fingerprint": "SYNTHETIC", "evidence": {"release": {"format_sha256": "SYNTHETIC"}}}
        self.coverage = {"generation_key": "SYNTHETIC-generation-key", "generated_tokens": 30}
        write_rows(e.generation.generation_root(self.cfg) / "outcomes.jsonl", self.outcomes)
        self.coverage["outcomes_sha256"] = sha(e.generation.generation_root(self.cfg) / "outcomes.jsonl")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)  # No fixture directories are deleted.
        self.stack.enter_context(patch.object(e, "gate"))
        self.stack.enter_context(patch.object(e, "load_prepared", side_effect=lambda cfg: (self.rows, self.prep)))
        self.stack.enter_context(patch.object(e.generation, "load_generation", side_effect=lambda cfg: (self.outcomes, self.coverage)))
        self.stack.enter_context(patch.object(e, "coverage_parent_rows", return_value=self.parent))
        self.control = {"manifest_sha256": digest(self.rows[:2]), "actual_tokens": 20}
        self.stack.enter_context(patch.object(e, "load_control", return_value=(self.rows[:2], self.control)))
        write_json(e.upstream_root(self.cfg, "select") / "selection/C1/batch_gate.json", {
            "bindings": {"candidate_sha256": self.prep["candidate_sha256"],
                         "generation_key": self.coverage["generation_key"],
                         "outcomes_sha256": self.coverage["outcomes_sha256"],
                         "c0_selection_sha256": digest(self.control), "generation_code_sha256": sha(e.generation.__file__)},
            "selection_sha256": digest(self.control),
            "diagnostics": {"proceed_to_training": True, "degeneracy_flags": {}}})

    def select(self):
        with patch.object(b_select, "solve", side_effect=oracle):
            return e.select(self.cfg)

    def fake_adapter(self):
        core = SimpleNamespace()

        def train(args):
            rows = read_rows(args.data)
            current = core.fingerprint(args)
            core.provenance_guard(args, rows, current)
            config = {**current, "mode": "continuation", "arm": args.arm, "epochs": args.epochs,
                      "seed": args.seed, "accumulation": args.accumulation, "learning_rate": args.lr,
                      "warmup_updates": args.warmup_updates, "microbatch_size": 1, "packing": False,
                      "parent_adapter_sha256": sha(Path(args.parent) / "adapter_model.safetensors"),
                      "optimizer_initialization": "fresh_from_verified_parent_weights"}
            write_json(args.out / "run_config.json", config)
            write_bytes(args.out / "final/adapter/adapter_model.safetensors", b"SYNTHETIC_CHANGED_WEIGHTS")
            write_json(args.out / "final/adapter/adapter_config.json", {"synthetic": True})
            write_json(args.out / "training_summary.json", {"complete": True, "processed_tokens": sum(r["supervised_tokens"] for r in rows),
                       "seen_ids": sorted(r["row_id"] for r in rows), "seen_groups": sorted(r["problem_group_id"] for r in rows),
                       "checkpoint": str(args.out / "final")})

        core.train = train
        return SimpleNamespace(core=core, tokenizer=lambda: object(), tokenizer_fingerprint="SYNTHETIC",
                               encode=lambda r, tok: {"counts": {"supervised_tokens": r["supervised_tokens"]}},
                               format={"prompt_template": "SYNTHETIC {problem}"})

    def train(self, smoke=False, cfg=None, attempt=None):
        cfg = cfg or self.cfg
        with patch.object(continuation, "gate"), \
                patch.object(continuation, "Person2Adapter", return_value=self.fake_adapter()), \
                patch.object(continuation, "load_prepared", return_value=(self.rows, self.prep)), \
                patch.object(continuation, "verify_adapter_update", return_value=["SYNTHETIC_lora"]):
            return runtime.train(cfg, attempt or ("esmoke01" if smoke else "epilot01"), smoke=smoke)


class SelectionTests(Fixture):
    def test_constant_vectors_reproduce_B_and_wrong_count_dominates_random(self):
        for value in (False, True):
            outcomes = [{"row_id": r["row_id"], "mistake": value} for r in self.rows]
            priorities, multiplier = e.priorities(self.rows, outcomes, 42, 2)
            chosen, _ = oracle(self.parent, self.rows, utilities={r["row_id"]: r["utility_integer"] for r in priorities})
            baseline, _ = oracle(self.parent, self.rows)
            self.assertEqual({r["row_id"] for r in chosen}, {r["row_id"] for r in baseline})
            actual = [r["row_id"] for r in e.order(priorities) if r["row_id"] in {r["row_id"] for r in chosen}]
            expected = [r["row_id"] for r in sorted(random_priorities(baseline, 42), key=lambda r: (r["stratum"], -r["utility"], r["tie_key"]))]
            self.assertEqual(actual, expected)
            self.assertGreater(multiplier, 2 * max(r["random_utility"] for r in priorities))
        report = self.select()
        self.assertEqual(report["mistakes_selected"], 2)
        self.assertNotIn("fit", read_rows(e.selection_root(self.cfg) / "manifest.jsonl")[0])

    def test_no_reference_cache_repeated_selection_and_seed_identity(self):
        before = e.source_hashes()
        with patch("capstone_person4.reference_scoring.load_scores", side_effect=AssertionError("E requested reference loss")):
            report = self.select()
            with patch.object(b_select, "solve", side_effect=AssertionError("Unnecessary second solve")):
                self.assertEqual(e.select(self.cfg), report)
        cfg = runtime.with_seed(self.cfg, 137)
        self.assertEqual(e.fingerprint(cfg), e.fingerprint(self.cfg))
        self.assertEqual(e.fingerprint(cfg, "train", seedless=True), e.fingerprint(self.cfg, "train", seedless=True))
        self.assertNotEqual(e.fingerprint(cfg, "train"), e.fingerprint(self.cfg, "train"))
        self.assertEqual(before, e.source_hashes())

    def test_wrong_parent_outcome_change_and_code_change_invalidate_selection(self):
        self.select()
        base = e.fingerprint(self.cfg)
        with patch.object(e, "code_hashes", return_value={"SYNTHETIC-code-change": "new"}):
            self.assertNotEqual(e.fingerprint(self.cfg), base)
            with self.assertRaises(OSError):
                e.load_selection(self.cfg)
        cfg = deepcopy(self.cfg)
        cfg["parent"]["adapter_sha256"] = "other-parent"
        with self.assertRaises((OSError, PipelineError)):
            e.load_selection(cfg)
        self.coverage["generation_key"] = "other-checker"
        self.assertNotEqual(e.fingerprint(self.cfg), base)
        self.coverage["outcomes_sha256"] = "changed"
        with self.assertRaisesRegex(PipelineError, "outcome content changed"):
            e.load_selection(self.cfg)

    def test_duplicates_partial_execution_errors_and_tampering_are_blocked(self):
        original = deepcopy(self.outcomes)
        for value in (original[:-1], original + [original[0]], [{**r, "status": "execution_error"} for r in original]):
            self.outcomes = value
            with self.assertRaises(PipelineError):
                e.inputs(self.cfg)
        self.outcomes = original
        self.select()
        (e.selection_root(self.cfg) / "manifest.jsonl").write_text('{}\n', encoding="utf-8")
        with self.assertRaises((PipelineError, KeyError)):
            e.load_selection(self.cfg)

    def test_insufficient_mistakes_fill_coverage_and_infeasible_budget_fails(self):
        for i, o in enumerate(self.outcomes):
            o.update(mistake=i == 0, correct=i != 0, status="incorrect" if i == 0 else "correct")
        report = self.select()
        self.assertEqual((report["rows"], report["mistakes_selected"], report["actual_tokens"]), (2, 1, 20))
        with self.assertRaisesRegex(PipelineError, "global supervised-token"):
            e.validate_budget(self.parent, [{**r, "supervised_tokens": 99} for r in self.rows[:2]], self.cfg["allocation"])
        with self.assertRaisesRegex(PipelineError, "source/length"):
            e.validate_budget(self.parent, self.rows[:1], self.cfg["allocation"])

    def test_frozen_classifier_preserves_truncation_and_ambiguity(self):
        for parse, correct, stop, expected in (("ok", False, "length", "incorrect"),
                                               ("no_answer", False, "length", "truncated"),
                                               ("ambiguous", False, "eos", "unresolved_equivalence")):
            status, _, flags = classify({"parse_status": parse, "correct": correct, "finish_reason": stop})
            self.assertEqual(status, expected)
            self.assertEqual("hit_length_limit" in flags, stop == "length")

    def test_C1_gate_must_bind_the_actual_grading_used_by_controls(self):
        e.controls(self.cfg, self.rows)
        self.coverage["generation_key"] = "SYNTHETIC-other-grading"
        with self.assertRaisesRegex(PipelineError, "C1 gate does not bind"):
            e.controls(self.cfg, self.rows)

    @unittest.skipUnless(HAS_SOLVER, "Pinned OR-Tools unavailable; exhaustive oracle tests still run")
    def test_real_shared_solver_agrees_with_oracle(self):
        priorities, _ = e.priorities(self.rows, self.outcomes, 42, len(self.parent))
        utilities = {r["row_id"]: r["utility_integer"] for r in priorities}
        for constant in (None, False, True):
            if constant is not None:
                values = [{"row_id": r["row_id"], "mistake": constant} for r in self.rows]
                priorities, _ = e.priorities(self.rows, values, 42, len(self.parent))
                utilities = {r["row_id"]: r["utility_integer"] for r in priorities}
            actual, report = b_select.solve(self.parent, self.rows, utilities=utilities)
            expected, _ = oracle(self.parent, self.rows, utilities=utilities)
            self.assertEqual(report["status"], "OPTIMAL")
            self.assertEqual({r["row_id"] for r in actual}, {r["row_id"] for r in expected})


class RuntimeTests(Fixture):
    def test_trainer_bindings_restore_even_after_failure(self):
        saved = {k: getattr(continuation, k) for k in ("load_selection", "artifact_root", "stage_fingerprint", "seed_agnostic_fingerprint")}
        with self.assertRaisesRegex(RuntimeError, "SYNTHETIC"):
            with runtime.trainer_bindings():
                self.assertIs(continuation.artifact_root, e.root)
                raise RuntimeError("SYNTHETIC failure")
        self.assertTrue(all(getattr(continuation, k) is v for k, v in saved.items()))

    def test_E_smoke_required_shared_trainer_is_reused_and_seed_isolation(self):
        self.select()
        with self.assertRaisesRegex(PipelineError, "E train --smoke"):
            self.train()
        parent_before = sha(Path(self.cfg["parent"]["adapter"]) / "adapter_model.safetensors")
        self.train(smoke=True)
        full = self.train()
        other = runtime.with_seed(self.cfg, 137)
        second = self.train(cfg=other)
        self.assertNotEqual(full["run_directory"], second["run_directory"])
        self.assertNotEqual(full["stage_fingerprint"], stage_fingerprint(self.cfg, "train"))
        self.assertEqual(full["actual_tokens"], 20)
        self.assertEqual(parent_before, sha(Path(self.cfg["parent"]["adapter"]) / "adapter_model.safetensors"))
        self.assertTrue(runtime.validate_run(self.cfg, "E", full["run_directory"])["report"]["valid_controlled_run"])
        with self.assertRaisesRegex(PipelineError, "not a matching"):
            runtime.validate_run(other, "E", full["run_directory"])

    def test_finalize_package_relative_checksums_and_tamper_rejection(self):
        self.select()
        self.train(smoke=True)
        full = self.train()
        original = runtime.validate_run

        def validate(cfg, method, directory, smoke=False):
            if method == "E":
                return original(cfg, method, directory, smoke)
            # Explicit synthetic control references; no real C0/C1 model claims.
            path = self.directory / ("SYNTHETIC-" + method)
            write_json(path / "token_reconciliation.json", {"synthetic": True})
            return {"directory": str(path), "report": {"checkpoint_sha256": "SYNTHETIC", "actual_tokens": 20}}

        with patch.object(handoff, "validate_run", side_effect=validate):
            result = handoff.finalize(self.cfg, seeds=[42])
        package = handoff.package([result["run_index"]], self.directory / "transfer")
        self.assertFalse(package["complete_twelve_run_matrix"])
        with zipfile.ZipFile(package["archive"]) as archive:
            for line in archive.read("SHA256SUMS.txt").decode().splitlines():
                expected, relative = line.split(None, 1)
                self.assertEqual(hashlib.sha256(archive.read(relative.strip())).hexdigest(), expected)
            index = next(csv.DictReader(io.StringIO(archive.read("e_run_index.csv").decode())))
            self.assertEqual(index["adapter_path"], "E/0.5B/parent17/seed42/final/adapter")
            self.assertIn(index["adapter_path"] + "/adapter_config.json", archive.namelist())
            self.assertIn("E/0.5B/parent17/seed42/final/run_config.json", archive.namelist())
        (Path(full["run_directory"]) / "final/adapter/adapter_model.safetensors").write_bytes(b"SYNTHETIC TAMPER")
        with self.assertRaises(PipelineError):
            handoff.package([result["run_index"]], self.directory / "tampered-transfer")
        self.assertFalse((self.directory / "tampered-transfer").exists())


GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash unavailable")
class SchedulerTests(unittest.TestCase):
    def test_six_jobs_and_preflight_failure_submits_nothing(self):
        directory = REPO / ".test_artifacts" / ("E-SCHEDULER-" + uuid.uuid4().hex)
        (directory / "bin").mkdir(parents=True)
        python = directory / "python.sh"
        python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n', encoding="utf-8", newline="\n")
        python.chmod(0o755)
        sbatch = directory / "bin/sbatch"
        sbatch.write_text('#!/bin/bash\nprintf "%s " "$@" >> "$E_TEST_LOG"\nprintf "\\n" >> "$E_TEST_LOG"\necho "$(( $(wc -l < "$E_TEST_LOG") + 100 ))"\n', encoding="utf-8", newline="\n")
        sbatch.chmod(0o755)
        log = directory / "jobs.log"
        env = {**os.environ, "C0_REPO": REPO.as_posix(), "C0_PYTHON": python.as_posix(),
               "PATH": (directory / "bin").as_posix() + os.pathsep + os.environ["PATH"], "E_TEST_LOG": log.as_posix()}
        args = [BASH, "scripts/sol/submit_e_pipeline.sh", "configs/c0.v4.0.5b.seed17.json", "SYNTHETIC-e"]
        for seeds in ("42 42", "bad", ""):
            result = subprocess.run(args, cwd=REPO, env={**env, "E_SEEDS": seeds}, capture_output=True, text=True)
            if seeds:  # empty means the documented default in shell parameter expansion
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(log.exists())
        calls = log.read_text().splitlines()
        self.assertEqual(len(calls), 6)
        self.assertIn("e_select.sbatch", calls[0])
        self.assertIn("afterok:101", calls[1])
        self.assertIn("--smoke", calls[1])
        for call, seed in zip(calls[2:5], (42, 137, 2026)):
            self.assertIn("afterok:102", call)
            self.assertIn(f"--training-seed {seed}", call)
        self.assertIn("afterok:103:104:105", calls[5])
        python.write_text('#!/bin/bash\nexit 2\n', encoding="utf-8", newline="\n")
        before = log.read_bytes()
        result = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

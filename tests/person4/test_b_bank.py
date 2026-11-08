"""C0 rerun on Method B's bank: explicit universe, B's bins, the vendored CP-SAT selector, seeds."""

from fractions import Fraction
import importlib.util
import io
import contextlib
from pathlib import Path
import sys
import unittest
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from capstone_person4 import b_select
from capstone_person4.audit import check_universe
from capstone_person4.config import load_config, stage_fingerprint
from capstone_person4.continuation import seed_agnostic_fingerprint, trainer_constants
from capstone_person4.data import explicit_universe, rebin
from capstone_person4.provenance import PipelineError, read_rows, sha, write_json, write_rows

B_ROOT = REPO.parent / "b_inputs/b_inputs_20261108T063441Z/files/scratch/sshind38/capstone_approach_b/outputs/B_method_B_document_v3"
HAS_SOLVER = importlib.util.find_spec("ortools") is not None


def bank_row(identity, source, tokens, bin_, group=None):
    return {"row_id": identity, "problem_group_id": group or "group-" + identity, "dataset_id": source,
            "problem": "Q " + identity, "completion": "A " + identity, "final_answer": "1",
            "prompt_tokens": 5, "completion_tokens": tokens - 1, "eos_tokens": 1, "supervised_tokens": tokens,
            "total_tokens": tokens + 5, "boundary_crossing_token": False, "length_bin": bin_}


class ExplicitUniverseTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("b-bank-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        # Cut points: orca (-inf,10],(10,20],(20,inf); numina (-inf,15],(15,inf)
        write_json(self.root / "bins.json", {"endpoint_rule": "right-closed; bisect_left; outer edges -inf and +inf",
                                             "source_inner_edges": {"orca": ["10", "20"], "numina": ["15"]}})
        self.bank = [bank_row("a", "orca", 10, 0), bank_row("b", "orca", 11, 1), bank_row("c", "orca", 25, 2),
                     bank_row("d", "numina", 15, 0), bank_row("e", "numina", 16, 1)]
        release = [{**r, "length_bin": 7} for r in self.bank] + [bank_row("z", "orca", 9, 0)]
        write_rows(self.root / "selection_bank.jsonl", release)
        write_rows(self.root / "universe.jsonl", self.bank)
        protection = self.root / "protection"
        write_rows(protection / "eligible.jsonl", [{"row_id": i} for i in ("a", "b", "c", "z")])
        write_rows(protection / "exclusions.jsonl", [{"row_id": "d", "reasons": [{"disposition": "exclude"}]}])
        write_rows(protection / "quarantine.jsonl", [{"row_id": "e", "reasons": [{"disposition": "quarantine"}]}])
        self.cfg = {"data": {"candidate_manifest": str(self.root / "selection_bank.jsonl"),
                             "universe_manifest": str(self.root / "universe.jsonl"), "universe_sha256": sha(self.root / "universe.jsonl"),
                             "length_bins": str(self.root / "bins.json"), "length_bins_sha256": sha(self.root / "bins.json"),
                             "candidate_count": 5}}

    def test_rebin_is_right_closed_and_source_specific(self):
        edges = {"orca": [Fraction(10), Fraction(20)], "numina": [Fraction(15)]}
        bins = [r["length_bin"] for r in rebin(self.bank, edges)]
        self.assertEqual(bins, [0, 1, 2, 0, 1])
        with self.assertRaisesRegex(PipelineError, "absent from the parent bins"):
            rebin([bank_row("q", "gsm", 3, 0)], edges)

    def test_universe_is_frozen_as_is_with_dispositions_disclosed(self):
        rows, edges, dispositions, counts = explicit_universe(self.cfg, self.root / "protection")
        self.assertEqual([r["row_id"] for r in rows], ["a", "b", "c", "d", "e"])
        self.assertEqual(counts, {"eligible": 3, "exclusions": 1, "quarantine": 1})
        self.assertEqual({d["row_id"]: d["disposition"] for d in dispositions}["e"], "quarantine")
        self.assertEqual(edges["orca"], [Fraction(10), Fraction(20)])

    def variant(self, name, rows=None, protection=None, **data):
        """A separate universe/protection copy per case; outputs are create-only."""
        root = self.root / name
        cfg = {"data": {**self.cfg["data"], **data}}
        if rows is not None:
            write_rows(root / "universe.jsonl", rows)
            cfg["data"].update(universe_manifest=str(root / "universe.jsonl"), universe_sha256=sha(root / "universe.jsonl"))
        scan = self.root / "protection"
        if protection is not None:
            scan = root / "protection"
            for kind in ("eligible", "exclusions", "quarantine"):
                write_rows(scan / (kind + ".jsonl"), protection.get(kind) or read_rows(self.root / "protection" / (kind + ".jsonl")))
        return cfg, scan

    def test_hard_stops(self):
        cases = {
            "count": (self.variant("count", candidate_count=4), "candidate_count declares"),
            "sha": (self.variant("sha", universe_sha256="0" * 64), "SHA256 mismatch"),
            "stored_bin": (self.variant("bin", rows=[{**self.bank[0], "length_bin": 1}] + self.bank[1:]), "does not reproduce"),
            "changed_field": (self.variant("field", rows=[{**self.bank[0], "completion": "edited"}] + self.bank[1:]), "beyond length_bin"),
            "unknown_row": (self.variant("unknown", rows=self.bank[1:] + [bank_row("y", "orca", 10, 0)]), "absent from the release"),
            "duplicate": (self.variant("dup", rows=self.bank[1:] + [self.bank[1]]), "Duplicate row"),
            "unscanned": (self.variant("unscanned", protection={"eligible": [{"row_id": "z"}]}), "did not cover"),
        }
        for name, ((cfg, scan), message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(PipelineError, message):
                explicit_universe(cfg, scan)


class SeedAndRecipeTests(unittest.TestCase):
    def test_training_seed_changes_only_the_train_root_and_smoke_key_is_shared(self):
        base = REPO / "configs/c0.v4.0.5b.seed17.json"
        seeds = [load_config(base, training_seed=s) for s in (42, 137, 2026)]
        self.assertEqual([c["training"]["seed"] for c in seeds], [42, 137, 2026])
        self.assertEqual(len({stage_fingerprint(c, "train") for c in seeds}), 3)
        for stage in ("prepare", "score-reference", "select"):
            self.assertEqual(len({stage_fingerprint(c, stage) for c in seeds}), 1)
        self.assertEqual(len({seed_agnostic_fingerprint(c) for c in seeds}), 1)
        self.assertEqual(load_config(base)["training"]["seed"], 42)
        with self.assertRaisesRegex(PipelineError, "training_seed"):
            load_config(base, training_seed=-1)

    def test_trainer_constants_are_read_from_source(self):
        source = ("BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)\n"
                  "torch.optim.AdamW(params, lr=lr, weight_decay=0.0)\n"
                  "torch.nn.utils.clip_grad_norm_(params, 1.0)\n")
        self.assertEqual(trainer_constants(source), {"quantization": "nf4", "double_quantization": "True",
                                                     "load_in_4bit": "True", "weight_decay": "0.0", "gradient_clip": "1.0"})
        self.assertIsNone(trainer_constants("")["quantization"])

    def test_universe_check_requires_the_pair_of_declarations(self):
        self.assertEqual(check_universe({"data": {}, "allocation": {}})["mode"], "frozen-sample")
        with self.assertRaisesRegex(PipelineError, "declared together"):
            check_universe({"data": {"universe_mode": "explicit"}, "allocation": {}})


@unittest.skipUnless((B_ROOT / "shared/mixed_candidate_bank.jsonl").exists(), "Method B bundle not present")
class RealBankTests(unittest.TestCase):
    def test_real_configs_pass_the_universe_check(self):
        for name in ("c0.v4.0.5b.seed17", "c0.v4.1.5b.seed123"):
            evidence = check_universe(load_config(REPO / "configs" / (name + ".json")))
            self.assertTrue(evidence["universe_sha256"].startswith("806dd78f"))
            self.assertTrue(evidence["b_selected_sha256"].startswith("378316a7"))

    @unittest.skipUnless(HAS_SOLVER, "OR-Tools 9.14.6206 not installed in this interpreter")
    def test_vendored_selector_reproduces_method_b_exactly(self):
        parent = read_rows(B_ROOT / "shared/parent_rebinned_reference.jsonl")
        bank = read_rows(B_ROOT / "shared/mixed_candidate_bank.jsonl")
        expected = set((B_ROOT / "mixed/budget_1x/1x/selected_ids.txt").read_text().split())
        for utilities in (None, {r["row_id"]: b_select.utility(r["row_id"]) for r in bank}):
            with contextlib.redirect_stdout(io.StringIO()):
                chosen, report = b_select.solve(parent, bank, 1, utilities=utilities)
            self.assertEqual(report["status"], "OPTIMAL")
            self.assertEqual(report["objective"], 912175166)
            self.assertEqual(report["actual_tokens"], 317506)
            self.assertEqual({r["row_id"] for r in chosen}, expected)
        with self.assertRaisesRegex(ValueError, "cover the whole common bank"):
            b_select.solve(parent, bank, 1, utilities={bank[0]["row_id"]: 1})


if __name__ == "__main__":
    unittest.main()

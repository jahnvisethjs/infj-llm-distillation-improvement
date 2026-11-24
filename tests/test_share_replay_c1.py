"""Synthetic CPU checks for the twelve-adapter file handoff, without Sol access."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_share_replay_c1", REPO / "scripts/share_replay_c1.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class ShareTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("share-replay-" + uuid.uuid4().hex)
        self.shared = self.root / "megha_shared"
        self.shared.mkdir(parents=True)
        self.destination = self.shared / "replay_c1_all12_SYNTHETIC"
        self.roots = {"0.5B": self.root / "root05", "1.5B": self.root / "root15"}
        self.runs = {}
        for size in self.roots:
            for parent in (17, 123):
                name = handoff.KNOWN.get((size, parent), "1234123412341234")
                for seed in handoff.SEEDS:
                    self.fixture_run(size, parent, seed, name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def fixture_run(self, size, parent, seed, name):
        branch = self.roots[size] / f"{size}_parent{parent}" / name
        run = branch / "runs" / f"seed{seed}_attempt01"
        train = branch / "training_inputs/full/train.jsonl"
        self.write(train, {"SYNTHETIC": "not a real training manifest"})
        config = {"model": "Qwen/Qwen2.5-" + size, "revision": "SYNTHETIC-revision",
                  "seed": seed, "arm": handoff.METHOD, "mode": "branch",
                  "parent": str(self.root / "parents" / f"{size}_seed{parent}" / "final"),
                  "parent_adapter_sha256": "SYNTHETIC-parent-hash", "manifest_sha256": handoff.sha(train)}
        self.write(run / "run_config.json", config)
        self.write(run / "final/run_config.json", config)
        self.write(run / "training_summary.json", {"complete": True, "processed_tokens": 318845})
        self.write(run / "training_log.jsonl", {"SYNTHETIC": "not a real loss record"})
        self.write(run / "final/adapter/adapter_config.json", {"SYNTHETIC": "not a real LoRA config"})
        weights = run / "final/adapter/adapter_model.safetensors"
        weights.write_bytes(f"SYNTHETIC-not-model-weights-{size}-{parent}-{seed}".encode())
        self.write(run / "completion.json", {"passed": True, "smoke_only": False, "reload_probe_matched": True,
            "method": handoff.METHOD, "seed": seed, "unique_examples": 1000, "identity": name + "0" * 48,
            "checkpoint": str(run / "final"), "adapter_sha256": handoff.sha(weights),
            "run_config_sha256": handoff.sha(run / "run_config.json"),
            "summary_sha256": handoff.sha(run / "training_summary.json"),
            "log_sha256": handoff.sha(run / "training_log.jsonl"), "parent_sha256": "SYNTHETIC-parent-hash",
            "actual_tokens": 318845, "portion_tokens": {"replay": 159422, "fresh": 159423}})
        self.write(run / "optimizer.pt", {"SYNTHETIC": "optimizer must not be exported"})
        self.write(branch / "smokes/seed42_smoke/final/adapter/adapter_model.safetensors", {"SYNTHETIC": "smoke"})
        self.runs[size, parent, seed] = run

    def snapshot(self):
        return {str(p): handoff.sha(p) for root in self.roots.values() for p in root.rglob("*") if p.is_file()}

    def test_all_twelve_adapters_metadata_and_checksums_are_complete(self):
        before = self.snapshot()
        ready = handoff.export(self.roots, self.shared, self.destination, grant_acl=False)
        self.assertTrue(ready["complete"])
        self.assertEqual(ready["adapters"], 12)
        self.assertEqual(self.snapshot(), before)
        with (self.destination / "run_index.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({(r["model_size"], r["parent_seed"], r["continuation_seed"]) for r in rows}), 12)
        for row in rows:
            adapter = Path(row["adapter_path"])
            self.assertEqual(adapter, self.destination / row["adapter_relative_path"])
            self.assertEqual(handoff.sha(adapter / "adapter_model.safetensors"), row["checkpoint_sha256"])
            self.assertTrue((adapter.parent / "run_config.json").is_file())
            self.assertTrue((adapter.parent.parent / "training_summary.json").is_file())
        for line in (self.destination / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(handoff.sha(self.destination / name), expected)
        self.assertFalse(list(self.destination.rglob("optimizer.pt")))
        self.assertFalse(list(self.destination.rglob("smokes")))

    def test_acl_grants_target_only_the_new_copy_and_the_correct_recipient(self):
        with patch.object(handoff.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as acl:
            handoff.export(self.roots, self.shared, self.destination)
        self.assertEqual(acl.call_count, 2)
        for call in acl.call_args_list:
            self.assertEqual(call.args[0][0], "setfacl")
            self.assertEqual(call.args[0][-1], str(self.destination))
            self.assertTrue(any("msures15" in value for value in call.args[0]))

    def test_corrupt_adapter_blocks_before_a_destination_is_created(self):
        run = self.runs["1.5B", 123, 2026]
        (run / "final/adapter/adapter_model.safetensors").write_bytes(b"SYNTHETIC corrupt weights")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            handoff.export(self.roots, self.shared, self.destination, grant_acl=False)
        self.assertFalse(self.destination.exists())

    def test_incomplete_training_blocks_before_copying_anything(self):
        path = self.runs["0.5B", 123, 137] / "completion.json"
        receipt = handoff.read(path)
        receipt["passed"] = False
        self.write(path, receipt)
        with self.assertRaisesRegex(ValueError, "Expected one completed"):
            handoff.export(self.roots, self.shared, self.destination, grant_acl=False)
        self.assertFalse(self.destination.exists())

    def test_multiple_parent123_identities_require_an_explicit_resolution(self):
        for seed in handoff.SEEDS:
            self.fixture_run("0.5B", 123, seed, "abcdabcdabcdabcd")
        with self.assertRaisesRegex(ValueError, "found 2"):
            handoff.export(self.roots, self.shared, self.destination, grant_acl=False)
        self.assertFalse(self.destination.exists())

    def test_existing_handoff_is_preserved(self):
        self.destination.mkdir()
        sentinel = self.destination / "existing.txt"
        sentinel.write_text("SYNTHETIC existing copy", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            handoff.export(self.roots, self.shared, self.destination, grant_acl=False)
        self.assertEqual(sentinel.read_text(), "SYNTHETIC existing copy")

    def test_a_wrong_model_or_parent_cannot_be_mislabeled_in_the_index(self):
        run = self.runs["1.5B", 17, 42]
        config = handoff.read(run / "run_config.json")
        config["parent"] = str(self.root / "parents/1.5B_seed123/final")
        self.write(run / "run_config.json", config)
        self.write(run / "final/run_config.json", config)
        receipt = handoff.read(run / "completion.json")
        receipt["run_config_sha256"] = handoff.sha(run / "run_config.json")
        self.write(run / "completion.json", receipt)
        with self.assertRaisesRegex(ValueError, "Model, parent"):
            handoff.export(self.roots, self.shared, self.destination, grant_acl=False)


if __name__ == "__main__":
    unittest.main()

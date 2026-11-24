"""Real native-source reconstruction; training/model loading are never executed."""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from capstone_routing import common, inputs, proofs

REPO = Path(__file__).resolve().parents[2]


class FunctionProofTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / ".test_artifacts" / ("routing-source-proof-" + uuid.uuid4().hex)
        notebook = common.read(REPO / "Person2_SOL_Seeds17_123_UpTo4000.ipynb")
        self.core = next("".join(c.get("source", [])).partition("\n")[2] for c in notebook["cells"]
                         if "".join(c.get("source", [])).startswith("%%writefile trainer_core.py\n"))
        self.core_path = self.root / "trainer_core.py"
        common.write_bytes(self.core_path, self.core.encode())

    def spec(self):
        source = proofs.reconstruct_c1(self.core)
        path = self.root / "C1_train.py"
        common.write_bytes(path, source.encode())
        return {"path": str(path), "sha256": proofs.C1_TRAIN_SHA, "representation": proofs.REPRESENTATION,
                "core": {"path": str(self.core_path), "sha256": proofs.CORE_SHA},
                "loader_function": "build_model", "save_function": "save_checkpoint",
                "format": "cumulative_lora_on_base", "parent_loaded_for_continuation": True}

    def test_exact_native_train_fingerprint(self):
        source = proofs.reconstruct_c1(self.core)
        self.assertEqual(proofs.text_sha(source), proofs.C1_TRAIN_SHA)
        tree = ast.parse(source)
        self.assertEqual([n.name for n in tree.body], ["train"])
        self.assertNotIn("def build_model", source)
        self.assertIn("adapter = Path(args.parent) / 'adapter'", source)

    def test_function_proof_verifies_separate_core_loader_and_saver(self):
        result = inputs.load_recipe(self.spec())
        self.assertEqual(result["format"], "cumulative_lora_on_base")
        self.assertEqual(result["core_sha256"], proofs.CORE_SHA)
        self.assertEqual(result["train_source_sha256"], proofs.C1_TRAIN_SHA)

    def test_changed_core_or_train_cannot_be_relabelled_as_a_proof(self):
        spec = self.spec()
        Path(spec["path"]).write_text(Path(spec["path"]).read_text() + "\n# changed\n")
        spec["sha256"] = common.sha(spec["path"])
        with self.assertRaises(common.RoutingError):
            inputs.load_recipe(spec)
        with self.assertRaises(common.RoutingError):
            proofs.reconstruct_c1(self.core.replace("adapter", "changed_adapter", 1))

    def test_export_writes_exact_bytes_without_executing_training(self):
        result = proofs.export_c1(self.core_path, self.root / "export")
        self.assertFalse(result["training_executed"])
        common.bundle(self.root / "export", "training_source_proof")
        self.assertEqual(common.sha(self.root / "export/C1_train.py"), proofs.C1_TRAIN_SHA)

    def test_function_source_is_bound_to_c1_run_metadata_and_wrapper(self):
        spec = self.spec()
        parent = self.root / "parent/final/adapter"
        cp = self.root / "run/final/adapter"
        common.write_bytes(parent / "adapter_model.safetensors", b"SYNTHETIC parent")
        common.write(parent.parent / "run_config.json", {"model": common.MODEL, "revision": common.REVISION,
                     "seed": 17, "manifest_sha256": "e" * 64})
        common.write_bytes(cp / "adapter_model.safetensors", b"SYNTHETIC child")
        common.write(cp / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": common.MODEL})
        train = {"model": common.MODEL, "revision": common.REVISION, "parent_adapter_sha256": common.sha(parent / "adapter_model.safetensors"),
                 "parent": str(parent.parent), "arm": "C1", "seed": 42, "epochs": 1, "manifest_sha256": "c" * 64,
                 "shared_universe_sha256": inputs.SCREENED_C1_BANK, "mode": "branch", "code_sha256": proofs.C1_TRAIN_SHA,
                 "wrapper_sha256": "50809b9e374fcc05a9b637c381396b46223ab85abcda3ca07cd4b6a5a4af882f"}
        common.write(cp.parent / "run_config.json", train)
        common.write(cp.parent.parent / "training_summary.json", {"complete": True, "processed_tokens": 318845})
        with patch.dict(inputs.PARENT_HASHES, {17: train["parent_adapter_sha256"]}):
            record = inputs.checkpoint_record({"checkpoint": str(cp), "training_source": spec})
            self.assertEqual(record["recipe"]["train_source_sha256"], train["code_sha256"])
            train["wrapper_sha256"] = "wrong"
            # Fixture-only corruption to check the recorded native-wrapper binding.
            (cp.parent / "run_config.json").write_text(json.dumps(train))
            with self.assertRaises(common.RoutingError):
                inputs.checkpoint_record({"checkpoint": str(cp), "training_source": spec})

    def test_register_adds_proofs_without_mutating_existing_pool(self):
        candidates = {common.identity(p, m, s): {"id": common.identity(p, m, s), "method": m,
                      "checkpoint": f"SYNTHETIC/{p}/{m}/{s}", "parent_checkpoint": f"SYNTHETIC/parent/{p}",
                      "checkpoint_sha256": common.digest([p, m, s]), "recipe": None}
                      for p in common.PARENTS for m in common.METHODS for s in common.SEEDS}
        original = {"candidates": candidates, "synthetic": False}
        upstream = self.root / "original"
        common.write(upstream / "registry.json", original)
        common.seal(upstream, "opportunity")
        pool = self.root / "pool"
        common.write(pool / "split.json", {"SYNTHETIC": True})
        common.seal(pool, "pool", [upstream])
        before = common.sha(upstream / "receipt.json"), common.sha(pool / "receipt.json")
        proof_root = self.root / "proof"
        proofs.export_c1(self.core_path, proof_root)
        def attach(spec):
            candidate = next(c for c in candidates.values() if c["checkpoint"] == spec["checkpoint"])
            return {**candidate, "recipe": {"source": spec["training_source"]}}
        with patch.object(inputs, "load_registry", return_value=original), patch.object(inputs, "checkpoint_record", side_effect=attach):
            result = proofs.register(upstream, self.core_path, self.core_path, proof_root, self.root / "registered")
        self.assertEqual(result["registered_checkpoints"], 18)
        self.assertEqual(before, (common.sha(upstream / "receipt.json"), common.sha(pool / "receipt.json")))
        common.bundle(pool, "pool")
        common.bundle(self.root / "registered", "registry")
        self.assertTrue(all(c["recipe"] is None for c in common.read(upstream / "registry.json")["candidates"].values()))


if __name__ == "__main__":
    unittest.main()

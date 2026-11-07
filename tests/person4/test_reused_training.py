"""Execute the reused update loop with a synthetic CPU model, never a real parent."""

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_pipeline import REPO, artifact_dir
from capstone_person4.person2_adapter import continuation_source, notebook_sources
from capstone_person4.provenance import read_json, write_json, write_rows


class ReusedTrainingTests(unittest.TestCase):
    def test_parent_weights_fresh_optimizer_final_partial_group_and_exact_tokens(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch unavailable")
        source = continuation_source(notebook_sources(REPO / "Person2_SOL_Seeds17_123_UpTo4000.ipynb")["trainer_core.py"])
        core = ModuleType("synthetic_person2_core")
        core.__file__ = str(REPO / "Person2_SOL_Seeds17_123_UpTo4000.ipynb")
        exec(compile(source, "<synthetic CPU reused engine>", "exec"), core.__dict__)
        root = artifact_dir()
        adapter = root / "SYNTHETIC_PARENT_ADAPTER"
        write_json(adapter / "adapter_config.json", {"synthetic": True})
        from capstone_person4.provenance import write_bytes
        write_bytes(adapter / "adapter_model.safetensors", b"SYNTHETIC FIXTURE, NOT MODEL WEIGHTS")
        rows = [{"row_id": f"synthetic-r{i}", "problem_group_id": f"synthetic-g{i}", "dataset_id": "orca",
                 "problem": "synthetic", "completion": "synthetic", "final_answer": "0", "synthetic": True,
                 "prompt_tokens": 1, "completion_tokens": 1, "supervised_tokens": 2, "total_tokens": 3} for i in range(3)]
        write_rows(root / "data.jsonl", rows)
        write_json(root / "template.json", {"template": "Question {problem}"})
        write_json(root / "policy.json", {"confirmed_with_person1": True, "add_special_tokens": False,
                   "append_eos": True, "max_total_tokens": 10, "provenance": "SYNTHETIC TEST ONLY"})
        encoded = [{"input_ids": [1, 2, 0], "labels": [-100, 2, 0],
                    "counts": {k: row[k] for k in core.COUNT_FIELDS}} for row in rows]
        core.provenance_guard = lambda *unused: None
        core.get_tokenizer = lambda: object()
        core.fingerprint = lambda unused: {"model": "SYNTHETIC", "revision": "SYNTHETIC"}
        core.reconcile = lambda *unused: ([{"match": True}] * 3, encoded)
        core.environment = lambda unused: {"packages": {}, "synthetic": True}
        core.probe_logits = lambda *unused: torch.tensor([1.0])

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_weight = torch.nn.Parameter(torch.full((3, 3), 0.1))
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.lora_weight[input_ids])

        loaded, saved = [], []
        def build(dtype, adapter=None, trainable=True):
            loaded.append(adapter)
            return Tiny()
        core.build_model = build
        def save(path, model, optimizer, scaler, tokenizer, state, config, enc, dtype):
            saved.append({"state": dict(state), "optimizer_states": len(optimizer.state),
                          "weights_changed": not torch.equal(model.lora_weight, torch.full((3, 3), 0.1))})
            write_json(Path(path) / "run_config.json", config)
        core.save_checkpoint = save
        transformers = ModuleType("transformers")
        transformers.set_seed = lambda seed: torch.manual_seed(seed)
        tensor, zeros = torch.tensor, torch.zeros
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return tensor(*args, **kwargs)
        def cpu_zeros(*args, **kwargs):
            kwargs["device"] = "cpu"
            return zeros(*args, **kwargs)
        original_adamw = torch.optim.AdamW
        fresh_states = []
        def new_adamw(*args, **kwargs):
            optimizer = original_adamw(*args, **kwargs)
            fresh_states.append(len(optimizer.state))
            return optimizer
        with patch.dict("sys.modules", {"transformers": transformers}), \
             patch.dict("os.environ", {"WORLD_SIZE": "1"}), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=False), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
             patch.object(torch, "tensor", side_effect=cpu_tensor), \
             patch.object(torch, "zeros", side_effect=cpu_zeros), \
             patch.object(torch, "autocast", side_effect=lambda *a, **kw: nullcontext()), \
             patch.object(torch.optim, "AdamW", side_effect=new_adamw), \
             patch.object(core.subprocess, "check_output", return_value="SYNTHETIC CPU TEST\n"):
            for method in ("B", "C0"):
                args = SimpleNamespace(data=root / "data.jsonl", template=root / "template.json", policy=root / "policy.json",
                                       mode="continuation", arm=method, epochs=1, accumulation=2, save_every=100, lr=0.01,
                                       seed=42, parent=adapter, resume=None, stop_after=None, clearance=None,
                                       out=root / method, compute_dtype="float32", warmup_updates=2)
                core.train(args)
        self.assertEqual(loaded, [adapter, adapter])
        self.assertEqual(fresh_states, [0, 0])
        for method in ("B", "C0"):
            summary = read_json(root / method / "training_summary.json")
            config = read_json(root / method / "run_config.json")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["processed_tokens"], 6)
            self.assertEqual(summary["optimizer_updates"], 2)
            self.assertEqual(summary["seen_ids"], sorted(r["row_id"] for r in rows))
            self.assertEqual(summary["parent_optimizer_updates"], 0)
            self.assertEqual(config["optimizer_initialization"], "fresh_from_verified_parent_weights")
            self.assertEqual(config["arm"], method)
        self.assertTrue(all(record["weights_changed"] for record in saved))


if __name__ == "__main__":
    unittest.main()

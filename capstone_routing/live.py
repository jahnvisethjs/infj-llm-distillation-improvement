"""Verified evaluator loader bridge, separate from the offline scoring harness."""

import ast
import importlib.util
from pathlib import Path
import time

from .common import (EVALUATOR_SHA, MODEL, PARSER_SHA, REVISION, SETTINGS, checked,
                     require, sha)
from .inputs import load_recipe, settings


def parser_module(spec):
    path = checked(spec["parser_path"], PARSER_SHA)
    module_spec = importlib.util.spec_from_file_location("routing_frozen_parser", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def evaluator_module(spec):
    """Load the pinned evaluator without replacing the process's answer_parser import."""
    path = checked(spec["path"], EVALUATOR_SHA)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Import)
                 and any(a.name == "answer_parser" for a in n.names))]
    namespace = {"__name__": "routing_frozen_evaluator", "__file__": str(path), "P": parser_module(spec),
                 "print": lambda *a, **kw: None}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def loader_code(spec):
    """Lift the exact native inference block; never rewrite the historical evaluator."""
    path = checked(spec["path"], EVALUATOR_SHA)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    starts = [i for i, n in enumerate(main.body) if isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names)]
    ends = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ctx_limit" for t in n.targets)]
    require(len(starts) == len(ends) == 1 and starts[0] < ends[0], "Native loader structure changed")
    nodes = main.body[starts[0]:ends[0] + 1]
    text = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    require("PeftModel.from_pretrained" in text and "merge_and_unload" in text, "Native loader semantics changed")
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path) + ":native_loader", "exec")


def verify_candidate(candidate):
    require(candidate.get("recipe") is not None, "Checkpoint loading representation is unverified; supply audited training_source evidence")
    require(load_recipe(candidate["recipe"]["source"]) == candidate["recipe"], "Checkpoint save representation changed")
    require(candidate["recipe"]["format"] == "cumulative_lora_on_base", "Unsupported checkpoint save format")
    checked(Path(candidate["checkpoint"]) / "adapter_model.safetensors", candidate["checkpoint_sha256"])
    checked(Path(candidate["parent_checkpoint"]) / "adapter_model.safetensors", candidate["parent_sha256"])
    for path, expected in candidate["evidence"].items():
        checked(path, expected)


class VerifiedGenerator:
    """One cached fully reconstructed checkpoint; no merging onto another adapter."""
    def __init__(self, evaluator):
        self.spec = evaluator
        self.native = evaluator_module(evaluator)
        self.code = loader_code(evaluator)
        self.cache_key = None
        self.state = None

    def sync(self):
        if self.state and self.state["device"] == "cuda":
            self.state["torch"].cuda.synchronize()

    def load(self, candidate):
        self.sync()
        started = time.perf_counter()
        verify_candidate(candidate)
        key = candidate["checkpoint_sha256"], candidate["model"], candidate["revision"]
        if self.cache_key == key:
            return time.perf_counter() - started, True
        # Drop the previous merged model completely. The native loader below
        # always creates a new original backbone before attaching this final LoRA.
        if self.state:
            torch = self.state["torch"]
            self.state.clear()
            self.state = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.cache_key = None
        native = self.native
        args = native["build_parser"]().parse_args([
            "--eval_file", "unused-live-input", "--out", "unused-live-output", "--run_id", "live",
            "--model", MODEL, "--revision", REVISION, "--adapter", candidate["checkpoint"],
            "--expect_train_manifest_sha256", candidate["train_manifest_sha256"],
            "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024", "--question_text", "p1_clean"])
        ckpt = native["resolve_checkpoint"](args)
        require(ckpt["checkpoint_sha256"] == candidate["checkpoint_sha256"], "Native loader resolved another checkpoint")
        config = {**native["generation_config"](args), "parser_version": native["P"].PARSER_VERSION,
                  "cut_at_next_question": native["P"].CUT_AT_NEXT_QUESTION,
                  "math_verify": native["version"]("math-verify"),
                  "evaluate_py_sha256": EVALUATOR_SHA, "answer_parser_sha256": PARSER_SHA}
        settings(config)
        state = {**native, "args": args, "config": config}
        exec(self.code, state)
        self.state, self.cache_key = state, key
        self.sync()
        return time.perf_counter() - started, False

    def generate(self, question, candidate):
        require(isinstance(question, str) and question.strip(), "Question must be nonempty text")
        loading, hit = self.load(candidate)
        state = self.state
        tokenizer, model, torch = state["tok"], state["model"], state["torch"]
        prompt = self.native["render_prompt"](question, "p1_clean")
        enc = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        require(enc.input_ids.shape[1] + 1024 <= state["ctx_limit"], "Question exceeds context limit; no silent truncation")
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=151643)
        self.sync()
        generation = time.perf_counter() - start
        raw_ids = generated[0, enc.input_ids.shape[1]:].tolist()
        ids = raw_ids[:raw_ids.index(151643)] if 151643 in raw_ids else raw_ids
        text = tokenizer.decode(ids, skip_special_tokens=True)
        return {"answer_text": text, "selected_checkpoint": candidate["id"],
                "checkpoint_path": candidate["checkpoint"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "lineage": {k: candidate[k] for k in ("model", "revision", "parent_seed", "parent_checkpoint", "parent_sha256",
                            "continuation_seed", "method", "train_manifest_sha256", "recipe")},
                "timing": {"loading_seconds": loading, "generation_seconds": generation, "checkpoint_cache_hit": hit},
                "generated_tokens": len(ids), "finish_reason": "eos" if 151643 in raw_ids else "length",
                "batch_size": 1, "decoding": "greedy", "prompt_sha256": SETTINGS["prompt_sha256"]}


class Answerer:
    def __init__(self, frozen_root, evaluator, authorize_gpu=False, experiment="primary_p17_t42"):
        require(authorize_gpu, "Live GPU answering requires explicit --authorize-gpu")
        from .router import FrozenRouter
        start = time.perf_counter()
        self.router = FrozenRouter(frozen_root, experiment)
        self.encoder_initialization_seconds = time.perf_counter() - start
        self.generator = VerifiedGenerator(evaluator)

    def answer(self, question):
        start = time.perf_counter()
        routed = self.router.route(question)
        routing_seconds = time.perf_counter() - start
        result = self.generator.generate(question, self.router.candidates[routed["selected_checkpoint"]])
        result["routing_scores"] = routed["routing_scores"]
        result["timing"].update(routing_seconds=routing_seconds,
                                end_to_end_seconds=time.perf_counter() - start,
                                encoder_initialization_seconds=self.encoder_initialization_seconds)
        return result


def answer(question, frozen_root, evaluator, authorize_gpu=False, experiment="primary_p17_t42"):
    """One-shot API; retain Answerer to reuse its single-checkpoint cache."""
    return Answerer(frozen_root, evaluator, authorize_gpu, experiment).answer(question)

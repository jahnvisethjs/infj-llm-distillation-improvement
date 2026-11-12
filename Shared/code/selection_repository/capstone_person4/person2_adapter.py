"""Thin interfaces over verified notebook definitions; no notebook execution."""

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from .provenance import PipelineError, digest, read_json, sha


def notebook_sources(path):
    result = {}
    for cell in read_json(path).get("cells", []):
        text = "".join(cell.get("source", []))
        first, _, source = text.partition("\n")
        if first in ("%%writefile trainer_core.py", "%%writefile person2.py"):
            name = first.split()[-1]
            if name in result:
                raise PipelineError(f"Ambiguous notebook definitions: {name}")
            ast.parse(source)
            result[name] = source
    if set(result) != {"trainer_core.py", "person2.py"}:
        raise PipelineError("Cannot locate the two expected importable Person 2 definition cells")
    return result


def continuation_source(source):
    """Change initialization/schedule only; retain Person 2's actual update/save loop.

    Fail closed if the known engine structure changes. The historical notebook is
    untouched, and this generated module is used only for the new continuation mode.
    """
    tree = ast.parse(source)
    train = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"), None)
    if train is None:
        raise PipelineError("Person 2 train definition missing")
    replacements = {"parent": 0, "lr": 0, "dtype": 0, "config": 0}

    class Adapt(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "args.parent":
                replacements["parent"] += 1
                return ast.parse("""
if args.parent:
    if args.mode != 'continuation' or args.epochs != 1:
        raise ValueError('Continuation requires a parent adapter and one pass')
    adapter = Path(args.parent)
    config['parent_adapter_sha256'] = sha(adapter / 'adapter_model.safetensors')
    config['optimizer_initialization'] = 'fresh_from_verified_parent_weights'
    config['warmup_updates'] = args.warmup_updates
    config['branch_optimizer_policy'] = 'fresh_optimizer_and_schedule'
    config['arm'] = args.arm
""").body[0]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == "lr":
                    replacements["lr"] += 1
                    node.value = ast.parse("args.lr * min(update / max(1, args.warmup_updates), 1.0)", mode="eval").body
                elif name == "dtype":
                    replacements["dtype"] += 1
                    node.value = ast.parse("getattr(torch, args.compute_dtype)", mode="eval").body
                elif name == "config":
                    replacements["config"] += 1
                    if not isinstance(node.value, ast.Call):
                        raise PipelineError("Unexpected Person 2 config construction")
                    for keyword in node.value.keywords:
                        if keyword.arg == "warmup_updates":
                            keyword.value = ast.parse("args.warmup_updates", mode="eval").body
            return self.generic_visit(node)

    Adapt().visit(train)
    if replacements != {"parent": 1, "lr": 1, "dtype": 1, "config": 1}:
        raise PipelineError(f"Person 2 source changed; review continuation adaptation: {replacements}")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def load_contract(root):
    root = Path(root)
    package_path = root / "capstone_person1"
    existing = sys.modules.get("capstone_person1")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if not origin or Path(origin).resolve() != (package_path / "__init__.py").resolve():
            raise PipelineError("A different capstone_person1 is already imported; use a fresh process with the configured release")
    else:
        spec = importlib.util.spec_from_file_location("capstone_person1", package_path / "__init__.py",
                                                     submodule_search_locations=[str(package_path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["capstone_person1"] = module
        spec.loader.exec_module(module)
    contract = importlib.import_module("capstone_person1.person1_data_contract")
    if Path(contract.__file__).resolve() != (package_path / "person1_data_contract.py").resolve():
        raise PipelineError("Imported encoder differs from configured immutable release")
    return contract


class Person2Adapter:
    def __init__(self, cfg, continuation=False):
        self.cfg = cfg
        notebook = Path(cfg["integration"]["notebook"])
        sources = notebook_sources(notebook)
        core_source = continuation_source(sources["trainer_core.py"]) if continuation else sources["trainer_core.py"]
        core = ModuleType("person4_reused_core")
        core.__file__ = str(notebook)
        exec(compile(core_source, str(notebook) + "::trainer_core", "exec"), core.__dict__)
        self.contract = load_contract(cfg["data"].get("inherited_encoder") or cfg["data"]["contract_root"])
        # Replace only the wrapper's import statement with the selected core object.
        wrapper_tree = ast.parse(sources["person2.py"])
        wrapper_tree.body = [node for node in wrapper_tree.body if not
                             (isinstance(node, ast.Import) and any(a.name == "trainer_core" for a in node.names))]
        wrapper = ModuleType("person4_reused_wrapper")
        wrapper.__file__ = str(notebook)
        wrapper.core = core
        exec(compile(wrapper_tree, str(notebook) + "::person2", "exec"), wrapper.__dict__)
        profile = cfg["active_profile"]
        if profile not in wrapper.MODELS or tuple(wrapper.MODELS[profile]) != (cfg["parent"]["model"], cfg["parent"]["revision"]):
            raise PipelineError("Configured model profile differs from Person 2's pinned model definitions")
        wrapper.select_model(profile)
        self.core, self.wrapper = core, wrapper
        self.format = read_json(Path(cfg["data"]["release"]) / "configs/formatting_contract.json")
        if self.contract.PROMPT_TEMPLATE != self.format["prompt_template"]:
            raise PipelineError("Encoder prompt does not match the frozen formatting contract")
        cap = cfg["data"]["max_total_tokens"]
        if type(cap) is not int or not 2 <= cap <= self.format["max_total_tokens"]:
            raise PipelineError("Explicit context cap must fit the verified formatting contract")

    def package_tokenizer(self, model):
        """Checksummed tokenizer assets saved beside the requested parent; no Hub access."""
        from .bindings import contained
        package = Path(self.cfg["inputs"]["parent_package"])
        seed = self.cfg["parent"]["training_seed"]
        entry = next((r for r in read_json(self.cfg["inputs"]["parent_index"]) if r["model"] == model and r["seed"] == seed), None)
        if entry is None:
            raise PipelineError(f"Parent package lacks a {model} seed-{seed} tokenizer entry")
        assets = contained(package, entry["adapter_directory"]).parent / "tokenizer"
        checksums = read_json(package / "SHA256SUMS.json")
        listed = {k: v for k, v in checksums.items() if k.startswith(assets.relative_to(package).as_posix() + "/")}
        if not {"tokenizer.json", "tokenizer_config.json"} <= {Path(k).name for k in listed}:
            raise PipelineError("Parent package tokenizer assets are not listed in SHA256SUMS.json")
        for relative, expected in listed.items():
            if sha(package / relative) != expected:
                raise PipelineError(f"Parent package tokenizer asset changed: {relative}")
        return assets

    def local_tokenizer(self, assets):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(assets), local_files_only=True, use_fast=True)
        tok.pad_token = tok.eos_token
        self.contract.verify_tokenizer(tok)
        return tok

    def tokenizer(self):
        assets = self.cfg["data"].get("tokenizer_assets")
        if not assets and self.cfg.get("protection"):
            assets = self.package_tokenizer(self.cfg["parent"]["model"])
        tok = self.local_tokenizer(assets) if assets else self.core.get_tokenizer()
        self.contract.verify_tokenizer(tok)
        # The reused trainer must use this exact tokenizer, including offline assets.
        self.core.get_tokenizer = lambda: tok
        if self.cfg.get("protection") and self.cfg["active_profile"] != "0.5B":
            # Person 2's 1.5B guard compares IDs/labels against the pinned 0.5B tokenizer.
            # Bind that reference to the checksummed package copy so compute nodes never
            # need Hub access; the guard itself is unchanged.
            reference = self.local_tokenizer(self.package_tokenizer(self.wrapper.MODELS["0.5B"][0]))
            self.wrapper.model_tokenizer = lambda model_id, revision: reference
        # Fingerprint actual serialization, vocabulary, special tokens and runtime.
        self.tokenizer_fingerprint = digest({"backend": tok.backend_tokenizer.to_str(),
                                            "special_tokens": tok.special_tokens_map,
                                            "revision": self.cfg["parent"]["revision"],
                                            "template": self.format["prompt_template"]})
        return tok

    def encode(self, row, tokenizer):
        encoded = self.wrapper.encode(row, tokenizer, self.format["prompt_template"], self.cfg["data"]["max_total_tokens"])
        encoded["attention_mask"] = [1] * len(encoded["input_ids"])
        if sum(v != -100 for v in encoded["labels"][1:]) != row["supervised_tokens"]:
            raise PipelineError(f"Effective shifted supervised count differs: {row['row_id']}")
        return encoded

    def load_parent(self, trainable=False):
        import torch
        dtype = getattr(torch, self.cfg["scoring"]["dtype"])
        adapter = Path(self.cfg["parent"]["adapter"])
        if sha(adapter / "adapter_model.safetensors") != self.cfg["parent"]["adapter_sha256"]:
            raise PipelineError("Parent adapter changed immediately before loading")
        model = self.core.build_model(dtype, adapter=adapter, trainable=trainable)
        active = model.active_adapters
        if callable(active):
            active = active()
        if not active or not model.peft_config:
            raise PipelineError("Trained parent adapter is not active")
        if not trainable:
            model.eval()
        return model

    def render_question(self, row):
        text = row.get("question", row.get("problem"))
        if not isinstance(text, str) or not text.strip():
            raise PipelineError("Question prompt requires nonempty original text")
        return self.contract.render_prompt(text)

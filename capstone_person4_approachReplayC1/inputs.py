"""Read-only import of the exact native screened C1 experiment and controls."""

import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

from . import VERSION
from .common import (BANK_SHA, EVALUATOR_SHA, EVAL_PACKAGES, MANIFEST_SHA, PARSER_SHA,
                     PARENT_SHA, RECIPE, RUNNER_SHA, SEEDS, TARGET,
                     PilotError, checked, code_hashes, digest, read, sha, source_sha, unique)


def native_context(cfg):
    # Do not import capstone_person4 before this: the frozen runner installs and
    # checks its own pinned selection-repository/runtime imports.
    path = checked(cfg["native_runner"], RUNNER_SHA)
    spec = importlib.util.spec_from_file_location("_replay_c1_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(work=cfg["work"], b_root=cfg["b_root"], model="0.5B", parent_seed=17)
    context = module.load_context(args)
    if module.artifact(context).resolve() != Path(cfg["c1_profile"]):
        raise PilotError("Native profile identity changed; restore the pinned C1 package and inputs")
    if context.entry["expected_adapter_sha256"] != PARENT_SHA:
        raise PilotError("Wrong original R1000 parent")
    checked(context.root / "data/train_n1000.jsonl", MANIFEST_SHA)
    checked(context.bank_path, BANK_SHA)
    module.verify_proof(context)
    return module, context


def recipe(config):
    if any(config.get(k) != v for k, v in RECIPE.items()) or config.get("dtype") != "torch.bfloat16":
        raise PilotError("Control training recipe differs from the recorded bf16 pilot recipe")
    if config.get("branch_optimizer_policy") not in (
            "fresh_optimizer_reset_10_update_warmup", "fresh_optimizer_and_scheduler", "fresh_optimizer_and_schedule"):
        raise PilotError("Control does not declare a fresh optimizer and warmup")


def token_policy(value):
    expected = {"add_special_tokens": False, "append_eos": True, "max_total_tokens": 2048}
    if any(value.get(k) != v for k, v in expected.items()):
        raise PilotError("Tokenization policy must use existing completion/EOS labels and 2048-token cap")
    return expected


def find_policy(cfg, expected, run_root, context):
    explicit = cfg.get("a_policy")
    candidates = [Path(explicit)] if explicit else []
    if not explicit:
        for root in (run_root, context.root / "data", context.runtime / context.entry["records_relative_path"]):
            candidates.extend(sorted(root.rglob("*policy*.json")))
    matches = [p for p in candidates if p.is_file() and sha(p) == expected]
    if not matches:
        raise PilotError("Cannot verify A's token policy. Set a_policy to the original policy JSON whose hash is " + expected)
    return checked(matches[0], expected)


def a_control(cfg, context, seed):
    path = Path(cfg["a_eval_root"]) / f"A_0.5B_p17_s{seed}_gsm.config.json"
    evaluation = read(path)
    checkpoint = Path(evaluation["checkpoint_id"])
    checked(checkpoint / "adapter_model.safetensors", evaluation.get("checkpoint_sha256"))
    directory = checkpoint.parents[1]
    config, summary = read(directory / "run_config.json"), read(directory / "training_summary.json")
    recipe(config)
    if (config.get("model") != context.entry["model"] or config.get("revision") != context.entry["revision"]
            or config.get("parent_adapter_sha256") != PARENT_SHA or config.get("seed") != seed
            or config.get("manifest_sha256") != MANIFEST_SHA or config.get("arm") != "REPEAT"
            or config.get("rows") != 1000 or evaluation.get("train_manifest_sha256") != MANIFEST_SHA
            or summary.get("complete") is not True or summary.get("processed_tokens") != TARGET
            or len(summary.get("seen_ids", [])) != 1000
            or set(summary["seen_ids"]) != {r["row_id"] for r in context.parent}
            or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in context.parent}):
        raise PilotError(f"A seed{seed}: checkpoint, recipe, replay exposure or token budget differs")
    native_template = context.condition / "training_inputs/prompt_template.json"
    if config.get("template_sha256") != sha(native_template):
        raise PilotError("A and native C1 prompt templates differ")
    policy = find_policy(cfg, config.get("policy_sha256"), directory, context)
    native_policy = context.condition / "training_inputs/tokenization_policy.json"
    if token_policy(read(policy)) != token_policy(read(native_policy)):
        raise PilotError("A/C1 effective tokenization policies differ")
    return {"method": "A", "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": evaluation["checkpoint_sha256"], "manifest_sha256": MANIFEST_SHA,
            "run_directory": str(directory), "run_config_sha256": sha(directory / "run_config.json"),
            "summary_sha256": sha(directory / "training_summary.json"), "policy_sha256": sha(policy)}


def validate_population(parent, bank):
    unique(parent)
    unique(bank)
    if len(parent) != 1000 or len(bank) != 4955:
        raise PilotError("Expected 1000 original parent examples and 4955 fresh candidates")
    # `synthetic` is genuine upstream source metadata (including Numina rows),
    # not a fixture marker. native_context's exact input hashes exclude fixtures.
    if sum(r["supervised_tokens"] for r in parent) != TARGET:
        raise PilotError("Parent's one-pass supervised tokens changed")
    for key in ("row_id", "problem_group_id"):
        if {r[key] for r in parent} & {r[key] for r in bank}:
            raise PilotError("Fresh candidates overlap original parent " + key)


def load(cfg):
    native, context = native_context(cfg)
    validate_population(context.parent, context.bank)
    scores, score_report = native.load_scores(context)
    outcomes, generation = native.load_generation(context)
    _, c1_selection = native.load_selection(context, "C1")
    if set(r["row_id"] for r in outcomes) != {r["row_id"] for r in context.bank} or len(outcomes) != len(context.bank):
        raise PilotError("C1 outcome coverage differs from the complete screened bank")
    for outcome in outcomes:
        if (outcome.get("status") == "execution_error" or type(outcome.get("mistake")) is not bool
                or outcome["mistake"] != (outcome.get("status") == "incorrect")):
            raise PilotError("Unresolved or invalid C1 correctness outcome")
    diagnostics = c1_selection["C1_diagnostics"]
    flags = diagnostics.get("degeneracy_flags")
    if (not isinstance(flags, dict) or len(flags) != 3 or any(flags.values())
            or not 0.05 <= diagnostics.get("mistake_share", -1) <= 0.95
            or not 0 <= diagnostics.get("C0_C1_jaccard", 1) < 0.95):
        raise PilotError("Native C1 diagnostic gate did not pass")
    controls = []
    for seed in SEEDS:
        directory = native.artifact(context, "runs", "C1", f"seed{seed}_attempt01")
        receipt = native.validate_completion(context, directory, "C1", seed)
        config = read(directory / "run_config.json")
        recipe(config)
        controls.extend([a_control(cfg, context, seed), {
            "method": "C1", "seed": seed, "checkpoint": str(Path(receipt["checkpoint"]) / "adapter"),
            "checkpoint_sha256": receipt["adapter_sha256"], "manifest_sha256": config["manifest_sha256"],
            "completion_sha256": sha(directory / "completion.json"), "actual_tokens": receipt["actual_tokens"]}])
    policy_path = context.condition / "training_inputs/tokenization_policy.json"
    token_policy(read(policy_path))
    binding = {"version": VERSION, "bank_sha256": BANK_SHA, "parent_sha256": PARENT_SHA,
               "parent_manifest_sha256": MANIFEST_SHA, "native_identity": context.identity,
               "native_dependencies": context.dependencies, "scores": score_report,
               "generation": generation, "native_C1_selection": c1_selection,
               "native_effective_trainer_sha256": source_sha(context.train_source),
               "policy_sha256": sha(policy_path), "controls": controls, "code": code_hashes(),
               "selection_seed": 42, "half_tolerance": "0.005", "stratum_tolerance": "0.05"}
    identity = digest(binding)
    root = Path(cfg["output_root"]) / "0.5B_parent17" / identity[:16]
    return SimpleNamespace(cfg=cfg, native=native, context=context, parent=context.parent,
                           rebinned=context.rebinned, bank=context.bank, scores=scores, outcomes=outcomes,
                           binding=binding, identity=identity, root=root, controls=controls)


def check_environment(cfg):
    import sys
    expected = Path(cfg["training_python"])
    if Path(sys.executable).absolute() != expected.absolute() and Path(sys.prefix) != expected.parent.parent:
        raise PilotError("Use the configured native training Python: " + str(expected))
    packages = {"torch": "2.6.0+cu124", "transformers": "4.57.1", "peft": "0.17.1",
                "accelerate": "1.10.1", "bitsandbytes": "0.48.1", "safetensors": "0.8.0", "numpy": "2.5.4"}
    for package, expected_version in packages.items():
        if importlib.metadata.version(package) != expected_version:
            raise PilotError(f"Training environment differs: {package} must be {expected_version}")
    subprocess.run([cfg["solver_python"], "-B", "-c",
                    "import ortools; assert ortools.__version__ == '9.14.6206'; from ortools.sat.python import cp_model"],
                   check=True, capture_output=True, text=True)


def preflight(cfg):
    check_environment(cfg)
    bundle = load(cfg)
    from .evaluation import native_c1_result, question_file
    for package, version in EVAL_PACKAGES.items():
        if importlib.metadata.version(package) != version:
            raise PilotError("Frozen evaluator environment differs: " + package)
    checked(bundle.context.runtime / "vendor/evaluation/evaluate.py", EVALUATOR_SHA)
    checked(bundle.context.runtime / "vendor/evaluation/answer_parser.py", PARSER_SHA)
    for benchmark in ("dev_gsm", "dev_math"):
        _, ids = question_file(bundle, benchmark)
        for control in (c for c in bundle.controls if c["method"] == "C1"):
            native_c1_result(bundle, control, benchmark, ids)
    return {"passed": True, "read_only": True, "identity": bundle.identity,
            "output_directory": str(bundle.root), "bank_rows": len(bundle.bank), "parent_rows": len(bundle.parent),
            "target_tokens": TARGET, "half_integer_bounds": [158626, 160219],
            "verified_controls": bundle.controls, "training_started": False,
            "note": "A predictions need common-runtime evaluation; no accuracy-based gate or GPU execution performed"}

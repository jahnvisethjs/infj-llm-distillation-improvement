"""Reconstruct the native C1 train-function proof without importing its trainer."""

import argparse
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

from .common import (METHODS, PARENTS, SEEDS, RoutingError, bundle, checked, destination, identity,
                     read, require, seal, sha, write, write_bytes)

CORE_SHA = "638b4befeab797f032f533b7ec605d3336f97142dec51782bfeb2d030a5ba498"
B_TRAIN_SHA = "ccb268042778615e10b256eb64f2f72eea7a9ff25a25ca8ed5834751be94f112"
C1_TRAIN_SHA = "3b908d61c27c600469984c05669f0ab9d851772a62f6dd3b5ed32239808ee6e8"
REPRESENTATION = "screened_c1_train_function_v1"


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reconstruct_c1(core):
    """Exact pure-string transformation from the inspected native continuation.py.

    The resulting function is data. Neither it nor any training entrypoint is
    executed here. Loader/save definitions remain in the pinned original core.
    """
    require(text_sha(core) == CORE_SHA, "C1 reconstruction needs the exact native Person 2 core")
    functions = [n for n in ast.parse(core).body if isinstance(n, ast.FunctionDef) and n.name == "train"]
    require(len(functions) == 1, "Expected one original train function")
    code = ast.get_source_segment(core, functions[0])

    def replace_once(old, new):
        nonlocal code
        require(code.count(old) == 1, "Native trainer transformation changed: " + old[:80])
        code = code.replace(old, new, 1)

    replace_once("experiment_status='preliminary' if args.mode == 'preliminary' else args.mode,",
                 "experiment_status='smoke_only' if args.smoke else 'preliminary_document_aligned_B',")
    replace_once("evaluation_readiness='pending' if args.mode == 'preliminary' else 'see_mode_and_clearance',",
                 "evaluation_readiness='smoke_not_evaluable' if args.smoke else 'see_frozen_protocol_and_parent_disclosure',")
    replace_once("mode=args.mode, arm=args.arm if args.mode == 'branch' else None,",
                 "mode='debug' if args.smoke else args.mode, arm=args.arm,")
    replace_once("branch_optimizer_policy='restore_parent_no_new_warmup' if args.mode == 'branch' else None,",
                 "branch_optimizer_policy='fresh_optimizer_and_scheduler',")
    replace_once("    restore = None\n", "    config.update(args.continuation_metadata)\n    restore = None\n")
    start = code.index("    if args.parent:\n")
    end = code.index("    if args.resume:\n", start)
    replacement = """    if not args.parent:
        raise ValueError('Continuation must start from the verified trained parent adapter')
    pcfg = read_json(Path(args.parent) / 'run_config.json')
    for k in ('model', 'revision'):
        if pcfg.get(k) != config[k]:
            raise ValueError(f'Parent configuration differs: {k}')
    parent_state = args.parent_metadata
    config['branch_budget'] = parent_budget_check(args, rows, encoded, pcfg, parent_state)
    config['parent_adapter_sha256'] = sha(Path(args.parent) / 'adapter' / 'adapter_model.safetensors')
    adapter = Path(args.parent) / 'adapter'
"""
    code = code[:start] + replacement + code[end:]
    replace_once("""    if parent_state:
        optimizer.load_state_dict(parent_state['optimizer']); scaler.load_state_dict(parent_state['scaler'])
        state['parent_optimizer_updates'] = parent_state['state']['optimizer_updates']
        # Same branch seed + fresh deterministic data order; parent optimizer restored for every arm.
""", "    # Parent weights are retained; parent optimizer/scaler/RNG state is never restored.\n")
    replace_once("if args.mode == 'debug' else None", "if (args.mode == 'debug' or args.smoke) else None")
    replace_once("lr = args.lr if args.mode == 'branch' else args.lr * min(update / 10.0, 1.0)",
                 "lr = args.lr * min(update / 10.0, 1.0)")
    require(text_sha(code) == B_TRAIN_SHA, "Reconstructed B train function differs from the native fingerprint")
    replace_once("'preliminary_document_aligned_B'", repr("screened_training_only_C1"))
    require(text_sha(code) == C1_TRAIN_SHA, "Reconstructed C1 train function differs from the recorded fingerprint")
    return code


def export_c1(core_path, output):
    core_path = checked(core_path, CORE_SHA).resolve()
    root = destination(output, [core_path])
    source = reconstruct_c1(core_path.read_text(encoding="utf-8"))
    source_path = root / "C1_train.py"
    # No added trailing newline: the recorded fingerprint covers these exact bytes.
    write_bytes(source_path, source.encode())
    spec = {"path": str(source_path), "sha256": C1_TRAIN_SHA,
            "representation": REPRESENTATION,
            "core": {"path": str(core_path), "sha256": CORE_SHA},
            "loader_function": "build_model", "save_function": "save_checkpoint",
            "format": "cumulative_lora_on_base", "parent_loaded_for_continuation": True}
    from .inputs import load_recipe
    recipe = load_recipe(spec)
    write(root / "proof.json", spec)
    write(root / "verification.json", {"recipe": recipe, "training_executed": False,
          "basis": "Native adapted_train_source plus hash-pinned screened C1 reporting-label replacement."})
    seal(root, "training_source_proof")
    return {"proof": str(root / "proof.json"), "core_sha256": CORE_SHA,
            "train_sha256": C1_TRAIN_SHA, "training_executed": False}


def register(registry_root, a_source, b_source, c1_proof_root, output):
    """Add verified loading recipes in a new registry; preserve existing pool receipts."""
    from .inputs import checkpoint_record, load_registry
    original = load_registry(registry_root)
    require(not original["synthetic"], "Real source registration does not accept synthetic checkpoints")
    require(set(original["candidates"]) == {identity(p, m, s) for p in PARENTS for m in METHODS for s in SEEDS},
            "Register sources for the complete declared 18-checkpoint experiment")
    bundle(c1_proof_root, "training_source_proof")
    root = destination(output, [registry_root, a_source, b_source, c1_proof_root])
    sources = {}
    for method, path in (("A", a_source), ("B", b_source)):
        path = Path(path).resolve()
        sources[method] = {"path": str(path), "sha256": sha(path), "loader_function": "build_model",
                           "save_function": "save_checkpoint", "format": "cumulative_lora_on_base",
                           "parent_loaded_for_continuation": True}
    sources["C1"] = read(Path(c1_proof_root) / "proof.json")
    result = deepcopy(original)
    for cid, candidate in original["candidates"].items():
        updated = checkpoint_record({"checkpoint": candidate["checkpoint"], "parent_checkpoint": candidate["parent_checkpoint"],
                                     "checkpoint_sha256": candidate["checkpoint_sha256"],
                                     "training_source": sources[candidate["method"]]})
        require({k: v for k, v in updated.items() if k != "recipe"}
                == {k: v for k, v in candidate.items() if k != "recipe"},
                f"Checkpoint metadata changed since the recorded opportunity analysis: {cid}")
        result["candidates"][cid] = updated
    write(root / "registry.json", result)
    write(root / "registration.json", {"source_specs": sources, "checkpoint_count": len(result["candidates"]),
                                      "checkpoint_weights_unchanged": True})
    seal(root, "registry", [registry_root, c1_proof_root])
    return {"registered_checkpoints": len(result["candidates"]), "registry": str(root),
            "checkpoint_weights_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    export = subs.add_parser("export-c1")
    export.add_argument("--core", required=True)
    export.add_argument("--output", required=True)
    attach = subs.add_parser("register")
    for name in ("registry", "a-source", "b-source", "c1-proof", "output"):
        attach.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        result = export_c1(args.core, args.output) if args.command == "export-c1" else register(
            args.registry, args.a_source, args.b_source, args.c1_proof, args.output)
    except (RoutingError, OSError, KeyError) as exc:
        parser.exit(2, str(exc) + "\n")
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

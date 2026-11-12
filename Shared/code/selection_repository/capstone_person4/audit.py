"""Evidence-backed stage gates. Existence alone never unlocks scientific stages."""

import importlib.metadata
import ast
import json
from pathlib import Path
import re
import struct
import zipfile

from .config import artifact_root, require
from .provenance import PipelineError, digest, environment, read_json, read_rows, sha, source_hashes, unique_rows, write_bytes, write_json

STAGES = {
    "prepare": ("release", "parent_exposure", "overlap", "interfaces", "universe"),
    "score-reference": ("release", "parent_exposure", "overlap", "parent_weights", "interfaces", "runtime"),
    "select": ("release", "parent_exposure", "overlap", "interfaces", "universe"),
    "train": ("release", "parent_exposure", "overlap", "parent_weights", "interfaces", "runtime", "recipe"),
    "evaluate": ("evaluation", "parent_weights", "parent_exposure"),
    "compare": ("evaluation",),
}


def checked_file(path, expected, label):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"{label}: supply existing file (currently {path!r})")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PipelineError(f"{label}: record an actual SHA256, not a placeholder")
    actual = sha(path)
    if actual != expected:
        raise PipelineError(f"{label}: SHA256 mismatch: {actual} != {expected}")
    return Path(path)


def check_release(cfg):
    if cfg.get("data", {}).get("inherited_release"):
        from .bindings import verify_release
        return verify_release(cfg)
    data = cfg["data"]
    release = Path(require(cfg, "data.release"))
    meta_path = checked_file(release / "metadata/release_manifest.json", require(cfg, "data.release_sha256"), "release manifest")
    meta = read_json(meta_path)
    if meta.get("readiness", {}).get("data_ready_for_trainer") is not True:
        raise PipelineError("Release does not declare DATA_READY_FOR_TRAINER")
    files = meta.get("files", {})
    paths = [Path(require(cfg, "data.candidate_manifest")), release / "configs/formatting_contract.json"]
    contract_root = Path(require(cfg, "data.contract_root"))
    paths.extend(contract_root / "capstone_person1" / name for name in
                 ("__init__.py", "person1_data_contract.py", "pipeline_controls.py", "instance_matching.py"))
    for path in paths:
        try:
            relative = path.relative_to(release).as_posix()
        except ValueError as exc:
            raise PipelineError(f"Use encoder/candidate files inside the verified immutable release: {path}") from exc
        checked_file(path, files.get(relative, {}).get("sha256"), relative)
    rows = read_rows(data["candidate_manifest"])
    unique_rows(rows)
    for row in rows:
        if row.get("synthetic") is True or row.get("eligible") is not True or row.get("group_excluded") is True:
            raise PipelineError(f"Ineligible/synthetic canonical candidate: {row.get('row_id')}")
        for key in ("problem", "completion", "solution", "final_answer", "dataset_id", "dataset_revision", "original_source"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise PipelineError(f"Missing canonical field {key}: {row['row_id']}")
        for key in ("prompt_tokens", "completion_tokens", "supervised_tokens", "total_tokens"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise PipelineError(f"Invalid canonical token count {key}: {row['row_id']}")
        if row["supervised_tokens"] != row["completion_tokens"] + 1 or row["total_tokens"] != row["prompt_tokens"] + row["supervised_tokens"]:
            raise PipelineError(f"Inconsistent canonical token accounting: {row['row_id']}")
        if "length_bin" not in row:
            raise PipelineError("Preserved source-specific length_bin required")
    fmt = read_json(release / "configs/formatting_contract.json")
    if (fmt.get("apply_chat_template") is not False or fmt.get("add_special_tokens") is not False
            or fmt.get("append_eos_count") != 1 or fmt.get("loss") != "completion_and_eos_only"):
        raise PipelineError("Unsupported serialization/mask/EOS policy; reconcile instead of silently changing it")
    if fmt.get("prompt_template_sha256") != __import__("hashlib").sha256(fmt["prompt_template"].encode()).hexdigest():
        raise PipelineError("Prompt-template fingerprint differs")
    return {"release_sha256": sha(meta_path), "candidate_sha256": sha(paths[0]), "rows": len(rows),
            "format_sha256": sha(paths[1]), "limitations": "References are structurally checked, not mathematically verified corpus-wide"}


def parent_exposure(cfg):
    if cfg.get("data", {}).get("inherited_release"):
        from .bindings import parent_exposure as bound_exposure
        return bound_exposure(cfg)
    parent = cfg["parent"]
    evidence_path = checked_file(require(cfg, "parent.exposure_evidence"), require(cfg, "parent.exposure_sha256"), "parent exposure evidence")
    evidence = read_json(evidence_path)
    run_path = checked_file(require(cfg, "parent.run_config"), require(cfg, "parent.run_config_sha256"), "parent run config")
    run = read_json(run_path)
    summary_path = checked_file(require(cfg, "parent.training_summary"), require(cfg, "parent.training_summary_sha256"), "parent training summary")
    summary = read_json(summary_path)
    expected = {"adapter_sha256": require(cfg, "parent.adapter_sha256"), "run_config_sha256": sha(run_path),
                "training_summary_sha256": sha(summary_path), "model": require(cfg, "parent.model"),
                "revision": require(cfg, "parent.revision"), "run_id": require(cfg, "parent.run_id")}
    if any(evidence.get(k) != v for k, v in expected.items()):
        raise PipelineError("Exposure evidence does not bind the configured run, model, summary and adapter")
    if evidence.get("complete_history") is not True or not evidence.get("history_basis"):
        raise PipelineError("Resolve all prior training stages; complete_history and evidence-based history_basis required")
    if run.get("model") != parent["model"] or run.get("revision") != parent["revision"] or summary.get("complete") is not True:
        raise PipelineError("Parent model/revision or completed-training state differs")
    all_rows, latest = {}, None
    for entry in evidence.get("exposures", []):
        path = Path(entry["manifest"])
        if not path.is_absolute():
            path = evidence_path.parent / path
        checked_file(path, entry.get("sha256"), "actual exposure manifest")
        rows = read_rows(path)
        unique_rows(rows)
        if any(r.get("synthetic") is True for r in rows):
            raise PipelineError("Synthetic exposure fixtures cannot bind a real parent")
        if type(entry.get("epochs")) is not int or entry["epochs"] < 1:
            raise PipelineError("Exposure evidence must record positive epochs/repeats for every stage")
        for row in rows:
            if row["row_id"] in all_rows and all_rows[row["row_id"]] != row:
                raise PipelineError("Conflicting parent exposure rows")
            all_rows[row["row_id"]] = row
        if sha(path) == run.get("manifest_sha256"):
            latest = rows
    if not all_rows or latest is None:
        raise PipelineError("Actual exposure manifests do not include the run-config training manifest")
    if set(summary.get("seen_ids", [])) != {r["row_id"] for r in latest} or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in latest}:
        raise PipelineError("Training-summary exposure IDs/groups do not match the actual last-stage manifest")
    if run.get("rows") != len(latest) or run.get("epochs") != next(e["epochs"] for e in evidence["exposures"] if e["sha256"] == run["manifest_sha256"]):
        raise PipelineError("Parent run row/epoch counts differ from exposure evidence")
    unique_rows(list(all_rows.values()))
    return list(all_rows.values()), {**expected, "exposure_sha256": sha(evidence_path), "unique_rows": len(all_rows)}


def check_weights(cfg):
    parent = cfg["parent"]
    adapter = Path(require(cfg, "parent.adapter"))
    weights = checked_file(adapter / "adapter_model.safetensors", require(cfg, "parent.adapter_sha256"), "parent adapter weights")
    conf = checked_file(adapter / "adapter_config.json", require(cfg, "parent.adapter_config_sha256"), "parent adapter config")
    a = read_json(conf)
    if a.get("base_model_name_or_path") != require(cfg, "parent.model") or a.get("peft_type") != "LORA":
        raise PipelineError("Expected a LoRA adapter for the configured base; merged/full weights require a separate loader")
    if a.get("revision") not in (None, cfg["parent"]["revision"]):
        raise PipelineError("Adapter base revision differs")
    with weights.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise PipelineError("Truncated safetensors file")
        length = struct.unpack("<Q", prefix)[0]
        if length > 32 * 1024 * 1024:
            raise PipelineError("Invalid safetensors header size")
        header = json.loads(stream.read(length))
        size = weights.stat().st_size - 8 - length
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if not tensors or not any("lora_A" in k for k in tensors) or not any("lora_B" in k for k in tensors):
            raise PipelineError("Safetensors lacks trained LoRA tensors")
        ranges = sorted(tuple(v["data_offsets"]) for v in tensors.values())
        if ranges[0][0] != 0 or ranges[-1][1] != size or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise PipelineError("Incomplete/overlapping adapter tensor data")
    if not re.fullmatch(r"[0-9a-f]{40}", require(cfg, "parent.revision")):
        raise PipelineError("Base revision must be an immutable 40-character commit")
    _, exposure = parent_exposure(cfg)
    return {"weights_sha256": sha(weights), "adapter_config_sha256": sha(conf), "tensor_count": len(tensors), **exposure}


def check_overlap(cfg):
    if cfg.get("protection"):
        from .protection import check_inputs
        return check_inputs(cfg)
    data = cfg["data"]
    path = checked_file(require(cfg, "data.overlap_report"), require(cfg, "data.overlap_sha256"), "overlap report")
    report = read_json(path)
    hashes, splits = {}, {}
    for role in ("development", "final"):
        manifest = checked_file(require(cfg, f"evaluation.{role}_manifest"), require(cfg, f"evaluation.{role}_sha256"), f"{role} questions")
        rows = read_rows(manifest)
        ids = [r.get("problem_id", r.get("row_id")) for r in rows]
        if None in ids or len(set(ids)) != len(ids):
            raise PipelineError(f"{role} questions require unique stable problem_id/row_id")
        if any(not isinstance(r.get("problem"), str) or not r["problem"].strip()
               or not r.get("benchmark") or (r.get("final_answer") is None and r.get("answer") is None) for r in rows):
            raise PipelineError(f"{role} questions require problem, benchmark and reference answer")
        splits[role] = rows
        hashes[role + "_sha256"] = sha(manifest)
    for identity in (lambda r: r.get("problem_id", r.get("row_id")),
                     lambda r: " ".join(r["problem"].split()), lambda r: r.get("problem_group_id")):
        development = {identity(r) for r in splits["development"]} - {None}
        final = {identity(r) for r in splits["final"]} - {None}
        if development & final:
            raise PipelineError("Development and final evaluation questions/groups overlap")
    expected = {"candidate_sha256": sha(require(cfg, "data.candidate_manifest")),
                "release_sha256": require(cfg, "data.release_sha256"), **hashes}
    if report.get("verified") is not True or not report.get("method") or not report.get("limitations"):
        raise PipelineError("Overlap report needs verified controls, method and explicit limitations")
    if any(report.get(k) != v for k, v in expected.items()) or not isinstance(report.get("excluded_group_ids"), list):
        raise PipelineError("Overlap report must bind this candidate release and both exact frozen question manifests")
    return {**expected, "excluded_group_ids": report["excluded_group_ids"], "overlap_sha256": sha(path)}


def check_interfaces(cfg):
    path = checked_file(require(cfg, "integration.notebook"), require(cfg, "integration.notebook_sha256"), "Person 2 notebook")
    from .person2_adapter import notebook_sources, continuation_source
    sources = notebook_sources(path)
    continuation_source(sources["trainer_core.py"])
    return {"notebook_sha256": sha(path), "source_hashes": {k: digest(v) for k, v in sources.items()}}


def check_runtime(cfg):
    require(cfg, "scoring.dtype")
    for dependency in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"):
        importlib.metadata.version(dependency)
    import torch
    if not torch.cuda.is_available():
        raise PipelineError("Person 2's 4-bit loader requires an allocated NVIDIA CUDA GPU")
    if cfg["scoring"]["dtype"] not in ("bfloat16", "float16"):
        raise PipelineError("Person 2 loader supports configured bfloat16/float16 autocast")
    if cfg["scoring"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise PipelineError("Configured bfloat16 is unsupported on this GPU")
    return {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, **environment()}


def check_recipe(cfg):
    recipe = require(cfg, "training")
    if recipe.get("epochs") != 1 or recipe.get("packing") is not False or recipe.get("microbatch_size") != 1:
        raise PipelineError("Continuation requires one pass, no packing, microbatch_size=1")
    if recipe.get("optimizer") != "AdamW" or recipe.get("schedule") != "linear_warmup_constant":
        raise PipelineError("Supported recipe: fresh AdamW with linear_warmup_constant")
    for key in ("accumulation", "warmup_updates", "save_every"):
        require(cfg, f"training.{key}")
        if type(recipe.get(key)) is not int or recipe[key] < (0 if key == "warmup_updates" else 1):
            raise PipelineError(f"Resolve positive training.{key} (warmup may be zero)")
    require(cfg, "training.learning_rate")
    if not isinstance(recipe.get("learning_rate"), (int, float)) or not 0 < recipe["learning_rate"] < 1:
        raise PipelineError("Resolve a finite positive training.learning_rate")
    if type(recipe.get("seed")) is not int:
        raise PipelineError("Resolve training.seed")
    return recipe


def check_evaluation(cfg):
    path = checked_file(require(cfg, "evaluation.evaluation_contract"), require(cfg, "evaluation.contract_sha256"), "evaluation contract")
    contract = read_json(path)
    check_overlap(cfg)
    if cfg.get("protection"):
        from .protection import canonical_question, question_sources
        protected, _ = question_sources(cfg)
        for role in ("development", "final"):
            manifest = checked_file(require(cfg, f"evaluation.{role}_manifest"),
                                    require(cfg, f"evaluation.{role}_sha256"), f"{role} evaluation questions")
            questions = [canonical_question(r) for r in read_rows(manifest)]
            identity = lambda r: (r["benchmark"], r["problem_id"], r["problem"], r["final_answer"])
            expected = {identity(r) for r in protected if r["role"] == role}
            if len(questions) != len(expected) or {identity(r) for r in questions} != expected:
                raise PipelineError("Evaluation questions differ from their frozen protected identities")
    for key in ("decoding", "max_new_tokens", "stopping", "checker_version", "precision", "prompt_sha256"):
        if not contract.get(key):
            raise PipelineError(f"Frozen evaluation contract lacks {key}")
    if contract["decoding"] != "greedy":
        raise PipelineError("Primary evaluation requires greedy decoding")
    script = checked_file(require(cfg, "evaluation.evaluator_script"), require(cfg, "evaluation.script_sha256"), "Person 3 evaluator adapter")
    tree = ast.parse(script.read_text(encoding="utf-8-sig"))
    if not any(isinstance(node, ast.FunctionDef) and node.name == "evaluate" for node in tree.body):
        raise PipelineError("Evaluator adapter must define evaluate(request)")
    # This is an explicit callable adapter contract, rather than guessing Person 3 CLI flags.
    if not contract.get("adapter_interface") == "person4-evaluate-v1":
        raise PipelineError("Provide the documented person4-evaluate-v1 adapter around the existing evaluator")
    dependencies = contract.get("evaluator_dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise PipelineError("Bind the actual evaluator and checker source dependencies by path and SHA256")
    for dependency in dependencies:
        dependency_path = Path(dependency["path"])
        if not dependency_path.is_absolute():
            dependency_path = path.parent / dependency_path
        checked_file(dependency_path, dependency.get("sha256"), "evaluator/checker dependency")
    validation = contract.get("checker_validation", {})
    validation_path = Path(validation.get("path") or "missing-checker-validation")
    if not validation_path.is_absolute():
        validation_path = path.parent / validation_path
    checked_file(validation_path, validation.get("sha256"), "checker validation evidence")
    checker = read_json(validation_path)
    if (checker.get("passed") is not True or checker.get("checker_version") != contract["checker_version"]
            or checker.get("source_dependencies") != dependencies or not checker.get("supported_answer_forms")
            or not isinstance(checker.get("cases_checked"), int) or checker["cases_checked"] < 1
            or checker.get("ambiguity_policy") != "unresolved_equivalence"):
        raise PipelineError("Checker fixtures/equivalence/ambiguity evidence is missing or not bound to this checker")
    return {"contract_sha256": sha(path), "script_sha256": sha(script)}


def check_universe(cfg):
    """Explicit common universe (Method B's bank) and its selector, when configured; otherwise no-op."""
    data, allocation = cfg.get("data", {}), cfg.get("allocation", {})
    if data.get("universe_mode") is None and allocation.get("selector") is None:
        return {"mode": "frozen-sample", "note": "no explicit universe declared"}
    if data.get("universe_mode") != "explicit" or allocation.get("selector") != "method-b-cpsat-v3":
        raise PipelineError("universe_mode=explicit and allocation.selector=method-b-cpsat-v3 must be declared together")
    bank = checked_file(require(cfg, "data.universe_manifest"), require(cfg, "data.universe_sha256"), "explicit universe bank")
    bins = checked_file(require(cfg, "data.length_bins"), require(cfg, "data.length_bins_sha256"), "parent length bins")
    recorded = checked_file(require(cfg, "allocation.b_selected_manifest"), require(cfg, "allocation.b_selected_sha256"), "Method B selected manifest")
    for key in ("deterministic_limit", "tie_limit", "stratum_tolerance", "utility_scale", "b_select_origin_sha256"):
        require(cfg, "allocation." + key)
    vendored = Path(__file__).with_name("b_select.py")
    if not vendored.is_file():
        raise FileNotFoundError("Vendored Method B selector b_select.py is missing")
    return {"universe_sha256": sha(bank), "length_bins_sha256": sha(bins), "b_selected_sha256": sha(recorded),
            "b_select_sha256": sha(vendored), "b_select_origin_sha256": allocation["b_select_origin_sha256"]}


CHECKS = {"release": check_release, "parent_exposure": lambda cfg: parent_exposure(cfg)[1],
          "parent_weights": check_weights, "overlap": check_overlap, "interfaces": check_interfaces,
          "runtime": check_runtime, "recipe": check_recipe, "evaluation": check_evaluation, "universe": check_universe}


def discover_inputs(cfg):
    """Read-only handoff leads. Inventory never substitutes for a readiness check."""
    notebook = Path(cfg["integration"]["notebook"])
    repo = notebook.parent
    inventory = {"notebook": {"path": str(notebook), "sha256": sha(notebook)},
                 "archives": [], "releases": [], "evaluation_exports": [], "verification": "discovery only; not stage authorization"}
    for path in sorted(repo.glob("*training*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            runs = []
            for name in names:
                if name.endswith("/run_config.json"):
                    config = json.loads(archive.read(name))
                    summary_name = name.removesuffix("run_config.json") + "training_summary.json"
                    summary = json.loads(archive.read(summary_name)) if summary_name in names else {}
                    runs.append({"entry": name, "model": config.get("model"), "revision": config.get("revision"),
                                 "rows": config.get("rows"), "epochs": config.get("epochs"), "seed": config.get("seed"),
                                 "manifest_sha256": config.get("manifest_sha256"), "complete": summary.get("complete"),
                                 "seen_id_count": len(summary.get("seen_ids", [])), "seen_group_count": len(summary.get("seen_groups", []))})
            inventory["archives"].append({"path": str(path), "sha256": sha(path), "entry_count": len(names), "runs": runs,
                                          "weight_entries": [n for n in names if n.endswith((".safetensors", ".bin"))]})
    for parent in (repo, repo.parent):
        release_root = parent / "capstone_person1/releases"
        if release_root.is_dir():
            for path in sorted(release_root.glob("*/metadata/release_manifest.json")):
                meta = read_json(path)
                inventory["releases"].append({"path": str(path), "sha256": sha(path),
                                               "release_id": meta.get("release_id"), "declared_readiness": meta.get("readiness")})
        exports = parent / "eval_results"
        if exports.is_dir():
            paths = sorted(exports.glob("*.config.json"))
            inventory["evaluation_exports"].append({"path": str(exports), "configuration_count": len(paths),
                        "source_scripts_present": [p.name for p in exports.glob("*.py")],
                        "role_note": "Inspect analysis/README.txt and frozen manifests; old outputs do not establish current evaluator readiness"})
    return inventory


def audit(cfg, write=False, target="approach"):
    targets = {"approach": ("prepare", "score-reference", "select", "train"),
               "inputs": ("prepare",), "evaluation": ("evaluate", "compare"), "all": tuple(STAGES)}
    if target not in targets:
        raise PipelineError(f"Unknown audit target: {target}")
    checks = []
    for name, check in CHECKS.items():
        try:
            evidence = check(cfg)
            status, remedy = "PASS", None
        except (OSError, ValueError, KeyError, TypeError, ImportError, struct.error) as exc:
            status = "MISSING" if isinstance(exc, (FileNotFoundError, ImportError)) or "Unresolved" in str(exc) else "MISMATCH"
            evidence, remedy = {"error": str(exc)}, f"Resolve {name}: {exc}"
        checks.append({"check": name, "status": status, "evidence": evidence, "remedy": remedy,
                       "affected_stages": [stage for stage, deps in STAGES.items() if name in deps]})
    by_name = {c["check"]: c for c in checks}
    gates = {stage: {"ready": all(by_name[d]["status"] == "PASS" for d in deps),
                     "blockers": [d for d in deps if by_name[d]["status"] != "PASS"]} for stage, deps in STAGES.items()}
    try:
        inventory = discover_inputs(cfg)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        inventory = {"status": "WARNING", "error": str(exc), "note": "Optional discovery failed; stage gates retain their independent evidence"}
    report = {"synthetic": False, "config_fingerprint": cfg["config_fingerprint"], "checks": checks,
              "stages": gates, "target": target, "ready": all(gates[s]["ready"] for s in targets[target]),
              "deferred_stages": [s for s in STAGES if s not in targets[target]],
              "environment": environment(), "code": source_hashes(), "input_inventory": inventory}
    if write:
        root = artifact_root(cfg) / "audit" / digest(report)[:16]
        write_json(root / "readiness_report.json", report)
        lines = ["# C0 readiness report", "", "| Check | Status | Remedy |", "| --- | --- | --- |"]
        lines += [f"| {c['check']} | {c['status']} | {(c['remedy'] or 'Verified; see JSON evidence').replace('|', '/')} |" for c in checks]
        lines += ["", "## Stage gates", ""] + [f"- {stage}: {'READY' if g['ready'] else 'BLOCKED: ' + ', '.join(g['blockers'])}" for stage, g in gates.items()]
        write_bytes(root / "readiness_report.md", ("\n".join(lines) + "\n").encode())
        report["report_directory"] = str(root)
    return report


def gate(cfg, stage):
    if stage not in STAGES:
        raise PipelineError(f"Unknown stage: {stage}")
    # Execute only relevant checks; selection does not import GPU code.
    results = {}
    for name in STAGES[stage]:
        try:
            results[name] = CHECKS[name](cfg)
        except (OSError, ValueError, KeyError, TypeError, ImportError, struct.error) as exc:
            raise PipelineError(f"{stage} blocked by {name}: {exc}") from exc
    return results

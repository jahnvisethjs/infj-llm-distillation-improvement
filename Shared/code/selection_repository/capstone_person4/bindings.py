"""Verify derivative releases and supplied parent histories without historical paths."""

from pathlib import Path
import re

from .provenance import PipelineError, read_json, read_rows, sha, unique_rows, write_json


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise PipelineError(f"Input path escapes its package: {relative}")
    return path


def verified(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PipelineError(f"Missing recorded SHA256: {path}")
    actual = sha(path)
    if actual != expected:
        raise PipelineError(f"SHA256 mismatch: {path}: {actual} != {expected}")
    return actual


def resolve_parent(cfg):
    """Resolve one explicit index entry using package-relative, checksum-bound paths."""
    inputs, parent = cfg["inputs"], dict(cfg["parent"])
    package = Path(inputs["parent_package"])
    if Path(inputs["parent_index"]) != package / "parent_index.json":
        raise PipelineError("Parent index must belong to the configured package")
    verified(inputs["parent_index"], inputs["parent_index_sha256"])
    verified(package / "SHA256SUMS.json", inputs["parent_checksums_sha256"])
    index = read_json(inputs["parent_index"])
    matches = [r for r in index if r["model"] == parent["model"] and r["seed"] == parent["training_seed"]]
    if len(matches) != 1 or matches[0]["model_revision"] != parent["revision"]:
        raise PipelineError("Parent index must identify exactly the requested model/revision/training seed")
    entry = matches[0]
    checksums = read_json(package / "SHA256SUMS.json")
    run = contained(package, entry["archive_run_directory"])
    adapter = contained(package, entry["adapter_directory"])
    paths = {"adapter": adapter, "run_config": run / "run_config.json", "training_summary": run / "training_summary.json"}
    for key, path in paths.items():
        if parent.get(key) is not None and Path(parent[key]) != path:
            raise PipelineError(f"Configured parent {key} differs from selected index entry")
        parent[key] = str(path)
    files = {"adapter_sha256": adapter / "adapter_model.safetensors",
             "adapter_config_sha256": adapter / "adapter_config.json",
             "run_config_sha256": paths["run_config"], "training_summary_sha256": paths["training_summary"]}
    for key, path in files.items():
        expected = checksums.get(path.relative_to(package).as_posix())
        if parent.get(key) is not None and parent[key] != expected:
            raise PipelineError(f"Configured {key} differs from parent package")
        parent[key] = expected
    parent["run_id"] = entry["archive_run_directory"]
    return parent


def verify_release(cfg):
    data = cfg["data"]
    root = Path(data["release"])
    meta_path = root / "metadata/release_manifest.json"
    verified(meta_path, data["release_sha256"])
    meta = read_json(meta_path)
    if meta.get("readiness", {}).get("data_ready_for_trainer") is not True:
        raise PipelineError("Release does not declare trainer readiness")
    checksum_path = root / "SHA256SUMS.txt"
    verified(checksum_path, data["checksums_sha256"])
    inventory = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        inventory[relative] = verified(contained(root, relative), expected)
    for relative, item in meta["files"].items():
        if inventory.get(relative) != item["sha256"]:
            raise PipelineError(f"Release manifest/checksum inventory disagree: {relative}")
    inherited = Path(data["inherited_release"])
    original_path = inherited / "metadata/release_manifest.json"
    verified(original_path, meta["original_manifest_sha256"])
    original = read_json(original_path)
    encoder = Path(data["inherited_encoder"])
    code = {}
    # Verify all supplied Python dependencies, including imports of the encoder.
    for path in sorted((encoder / "capstone_person1").rglob("*.py")):
        relative = path.relative_to(encoder).as_posix()
        expected = original["files"].get("code/" + relative, {}).get("sha256")
        code[relative] = verified(path, expected)
    for name in ("__init__.py", "person1_data_contract.py", "pipeline_controls.py", "instance_matching.py"):
        if "capstone_person1/" + name not in code:
            raise PipelineError(f"Missing inherited encoder dependency: {name}")
    fmt_path = root / "configs/formatting_contract.json"
    fmt = read_json(fmt_path)
    verified(fmt_path, original["files"]["configs/formatting_contract.json"]["sha256"])
    if (fmt.get("apply_chat_template") is not False or fmt.get("add_special_tokens") is not False
            or fmt.get("append_eos_count") != 1 or fmt.get("loss") != "completion_and_eos_only"):
        raise PipelineError("Unsupported serialization/mask/EOS policy")
    import hashlib
    if hashlib.sha256(fmt["prompt_template"].encode()).hexdigest() != fmt["prompt_template_sha256"]:
        raise PipelineError("Prompt template hash differs")
    candidate = Path(data["candidate_manifest"])
    if candidate != root / "manifests/selection_bank.jsonl":
        raise PipelineError("V4 candidates must come from the supplied selection_bank.jsonl")
    rows = read_rows(candidate)
    unique_rows(rows)
    for row in rows:
        for key in ("problem", "completion", "solution", "final_answer", "dataset_id", "dataset_revision", "original_source"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise PipelineError(f"Missing canonical {key}: {row['row_id']}")
        for key in ("prompt_tokens", "completion_tokens", "supervised_tokens", "total_tokens"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise PipelineError(f"Invalid token count {key}: {row['row_id']}")
        if row["supervised_tokens"] != row["completion_tokens"] + 1 or row["total_tokens"] != row["prompt_tokens"] + row["supervised_tokens"]:
            raise PipelineError("Canonical token accounting differs")
        if "length_bin" not in row:
            raise PipelineError("Missing inherited length bin")
    return {"release_sha256": sha(meta_path), "candidate_sha256": sha(candidate), "rows": len(rows),
            "format_sha256": sha(fmt_path), "inventory": inventory,
            "inherited_release_sha256": sha(original_path), "encoder_files": code,
            "limitations": "Structural reference checks do not prove mathematical correctness"}


def parent_exposure(cfg):
    parent, data = cfg["parent"], cfg["data"]
    if resolve_parent(cfg) != parent:
        raise PipelineError("Selected parent binding changed")
    package = Path(cfg["inputs"]["parent_package"])
    checksums = read_json(package / "SHA256SUMS.json")
    run_path = Path(parent["run_config"])
    run_root = run_path.parent
    inventory = {}
    for relative, expected in checksums.items():
        path = contained(package, relative)
        if run_root in path.parents:
            inventory[relative] = verified(path, expected)
    run = read_json(run_path)
    summary = read_json(parent["training_summary"])
    template_path, policy_path = run_root / "prompt_template.json", run_root / "tokenization_policy.json"
    verified(template_path, run["template_sha256"])
    verified(policy_path, run["policy_sha256"])
    formatting = read_json(Path(data["release"]) / "configs/formatting_contract.json")
    policy = read_json(policy_path)
    if (read_json(template_path).get("template") != formatting["prompt_template"]
            or policy.get("add_special_tokens") is not False or policy.get("append_eos") is not True
            or policy.get("max_total_tokens") != formatting["max_total_tokens"]):
        raise PipelineError("Parent serialization/tokenization policy differs from V4")
    manifest = Path(data["release"]) / "manifests/train_n1000.jsonl"
    verified(manifest, run["manifest_sha256"])
    rows = read_rows(manifest)
    ids, groups = unique_rows(rows)
    entry = next(r for r in read_json(cfg["inputs"]["parent_index"])
                 if r["model"] == parent["model"] and r["seed"] == parent["training_seed"])
    if (run.get("model") != parent["model"] or run.get("revision") != parent["revision"]
            or run.get("seed") != parent["training_seed"] or run.get("parent", "missing") is not None
            or run.get("epochs") != 1 or run.get("rows") != len(rows)
            or summary.get("complete") is not True or set(summary.get("seen_ids", [])) != ids
            or set(summary.get("seen_groups", [])) != groups or len(summary["seen_ids"]) != len(ids)
            or len(summary["seen_groups"]) != len(groups) or entry["complete"] is not True
            or entry["manifest_sha256"] != run["manifest_sha256"] or entry["unique_examples"] != len(rows)):
        raise PipelineError("Parent seed/model/complete history/exposure binding differs")
    seen = read_rows(run_root / "seen_groups.csv")
    labels = read_rows(run_root / "checks/train_n1000/actual_labels.csv")
    if len(seen) != len(groups) or {r["problem_group_id"] for r in seen} != groups:
        raise PipelineError("Saved parent groups differ from actual manifest")
    counts = {r["row_id"]: int(r["actual_supervised_tokens"]) for r in labels}
    expected = {r["row_id"]: r["supervised_tokens"] for r in rows}
    if len(labels) != len(rows) or counts != expected or sum(counts.values()) != summary.get("processed_tokens") or sum(counts.values()) != entry["processed_tokens"]:
        raise PipelineError("Saved actual-label counts differ from V4 parent manifest")
    encoder = Path(data["inherited_encoder"]) / "capstone_person1"
    for name, expected_hash in run["reference_code_sha256"].items():
        verified(contained(encoder, name), expected_hash)
    return rows, {"model": parent["model"], "revision": parent["revision"], "run_id": parent["run_id"],
                  "parent_training_seed": parent["training_seed"], "adapter_sha256": parent["adapter_sha256"],
                  "run_config_sha256": sha(run_path), "training_summary_sha256": sha(parent["training_summary"]),
                  "template_sha256": sha(template_path), "tokenization_policy_sha256": sha(policy_path),
                  "manifest_sha256": sha(manifest), "unique_rows": len(rows), "processed_tokens": sum(counts.values()),
                  "complete_history": True, "history_basis": "Pinned base; saved parent:null; one epoch; exact exposure and label records",
                  "recorded_label_evidence": "historical; actual re-encoding required before model stages", "inventory": inventory}


def write_bindings(cfg, release, parent):
    from .config import artifact_root
    root = artifact_root(cfg, "prepare") / "input_bindings"
    write_json(root / "release_binding.json", release)
    write_json(root / "parent_binding.json", parent)
    write_json(root / "encoder_binding.json", {"inherited_release_sha256": release["inherited_release_sha256"],
                                              "files": release["encoder_files"], "format_sha256": release["format_sha256"]})
    write_json(root / "config_snapshots" / (cfg["config_fingerprint"] + ".json"), cfg)
    return str(root)

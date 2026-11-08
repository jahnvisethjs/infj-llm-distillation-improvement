"""Explicit, config-relative inputs; unresolved values never become guessed defaults."""

from copy import deepcopy
from pathlib import Path

from .provenance import PipelineError, digest, read_json

PATH_FIELDS = {
    "output_root", "notebook", "release", "contract_root", "candidate_manifest",
    "overlap_report", "adapter", "run_config", "exposure_evidence", "training_manifest",
    "development_manifest", "final_manifest", "evaluation_contract", "checkpoint",
    "evaluator_script", "predictions", "training_summary", "method_freeze",
    "inherited_release", "inherited_encoder", "parent_package", "parent_index",
    "reserved_manifest", "manifest", "matcher_source", "checker_source", "reviews",
    "tokenizer_assets", "universe_manifest", "length_bins", "b_selected_manifest",
}


def load_config(path, profile=None, training_seed=None):
    path = Path(path).resolve()
    raw = read_json(path)
    if raw.get("schema_version") != 1:
        raise PipelineError("Expected configuration schema_version=1")
    selected = profile or raw.get("active_profile")
    if selected not in raw.get("profiles", {}):
        raise PipelineError(f"Unknown active model profile: {selected}")
    cfg = deepcopy(raw)
    cfg["parent"] = cfg.pop("profiles")[selected]
    cfg["active_profile"] = selected
    if training_seed is not None:
        # Continuation-seed replication: the training section is in the train fingerprint,
        # so each seed gets its own artifact root; preparation/scoring/selection are shared.
        if type(training_seed) is not int or training_seed < 0:
            raise PipelineError("training_seed must be a nonnegative integer")
        cfg.setdefault("training", {})["seed"] = training_seed

    def resolve(obj):
        if isinstance(obj, dict):
            return {k: str((path.parent / v).resolve()) if k in PATH_FIELDS and isinstance(v, str)
                    else resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    cfg = resolve(cfg)
    if cfg.get("data", {}).get("inherited_release"):
        from .bindings import resolve_parent
        cfg["parent"] = resolve_parent(cfg)
    if not cfg.get("output_root"):
        raise PipelineError("Set output_root outside the immutable data release")
    output = Path(cfg["output_root"])
    release = cfg.get("data", {}).get("release")
    immutable = [release, cfg.get("data", {}).get("inherited_release"),
                 cfg.get("data", {}).get("inherited_encoder"), cfg.get("inputs", {}).get("parent_package")]
    if any(p and (output == Path(p) or Path(p) in output.parents) for p in immutable):
        raise PipelineError("output_root must be outside immutable input directories")
    if cfg.get("synthetic") is not False:
        raise PipelineError("Real-stage configuration requires synthetic=false; use dry-run for fixtures")
    cfg["config_fingerprint"] = digest(cfg)
    return cfg


def without_paths(value):
    """Fingerprints bind content hashes, never machine-specific input locations."""
    if isinstance(value, dict):
        return {k: without_paths(v) for k, v in value.items() if k not in PATH_FIELDS}
    if isinstance(value, list):
        return [without_paths(v) for v in value]
    return value


def require(cfg, dotted):
    current = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current or current[part] is None:
            raise PipelineError(f"Unresolved required configuration: {dotted}")
        current = current[part]
    return current


STAGE_SOURCES = {
    "prepare": ("config.py", "audit.py", "bindings.py", "protection.py", "data.py", "person2_adapter.py", "provenance.py"),
    "score-reference": ("reference_scoring.py", "person2_adapter.py", "provenance.py"),
    "select": ("selection.py", "allocation.py", "b_select.py", "continuation.py", "provenance.py"),
    "train": ("continuation.py", "person2_adapter.py", "provenance.py"),
    "evaluate": ("evaluation.py", "provenance.py"),
}


def stage_fingerprint(cfg, stage):
    """Only dependencies of this stage and its prerequisites enter its identity."""
    from .provenance import source_hashes
    if stage not in STAGE_SOURCES:
        raise PipelineError(f"Unknown dependency stage: {stage}")
    parent = cfg.get("parent", {})
    preparation = {k: cfg.get(k) for k in ("data", "protection", "integration", "inputs")}
    preparation["parent_exposure"] = {k: v for k, v in parent.items() if k not in ("adapter", "adapter_sha256", "adapter_config_sha256")}
    if not cfg.get("protection"):
        preparation["questions"] = {k: cfg.get("evaluation", {}).get(k) for k in
                                    ("development_manifest", "development_sha256", "final_manifest", "final_sha256")}
    sources = source_hashes()
    payload = {"preparation": preparation, "code": {k: sources[k] for k in STAGE_SOURCES["prepare"]}}
    if stage != "prepare":
        payload.update(parent=parent, scoring=cfg.get("scoring"))
        payload["code"].update({k: sources[k] for k in STAGE_SOURCES["score-reference"]})
    if stage in ("select", "train", "evaluate"):
        payload["allocation"] = cfg.get("allocation")
        payload["code"].update({k: sources[k] for k in STAGE_SOURCES["select"]})
    if stage in ("train", "evaluate"):
        payload["training"] = cfg.get("training")
        payload["code"].update({k: sources[k] for k in STAGE_SOURCES["train"]})
    if stage == "evaluate":
        payload["evaluation"] = cfg.get("evaluation")
        payload["code"].update({k: sources[k] for k in STAGE_SOURCES["evaluate"]})
    return digest(without_paths(payload))


def artifact_root(cfg, stage="train"):
    return Path(cfg["output_root"]) / cfg["active_profile"] / stage_fingerprint(cfg, stage)[:16]

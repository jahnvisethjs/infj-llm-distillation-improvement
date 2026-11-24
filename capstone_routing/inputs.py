"""Bounded SOL discovery and checkpoint/evaluator evidence validation."""

import ast
import gzip
import hashlib
from pathlib import Path
import re

from .common import (METHODS, MODEL, PARENTS, QUESTIONS, REVISION, SEEDS, SETTINGS,
                     RoutingError, benchmark, bundle, checked, checked_spec, identity,
                     keyed, question, read, require, rows, sha)

ROOTS = {"a": "/scratch/msures15/eval_results",
         "b": "/scratch/sshind38/capstone_approach_b/outputs/B_method_B_document_v3/mixed/budget_1x/evaluation",
         "c1": "/scratch/sshind38/capstone_c0_c1_work/outputs/C0_C1_development_evaluation_v1/931e8fb821c7d70f/results"}
PARENT_HASHES = {17: "11a9de70df8b102e57676fb1c4b8ed61f9f0d1e7f524c6c0f0169368ea17b1cb",
                 123: "aa11429355543bcfe35e47856a3c6ada4e5e0eb7f4647318c1c68aee5bf1c3a2"}
ARMS = {"REPEAT": "A", "RANDOM": "B", "B": "B", "C1": "C1"}
SCREENED_C1_BANK = "ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f"


def settings(config, question_sha=None):
    expected = {**SETTINGS, **({"eval_file_sha256": question_sha} if question_sha else {})}
    differences = [k for k, v in expected.items() if type(config.get(k)) is not type(v) or config.get(k) != v]
    require(not differences, "Evaluator contract mismatch: " + ", ".join(differences))


def sibling_config(checkpoint):
    candidates = [checkpoint.parent / "run_config.json", checkpoint.parent.parent / "run_config.json"]
    existing = [p for p in candidates if p.is_file()]
    require(existing, f"Missing training metadata: {checkpoint}")
    configs = [read(p) for p in existing]
    require(all(c == configs[0] for c in configs), f"Conflicting training metadata: {checkpoint}")
    return existing[0]


def source_text(spec):
    path = checked_spec(spec)
    if path.suffix == ".ipynb":
        cells = ["".join(c.get("source", [])) for c in read(path)["cells"]]
        found = [s.partition("\n")[2] for s in cells if s.startswith("%%writefile trainer_core.py\n")]
        require(len(found) == 1, "Expected one Person 2 trainer definition cell")
        return found[0]
    return path.read_text(encoding="utf-8-sig")


def load_recipe(spec):
    """Only the explicitly audited cumulative-PEFT save representation is supported."""
    text = source_text(spec)
    representation = spec.get("representation", "module")
    verified_text = text
    function_proof = {}
    if representation != "module":
        from .proofs import CORE_SHA, C1_TRAIN_SHA, REPRESENTATION, reconstruct_c1
        require(representation == REPRESENTATION, "Unsupported training source representation")
        core_spec = spec.get("core", {})
        require(core_spec.get("sha256") == CORE_SHA, "Function proof requires the exact native loader/save core")
        core = checked_spec(core_spec).read_text(encoding="utf-8")
        require(text == reconstruct_c1(core) and spec.get("sha256") == C1_TRAIN_SHA,
                "Generated C1 function does not match its pinned reconstruction")
        verified_text = core
        function_proof = {"core_sha256": CORE_SHA, "train_source_sha256": C1_TRAIN_SHA}
    tree = ast.parse(verified_text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    loader = functions.get(spec.get("loader_function"))
    saver = functions.get(spec.get("save_function", "save_checkpoint"))
    require(loader is not None and saver is not None, "Name the audited trainer loader/save functions")
    load = ast.unparse(loader)
    save = ast.unparse(saver)
    require("PeftModel.from_pretrained" in load and "model.save_pretrained" in save,
            "Trainer does not use the supported existing-adapter/full-adapter save path")
    require("merge_and_unload" not in verified_text and "merge_and_unload" not in text and "get_peft_model" not in save,
            "Merged/reinitialized training save formats require a separate audited loader")
    require(spec.get("format") == "cumulative_lora_on_base" and spec.get("parent_loaded_for_continuation") is True,
            "Explicit audited continuation loading declaration is required")
    return {"format": "cumulative_lora_on_base", "source": spec,
            "loader_source_sha256": hashlib.sha256(load.encode()).hexdigest(),
            "save_source_sha256": hashlib.sha256(save.encode()).hexdigest(), **function_proof}


def checkpoint_record(spec):
    cp = Path(spec["checkpoint"]).resolve()
    run_path = Path(spec["run_config"]) if spec.get("run_config") else sibling_config(cp)
    train = read(run_path)
    require(train.get("model") == MODEL and train.get("revision") == REVISION, "Wrong backbone/revision")
    require(not train.get("smoke_only", train.get("training_smoke_only", False))
            and train.get("mode") not in ("debug", "smoke"), "Smoke checkpoint is not a candidate")
    method = ARMS.get(train.get("arm"))
    require(method in METHODS, "Unsupported policy (expected A/REPEAT, B/RANDOM, C1)")
    if method == "C1":
        require(train.get("shared_universe_sha256") == SCREENED_C1_BANK, "C1 is not the declared screened-bank experiment")
    require(spec.get("method", method) == method, "Method label conflicts with checkpoint arm")
    seed = train.get("continuation_training_seed", train.get("seed"))
    require(train.get("seed") == seed, "Conflicting continuation seeds")
    parent_hash = train.get("parent_adapter_sha256")
    parents = [p for p, h in PARENT_HASHES.items() if h == parent_hash]
    require(len(parents) == 1, "Checkpoint has no recognized original 0.5B parent")
    parent = parents[0]
    require(spec.get("parent_seed", parent) == parent and spec.get("continuation_seed", seed) == seed,
            "Explicit seed mapping conflicts with training evidence")
    cid = identity(parent, method, seed)
    config = read(cp / "adapter_config.json")
    require(config.get("peft_type") == "LORA" and config.get("base_model_name_or_path") == MODEL,
            "Unsupported adapter configuration")
    summary_path = Path(spec["training_summary"]) if spec.get("training_summary") else cp.parent.parent / "training_summary.json"
    summary = read(summary_path)
    require(summary.get("complete") is True, "Training did not complete")
    budget = train.get("branch_budget", {})
    target = budget.get("target_supervised_tokens", 318845)
    actual = summary.get("processed_tokens")
    require(target == 318845 and type(actual) is int and abs(actual - target) <= target * .005,
            "Training budget differs from the matched 1x continuation")
    require(train.get("epochs") == 1, "Expected one-pass continuation")
    parent_path = spec.get("parent_checkpoint", train.get("parent"))
    require(parent_path, "Missing parent checkpoint location")
    parent_cp = Path(parent_path)
    if parent_cp.name != "adapter":
        parent_cp = parent_cp / "adapter"
    checked(parent_cp / "adapter_model.safetensors", parent_hash)
    parent_run = Path(spec["parent_run_config"]) if spec.get("parent_run_config") else sibling_config(parent_cp)
    parent_train = read(parent_run)
    require(parent_train.get("seed") == parent and parent_train.get("model") == MODEL
            and parent_train.get("revision") == REVISION, "Parent metadata conflicts with lineage")
    cp_hash = sha(cp / "adapter_model.safetensors")
    if spec.get("checkpoint_sha256"):
        require(spec["checkpoint_sha256"] == cp_hash, "Checkpoint bytes differ from index")
    source = spec.get("training_source")
    # A declaration is not a substitute for source bytes; unknown formats remain
    # usable offline but cannot generate labels or live answers.
    recipe = load_recipe(source) if source else None
    if source:
        recorded = {train.get(k) for k in ("effective_train_source_sha256", "code_sha256", "notebook_sha256")}
        text_hash = hashlib.sha256(source_text(source).encode()).hexdigest()
        require(source["sha256"] in recorded or text_hash in recorded,
                "Audited loader/save source is not hash-bound to this training run")
        if source.get("representation") == "screened_c1_train_function_v1":
            require(method == "C1" and train.get("wrapper_sha256") ==
                    "50809b9e374fcc05a9b637c381396b46223ab85abcda3ca07cd4b6a5a4af882f",
                    "Function-only proof requires the matching native screened C1 wrapper")
    files = {str(p.resolve()): sha(p) for p in (run_path, summary_path, parent_run, cp / "adapter_config.json")}
    return {"id": cid, "method": method, "parent_seed": parent, "continuation_seed": seed,
            "checkpoint": str(cp), "checkpoint_sha256": cp_hash, "parent_checkpoint": str(parent_cp.resolve()),
            "parent_sha256": parent_hash, "parent_manifest_sha256": parent_train["manifest_sha256"],
            "train_manifest_sha256": train["manifest_sha256"], "model": MODEL, "revision": REVISION,
            "target_tokens": target, "actual_tokens": actual, "recipe": recipe, "evidence": files,
            "construction": {k: train.get(k) for k in ("arm", "epochs", "seed", "dtype", "learning_rate",
                "accumulation", "microbatch_size", "warmup_updates", "packing", "branch_optimizer_policy",
                "initial_optimizer_state", "shared_universe_sha256", "template_sha256", "policy_sha256")}}


def prediction_path(config_path):
    stem = str(config_path).removesuffix(".config.json")
    candidates = [Path(stem + suffix) for suffix in (".jsonl", ".jsonl.gz")]
    found = [p for p in candidates if p.is_file()]
    require(len(found) == 1, f"Missing or ambiguous predictions beside {config_path}")
    return found[0]


def prediction_records(spec, candidate, questions):
    path, sidecar = Path(spec["predictions"]), Path(spec["config"])
    config = read(sidecar)
    settings(config, spec["questions_sha256"])
    require(config.get("checkpoint_sha256") == candidate["checkpoint_sha256"]
            and config.get("train_manifest_sha256") == candidate["train_manifest_sha256"]
            and config.get("train_seed") == candidate["continuation_seed"]
            and ARMS.get(config.get("train_arm")) == candidate["method"], "Prediction checkpoint/arm/seed mismatch")
    receipt_path = spec.get("receipt")
    if receipt_path:
        receipt = read(receipt_path)
        require(receipt.get("complete") is True, "Evaluation receipt is incomplete")
        checked(sidecar, receipt.get("config_sha256"))
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == receipt.get("predictions_sha256"), "Prediction receipt hash mismatch")
    records = keyed(rows(path))
    require(set(records) <= set(questions), "Prediction file contains unknown question IDs")
    outcomes = {}
    counts = {"missing_rows": len(questions) - len(records), "invalid_answers": 0}
    for key in questions:
        row = records.get(key)
        if row is None:
            outcomes[key] = 0
            continue
        status, correct = row.get("parse_status"), row.get("correct")
        require(type(correct) is bool or type(correct) is int and correct in (0, 1), "Correctness must be boolean or 0/1")
        require(status not in ("error", "verify_error", "execution_error"), "Checker execution failure blocks comparison")
        require(status in ("ok", "empty", "empty_marker", "no_answer", "ambiguous", "unparseable_number", "unparseable_math"),
                f"Unknown parser status: {status}")
        missing_answer = row.get("parsed_answer") in (None, "") or row.get("raw_output") in (None, "")
        require(not correct or status == "ok" and not missing_answer, "Inconsistent correctness/answer/parser status")
        for field, expected in (("checkpoint_sha256", candidate["checkpoint_sha256"]),
                                ("train_manifest_sha256", candidate["train_manifest_sha256"]),
                                ("model_revision", MODEL + "@" + REVISION), ("parser_version", SETTINGS["parser_version"]),
                                ("eval_config_hash", config.get("eval_config_hash"))):
            require(row.get(field) == expected, f"Prediction row conflicts with {field}")
        outcomes[key] = int(bool(correct) and status == "ok" and not missing_answer)
        counts["invalid_answers"] += int(status != "ok" or missing_answer)
    return outcomes, {**counts, "rows": len(records), "questions": len(questions),
                       "predictions_sha256": sha(path), "config_sha256": sha(sidecar)}


def development_questions(cfg):
    result, specs = {}, {}
    for bench in ("gsm", "math"):
        spec = cfg["questions"][bench]
        require(spec["sha256"] == QUESTIONS[bench], "Development question manifest is not frozen")
        records = keyed(rows(checked_spec(spec)))
        require(len(records) == 500 and all(k[0] == bench for k in records), "Expected 500 frozen questions per benchmark")
        for row in records.values():
            question(row)
        result.update(records)
        specs[bench] = spec
    return result, specs


def inspect(cfg):
    """Read only. No model imports, jobs, downloads, or artifact writes."""
    candidates, predictions, rejected, ambiguous = {}, {}, [], set()
    explicit = cfg.get("checkpoints", [])
    for spec in explicit:
        try:
            record = checkpoint_record(spec)
            if record["id"] in candidates or record["id"] in ambiguous:
                candidates.pop(record["id"], None)
                ambiguous.add(record["id"])
                raise RoutingError("Duplicate checkpoint identity in explicit index; no candidate was selected")
            candidates[record["id"]] = record
        except (OSError, KeyError, ValueError) as exc:
            rejected.append({"source": spec.get("checkpoint"), "reason": str(exc)})
    indexed = list(cfg.get("predictions", []))
    if not indexed:
        for policy, root in {**ROOTS, **cfg.get("roots", {})}.items():
            directory = Path(root)
            if not directory.is_dir():
                rejected.append({"source": str(directory), "reason": "Input root is unavailable"})
                continue
            for sidecar in sorted(directory.rglob("*.config.json")):
                if any(part in ("native", "smoke", "archives", "evaluation_recovery_archives") for part in sidecar.parts):
                    continue
                try:
                    config = read(sidecar)
                    if ARMS.get(config.get("train_arm")) != {"a": "A", "b": "B", "c1": "C1"}[policy]:
                        continue
                    if config.get("eval_file_sha256") not in QUESTIONS.values():
                        continue
                    bench = next(b for b, h in QUESTIONS.items() if h == config["eval_file_sha256"])
                    source = cfg.get("training_sources", {}).get(ARMS[config["train_arm"]])
                    record = checkpoint_record({"checkpoint": config["checkpoint_id"],
                                                **({"training_source": source} if source else {})})
                    cid = record["id"]
                    require(cid not in ambiguous, "Ambiguous checkpoint identity requires an explicit index")
                    if cid in candidates:
                        if candidates[cid]["checkpoint_sha256"] != record["checkpoint_sha256"]:
                            candidates.pop(cid)
                            ambiguous.add(cid)
                            raise RoutingError("Conflicting candidate hashes; no checkpoint selected for this identity")
                    else:
                        candidates[cid] = record
                    item = {"candidate": cid, "benchmark": bench, "predictions": str(prediction_path(sidecar)),
                            "config": str(sidecar), "questions_sha256": QUESTIONS[bench]}
                    if (sidecar.parent / "completion.json").is_file():
                        item["receipt"] = str(sidecar.parent / "completion.json")
                    indexed.append(item)
                except (OSError, KeyError, ValueError) as exc:
                    rejected.append({"source": str(sidecar), "reason": str(exc)})
    for item in indexed:
        key = item["candidate"] + "/" + benchmark(item["benchmark"])
        predictions.setdefault(key, []).append(item)
    return {"candidates": candidates, "predictions": predictions, "rejected": rejected,
            "expected_candidates": [identity(p, m, s) for p in PARENTS for s in SEEDS for m in METHODS],
            "synthetic": cfg.get("synthetic", False)}


def load_registry(root, recheck=True):
    bundle(root)
    registry = read(Path(root) / "registry.json")
    if recheck:
        for candidate in registry["candidates"].values():
            checked(Path(candidate["checkpoint"]) / "adapter_model.safetensors", candidate["checkpoint_sha256"])
            checked(Path(candidate["parent_checkpoint"]) / "adapter_model.safetensors", candidate["parent_sha256"])
            for path, expected in candidate["evidence"].items():
                checked(path, expected)
            if candidate["recipe"]:
                require(load_recipe(candidate["recipe"]["source"]) == candidate["recipe"], "Checkpoint loading recipe changed")
    return registry

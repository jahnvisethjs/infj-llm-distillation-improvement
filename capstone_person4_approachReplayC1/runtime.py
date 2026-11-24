"""Reuse native screened-C1's effective training loop, with explicit replay provenance."""

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

from . import METHOD, VERSION
from .common import (BANK_SHA, PARENT_SHA, RECIPE, SEEDS, TARGET, PilotError, attempt_name,
                     checked, digest, read, rows, sha, source_sha, write, write_bytes, write_rows)
from .selection import load_selection


def execution_source(source):
    label = "'preliminary_document_aligned_B'"
    if source.count(label) != 1:
        raise PilotError("Native effective trainer structure changed; review rather than adapting blindly")
    return source.replace(label, repr("screened_replay_c1_50_50_pilot"), 1)


def run_directory(bundle, seed, attempt, smoke=False):
    if seed not in SEEDS or (smoke and seed != 42):
        raise PilotError("Use continuation seeds 42/137/2026; the disposable smoke uses seed42")
    return bundle.root / ("smokes" if smoke else "runs") / f"seed{seed}_{attempt_name(attempt)}"


def training_rows(data, smoke=False):
    if smoke:
        if len(data["replay"]) < 8 or len(data["fresh"]) < 8:
            raise PilotError("Smoke requires eight verified examples from each portion")
        ids = {r["row_id"] for r in data["replay"][:8] + data["fresh"][:8]}
        return [r for r in data["train"] if r["row_id"] in ids]
    return data["train"]


def reconcile_labels(selected, encoded, members):
    if len(selected) != len(encoded):
        raise PilotError("Actual encoder coverage differs from the training manifest")
    membership = {r["row_id"]: r for r in members}
    totals = {"replay": 0, "fresh": 0}
    for row, actual in zip(selected, encoded):
        count = actual["counts"]["supervised_tokens"]
        member = membership[row["row_id"]]
        if count != row["supervised_tokens"] or count != member["supervised_tokens"]:
            raise PilotError("Actual trainer label count differs for " + row["row_id"])
        totals[member["portion"]] += count
    return totals


def validate_run(bundle, directory, seed, smoke=False, receipt=None):
    data, selection = load_selection(bundle)
    selected = training_rows(data, smoke)
    directory = Path(directory).resolve()
    parent = bundle.root / ("smokes" if smoke else "runs")
    if directory.parent != parent.resolve():
        raise PilotError("Run completion is outside this pilot")
    if receipt is None:
        receipt = read(directory / "completion.json")
    expected = {"passed": True, "version": VERSION, "identity": bundle.identity, "method": METHOD,
                "seed": seed, "smoke_only": smoke, "reload_probe_matched": True,
                "parent_sha256": PARENT_SHA, "selection_sha256": digest(selection),
                "unique_examples": len(selected), "actual_tokens": sum(r["supervised_tokens"] for r in selected),
                "effective_train_source_sha256": source_sha(execution_source(bundle.context.train_source))}
    if any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items()):
        raise PilotError("Run identity, selection, parent, or smoke evidence changed")
    if not receipt.get("changed_adapter_tensors"):
        raise PilotError("No adapter parameter update recorded")
    for filename, key in (("run_config.json", "run_config_sha256"), ("training_summary.json", "summary_sha256"),
                          ("training_log.jsonl", "log_sha256")):
        checked(directory / filename, receipt[key])
    config, summary = read(directory / "run_config.json"), read(directory / "training_summary.json")
    if any(config.get(k) != v for k, v in RECIPE.items()):
        raise PilotError("Executed training recipe changed")
    expected_config = {"seed": seed, "model": bundle.context.entry["model"], "revision": bundle.context.entry["revision"],
                       "dtype": "torch.bfloat16", "arm": METHOD, "mode": "debug" if smoke else "branch", "smoke_only": smoke,
                       "pilot_identity": bundle.identity, "selection_sha256": digest(selection),
                       "shared_universe_sha256": BANK_SHA, "parent_adapter_sha256": PARENT_SHA,
                       "initial_optimizer_state": "empty", "fresh_warmup_updates": 10,
                       "branch_optimizer_policy": "fresh_optimizer_and_scheduler",
                       "effective_train_source_sha256": expected["effective_train_source_sha256"],
                       "code_sha256": expected["effective_train_source_sha256"]}
    mismatches = {k: {"expected": v, "actual": config.get(k)}
                  for k, v in expected_config.items() if config.get(k) != v}
    if mismatches:
        raise PilotError("Executed run configuration differs: " + str(mismatches))
    manifest = bundle.root / "training_inputs" / ("smoke" if smoke else "full") / "train.jsonl"
    checked(manifest, config["manifest_sha256"])
    checked(manifest.parent / "effective_train.py", expected["effective_train_source_sha256"])
    if rows(manifest) != selected:
        raise PilotError("Executed manifest differs from the frozen mixed selection")
    if (summary.get("complete") is not True or summary.get("processed_tokens") != expected["actual_tokens"]
            or len(summary.get("seen_ids", [])) != len(selected)
            or set(summary["seen_ids"]) != {r["row_id"] for r in selected}
            or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in selected}):
        raise PilotError("Training exposure is incomplete or differs from the manifest")
    totals = {portion: sum(r["supervised_tokens"] for r in data["membership"]
                          if r["portion"] == portion and r["row_id"] in set(summary["seen_ids"]))
              for portion in ("replay", "fresh")}
    if receipt.get("portion_tokens") != totals:
        raise PilotError("Reconciled replay/fresh consumption changed")
    bundle.native.validate_consumption(summary, selected, rows(directory / "training_log.jsonl"))
    checkpoint = directory / "final"
    if Path(summary["checkpoint"]).resolve() != checkpoint or Path(receipt["checkpoint"]).resolve() != checkpoint:
        raise PilotError("Run does not identify its own final checkpoint")
    if read(checkpoint / "run_config.json") != config:
        raise PilotError("Checkpoint/run configurations differ")
    checked(checkpoint / "adapter/adapter_model.safetensors", receipt["adapter_sha256"])
    c = bundle.context
    checked(c.root / c.entry["adapter_relative_path"] / "adapter_model.safetensors", PARENT_SHA)
    return receipt


def require_smoke(bundle):
    for path in sorted((bundle.root / "smokes").glob("seed42_*/completion.json")):
        validate_run(bundle, path.parent, 42, smoke=True)
        return
    raise PilotError("Run this pilot's matching 16-example smoke before full training")


def train(bundle, seed, attempt="attempt01", smoke=False):
    data, selection = load_selection(bundle)
    dest = run_directory(bundle, seed, attempt, smoke)
    if (dest / "completion.json").is_file():
        return validate_run(bundle, dest, seed, smoke)
    if dest.exists():
        raise PilotError("Partial attempt preserved; choose a new attempt and restart from the original parent")
    if not smoke:
        require_smoke(bundle)
    c, native = bundle.context, bundle.native
    # This is the first GPU operation; preflight and selection never call it.
    torch = c.legacy.gpu()
    if not torch.cuda.is_bf16_supported():
        raise PilotError("This pilot requires the recorded bf16 recipe")
    parent_config = read(c.runtime / c.entry["records_relative_path"] / "run_config.json")
    for name, version in parent_config["environment"]["packages"].items():
        if importlib.metadata.version(name) != version:
            raise PilotError("Native parent/training environment mismatch: " + name)
    p2 = c.common.load_person2()
    p2.select_model("0.5B")
    tokenizer = p2.get_tokenizer()
    if native.digest(tokenizer.backend_tokenizer.to_str()) != native.verify_proof(c)["tokenizer_sha256"]:
        raise PilotError("Training tokenizer differs from native actual-label verification")
    selected = training_rows(data, smoke)
    encoded = [p2.encode(r, tokenizer, c.contract["prompt_template"], c.contract["max_total_tokens"]) for r in selected]
    totals = reconcile_labels(selected, encoded, data["membership"])
    expected_tokens = sum(totals.values())
    inputs = bundle.root / "training_inputs" / ("smoke" if smoke else "full")
    write_rows(inputs / "train.jsonl", selected)
    for name in ("prompt_template.json", "tokenization_policy.json"):
        write_bytes(inputs / name, (c.condition / "training_inputs" / name).read_bytes())
    source = execution_source(c.train_source)
    write_bytes(inputs / "effective_train.py", source.encode())
    core = p2.core
    exec(compile(source, str(c.runtime / "continuation.py") + "::replay_c1", "exec"), core.__dict__)
    original_parent = c.root / c.entry["adapter_relative_path"]
    parent_groups = {r["problem_group_id"] for r in bundle.parent}

    def guard(args, actual_rows, current):
        if (actual_rows != selected or args.arm != METHOD or args.mode != "branch"
                or (core.MODEL, core.REVISION) != (c.entry["model"], c.entry["revision"])):
            raise PilotError("Executed manifest/model/method differs")
        # Membership and freshness were checked separately by load_selection;
        # replay is intentional and must never pass a fresh-only guard.
        reconcile_labels(actual_rows, encoded, data["membership"])

    def budget_check(args, actual_rows, actual_encoded, pcfg, state):
        actual = reconcile_labels(actual_rows, actual_encoded, data["membership"])
        if actual != totals or (not smoke and abs(sum(actual.values()) - TARGET) > TARGET * 0.005):
            raise PilotError("Actual trainer labels differ from the planned portion budgets")
        return {"actual_supervised_tokens": sum(actual.values()), "target_supervised_tokens": TARGET,
                "portion_tokens": actual, "smoke_only": smoke,
                "parent_historical_processed_tokens": state["state"]["processed_tokens"]}

    core.provenance_guard, core.parent_budget_check, core.get_tokenizer = guard, budget_check, p2.get_tokenizer
    core.fingerprint = lambda args: {
        "manifest_sha256": sha(args.data), "template_sha256": sha(args.template), "policy_sha256": sha(args.policy),
        "model": c.entry["model"], "revision": c.entry["revision"], "code_sha256": source_sha(source),
        "wrapper_sha256": sha(__file__), "reference_code_sha256": {
            name: sha(c.runtime / "vendor/capstone_person1" / name)
            for name in ("__init__.py", "person1_data_contract.py", "pipeline_controls.py", "instance_matching.py")}}
    args = SimpleNamespace(
        data=str(inputs / "train.jsonl"), template=str(inputs / "prompt_template.json"),
        policy=str(inputs / "tokenization_policy.json"), verification=None, out=str(dest), mode="branch", arm=METHOD,
        parent=str(original_parent.parent), epochs=1, seed=seed, accumulation=8, lr=1e-4,
        save_every=c.cfg["training"]["checkpoint_every_updates"], resume=None, stop_after=None, clearance=None, smoke=smoke,
        parent_metadata={"state": {"seen_groups": sorted(parent_groups), "processed_tokens": TARGET},
                         "one_pass_supervised_tokens": TARGET},
        continuation_metadata={"method": METHOD, "pilot_identity": bundle.identity,
            "selection_sha256": digest(selection), "smoke_only": smoke, "continuation_training_seed": seed,
            "shared_universe_sha256": BANK_SHA, "initial_optimizer_state": "empty", "fresh_warmup_updates": 10,
            "original_parent_adapter_sha256": PARENT_SHA, "effective_train_source_sha256": source_sha(source),
            "protocol_version": VERSION, "original_matcher_flags_resolved": False})
    core.train(args)
    summary = read(dest / "training_summary.json")
    native.validate_consumption(summary, selected, rows(dest / "training_log.jsonl"))
    if summary.get("complete") is not True or summary.get("processed_tokens") != expected_tokens:
        raise PilotError("Native trainer did not complete the selected one-pass budget")
    checked(original_parent / "adapter_model.safetensors", PARENT_SHA)
    from capstone_person4.continuation import verify_adapter_update
    checkpoint = Path(summary["checkpoint"])
    if checkpoint.resolve() != (dest / "final").resolve():
        raise PilotError("Native trainer returned an unexpected checkpoint")
    changed = verify_adapter_update(original_parent / "adapter_model.safetensors",
                                    checkpoint / "adapter/adapter_model.safetensors")
    reloaded = core.build_model(torch.bfloat16, adapter=checkpoint / "adapter", trainable=False)
    observed = core.probe_logits(reloaded, encoded[0], torch.bfloat16)
    expected = torch.load(checkpoint / "probe.pt", map_location="cpu", weights_only=True)
    if not torch.allclose(observed, expected, rtol=1e-3, atol=1e-3):
        raise PilotError("Saved adapter reload probe differs")
    receipt = {"passed": True, "version": VERSION, "identity": bundle.identity, "method": METHOD,
               "seed": seed, "smoke_only": smoke, "reload_probe_matched": True,
               "changed_adapter_tensors": changed, "parent_sha256": PARENT_SHA,
               "selection_sha256": digest(selection), "portion_tokens": totals, "actual_tokens": expected_tokens,
               "unique_examples": len(selected), "effective_train_source_sha256": source_sha(source),
               "checkpoint": str(checkpoint), "adapter_sha256": sha(checkpoint / "adapter/adapter_model.safetensors"),
               "run_config_sha256": sha(dest / "run_config.json"), "summary_sha256": sha(dest / "training_summary.json"),
               "log_sha256": sha(dest / "training_log.jsonl"), "training_seconds": summary.get("training_seconds")}
    write(dest / "resolved_config.json", bundle.cfg)
    # Validate the candidate against actual files before publishing success.
    # A failed final metadata/consumption check must leave no passed receipt.
    validate_run(bundle, dest, seed, smoke, receipt=receipt)
    write(dest / "completion.json", receipt)
    return receipt

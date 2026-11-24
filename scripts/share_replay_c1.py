"""Copy the twelve trained replay+C1 adapters into a new, checksummed shared folder."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SEEDS = (42, 137, 2026)
METHOD = "REPLAY_C1_50_50"
KNOWN = {("0.5B", 17): "9d3a8ca033368aca", ("1.5B", 17): "5a0577ec24d78630",
         ("1.5B", 123): "dd521be5e744f065"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked(path, expected):
    if not expected or sha(path) != expected:
        raise ValueError("Hash mismatch: " + str(path))


def validate_run(branch, size, parent, seed):
    run = branch / "runs" / f"seed{seed}_attempt01"
    receipt = read(run / "completion.json")
    expected = {"passed": True, "smoke_only": False, "reload_probe_matched": True,
                "method": METHOD, "seed": seed, "unique_examples": 1000}
    if any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Not a passed full replay+C1 training run: " + str(run))
    if receipt["identity"][:16] != branch.name or Path(receipt["checkpoint"]).resolve() != (run / "final").resolve():
        raise ValueError("Training identity/checkpoint path mismatch: " + str(run))
    for filename, key in (("run_config.json", "run_config_sha256"),
                          ("training_summary.json", "summary_sha256"), ("training_log.jsonl", "log_sha256")):
        checked(run / filename, receipt[key])
    checked(run / "final/adapter/adapter_model.safetensors", receipt["adapter_sha256"])
    if not (run / "final/adapter/adapter_config.json").is_file():
        raise ValueError("Adapter configuration missing: " + str(run))
    config, summary = read(run / "run_config.json"), read(run / "training_summary.json")
    if (config.get("model") != "Qwen/Qwen2.5-" + size or config.get("seed") != seed
            or config.get("arm") != METHOD or config.get("mode") != "branch"
            or Path(config["parent"]).parent.name != f"{size}_seed{parent}"
            or config.get("parent_adapter_sha256") != receipt["parent_sha256"]
            or summary.get("complete") is not True or summary.get("processed_tokens") != receipt["actual_tokens"]
            or read(run / "final/run_config.json") != config):
        raise ValueError("Model, parent, seed, training exposure or saved configuration mismatch: " + str(run))
    checked(branch / "training_inputs/full/train.jsonl", config["manifest_sha256"])
    return run, receipt, config


def select_runs(roots):
    result = []
    for size in ("0.5B", "1.5B"):
        for parent in (17, 123):
            profile = roots[size] / f"{size}_parent{parent}"
            if (size, parent) in KNOWN:
                branch = profile / KNOWN[size, parent]
            else:
                # The new parent123 identity is printed by its own preflight.
                # Resolve only a unique branch with all three passed full runs.
                candidates = []
                for candidate in profile.glob("*"):
                    passed = []
                    for seed in SEEDS:
                        receipt_path = candidate / "runs" / f"seed{seed}_attempt01/completion.json"
                        if receipt_path.is_file():
                            receipt = read(receipt_path)
                            passed.append(receipt.get("passed") is True and receipt.get("smoke_only") is False)
                    if len(passed) == 3 and all(passed):
                        candidates.append(candidate)
                if len(candidates) != 1:
                    raise ValueError(f"Expected one completed {size} parent{parent} branch; found {len(candidates)}")
                branch = candidates[0]
            for seed in SEEDS:
                run, receipt, config = validate_run(branch, size, parent, seed)
                result.append((size, parent, seed, branch, run, receipt, config))
    return result


def copy_checked(source, destination):
    if source.is_symlink():
        raise ValueError("Unexpected symlink in handoff input: " + str(source))
    expected = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy contents without importing private source permissions into the share.
    shutil.copyfile(source, destination)
    checked(destination, expected)


def export(roots, shared_root, destination, recipient="msures15", grant_acl=True):
    shared_root, destination = Path(shared_root).resolve(), Path(destination).resolve()
    if not shared_root.is_dir() or destination.parent != shared_root:
        raise ValueError("Destination must be a new direct child of the existing Megha shared folder")
    runs = select_runs(roots)  # Validate all twelve before creating a destination.
    if len(runs) != 12:
        raise ValueError("Expected exactly twelve full adapters")
    destination.mkdir(exist_ok=False)
    if grant_acl:
        subprocess.run(["setfacl", "-m", f"u:{recipient}:r-x,d:u:{recipient}:r-x", str(destination)], check=True)
    index, copied_profiles = [], set()
    for size, parent, seed, branch, run, receipt, config in runs:
        profile = destination / f"{size}_parent{parent}"
        copied_run = profile / "runs" / run.name
        names = ["completion.json", "run_config.json", "training_summary.json", "training_log.jsonl",
                 "final/run_config.json"]
        if (run / "resolved_config.json").is_file():
            names.append("resolved_config.json")
        for relative in names:
            copy_checked(run / relative, copied_run / relative)
        for source in sorted((run / "final/adapter").rglob("*")):
            if source.is_file():
                copy_checked(source, copied_run / source.relative_to(run))
        checked(copied_run / "final/adapter/adapter_model.safetensors", receipt["adapter_sha256"])
        if (size, parent) not in copied_profiles:
            for folder in ("selection", "training_inputs/full", "reports"):
                for source in sorted((branch / folder).rglob("*")):
                    if source.is_file() and source.suffix in (".json", ".jsonl", ".csv", ".py"):
                        copy_checked(source, profile / source.relative_to(branch))
            copied_profiles.add((size, parent))
        adapter = copied_run / "final/adapter"
        index.append({"model_size": size, "model": config["model"], "revision": config["revision"],
                      "parent_seed": parent, "continuation_seed": seed, "method": METHOD,
                      "adapter_path": str(adapter), "adapter_relative_path": adapter.relative_to(destination).as_posix(),
                      "checkpoint_sha256": receipt["adapter_sha256"], "train_manifest_sha256": config["manifest_sha256"],
                      "parent_adapter_sha256": receipt["parent_sha256"], "actual_supervised_tokens": receipt["actual_tokens"],
                      "replay_tokens": receipt["portion_tokens"]["replay"], "fresh_tokens": receipt["portion_tokens"]["fresh"],
                      "identity": receipt["identity"], "source_run": str(run)})
    with (destination / "run_index.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    (destination / "README.txt").write_text(
        "Twelve full REPLAY_C1_50_50 adapters: sizes 0.5B/1.5B, parents 17/123, continuation seeds 42/137/2026.\n"
        "Use run_index.csv for model, parent, training seed, adapter path and expected manifest/checkpoint hashes.\n"
        "Each copied run preserves final/adapter, final/run_config.json and the complete training records.\n"
        "Original receipt paths remain source provenance; run_index.csv maps them to the shared adapters.\n"
        "If copying this folder elsewhere, join its new root with adapter_relative_path.\n"
        "Verify SHA256SUMS.txt before evaluation. No GPU work was performed by this export.\n"
        "Use the same frozen evaluation questions, parser and runtime across all comparison arms.\n",
        encoding="utf-8")
    with (destination / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as stream:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                stream.write(sha(path) + "  " + path.relative_to(destination).as_posix() + "\n")
    if grant_acl:
        subprocess.run(["setfacl", "-R", "-m", f"u:{recipient}:rX", str(destination)], check=True)
    ready = {"complete": True, "adapters": len(index), "recipient": recipient,
             "directory": str(destination), "index": str(destination / "run_index.csv"),
             "checksum_manifest_sha256": sha(destination / "SHA256SUMS.txt"), "gpu_execution_performed": False}
    with (destination / "READY.json").open("x", encoding="utf-8") as stream:
        json.dump(ready, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root05", type=Path, default=Path("/scratch/jseth2/replay_c1_4955_pilot"))
    parser.add_argument("--root15", type=Path, default=Path("/scratch/jseth2/replay_c1_15b_4955_pilot"))
    parser.add_argument("--shared-root", type=Path, default=Path("/scratch/jseth2/megha_shared"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--recipient", default="msures15")
    args = parser.parse_args()
    try:
        result = export({"0.5B": args.root05, "1.5B": args.root15}, args.shared_root, args.destination, args.recipient)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print("STOP: " + str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

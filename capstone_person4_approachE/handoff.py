"""Additive E evaluation transfer, without rewriting completed C0/C1 handoffs."""

import csv
import io
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from capstone_person4.bindings import verified
from capstone_person4.provenance import PipelineError, digest, read_json, sha, write_bytes, write_json
from . import selection as e
from .runtime import run_directory, validate_run, with_seed, attempt_name


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def finalize(cfg, attempt="epilot01", seeds=(42, 137, 2026), c0_attempt="pilot01", c1_attempt="c1pilot01"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise PipelineError("Provide unique continuation seeds")
    index = []
    comparisons = []
    for seed in seeds:
        current = with_seed(cfg, seed)
        arm = validate_run(current, "E", run_directory(current, attempt))
        matched = {}
        for method, control_attempt in (("C0", c0_attempt), ("C1", c1_attempt)):
            directory = e.upstream_root(current) / "runs" / method / attempt_name(control_attempt)
            matched[method] = validate_run(current, method, directory)
        verified(Path(current["parent"]["adapter"]) / "adapter_model.safetensors", current["parent"]["adapter_sha256"])
        report = {"approach_complete": True, "synthetic": False, "method": "E", "evaluation_status": "pending",
                  "parent": current["parent"], "continuation_seed": seed, "recipe": current["training"],
                  "bindings": e.binding(current), "selection_directory": str(e.selection_root(current)),
                  "selection_sha256": digest(arm["selection"]),
                  "run_directory": arm["directory"], "checkpoint": arm["checkpoint"],
                  "checkpoint_sha256": arm["report"]["checkpoint_sha256"],
                  "reconciliation_sha256": sha(Path(arm["directory"]) / "token_reconciliation.json"),
                  "actual_tokens": arm["report"]["actual_tokens"],
                  "controls": {m: {"run_directory": a["directory"], "adapter_sha256": a["report"]["checkpoint_sha256"],
                                   "reconciliation_sha256": sha(Path(a["directory"]) / "token_reconciliation.json"),
                                   "actual_tokens": a["report"]["actual_tokens"]} for m, a in matched.items()},
                  "protected_question_sources": current.get("protection", {}).get("question_manifests", []),
                  "comparison_requests": ["C1_minus_E", "E_minus_C0", "E_minus_B", "E_minus_parent"],
                  "final_evaluation_authorized": False}
        handoff = e.root(current) / "handoffs" / digest(report)[:16] / "evaluation_handoff.json"
        write_json(handoff, report)
        run = Path(arm["directory"])
        index.append({"method": "E", "model_size": current["active_profile"],
                      "parent_seed": current["parent"]["training_seed"], "continuation_seed": seed,
                      "model": current["parent"]["model"], "revision": current["parent"]["revision"],
                      "adapter_path": arm["checkpoint"], "adapter_sha256": arm["report"]["checkpoint_sha256"],
                      "adapter_config_sha256": sha(Path(arm["checkpoint"]) / "adapter_config.json"),
                      "parent_adapter_sha256": current["parent"]["adapter_sha256"],
                      "training_manifest_sha256": arm["run_config"]["manifest_sha256"],
                      "actual_tokens": arm["report"]["actual_tokens"], "run_directory": str(run),
                      "selection_directory": str(e.selection_root(current)), "handoff_path": str(handoff),
                      "handoff_sha256": sha(handoff),
                      "token_reconciliation_sha256": sha(run / "token_reconciliation.json"),
                      "e_provenance_sha256": sha(run / "e_provenance.json"),
                      "resolved_config_sha256": sha(run / "resolved_config.json"),
                      "matched_C0_directory": matched["C0"]["directory"], "matched_C1_directory": matched["C1"]["directory"]})
        for method, control in matched.items():
            comparisons.append({"model_size": current["active_profile"], "parent_seed": current["parent"]["training_seed"],
                                "continuation_seed": seed, "control": method, "E_tokens": arm["report"]["actual_tokens"],
                                "control_tokens": control["report"]["actual_tokens"],
                                "absolute_token_difference": abs(arm["report"]["actual_tokens"] - control["report"]["actual_tokens"])})
    output = e.root(cfg, "select") / "handoffs" / digest(index)[:16]
    write_bytes(output / "e_run_index.csv", csv_bytes(index))
    write_bytes(output / "pair_budget_report.csv", csv_bytes(comparisons))
    return {"evaluation_status": "pending", "E_runs": len(index), "run_index": str(output / "e_run_index.csv")}


INSTRUCTIONS = """# E evaluation handoff

Evaluate the E adapters listed in e_run_index.csv on the same frozen development
GSM8K and MATH questions and evaluator protocol used for C0/C1. Gold answers are
grading inputs only. Use each E row's training_manifest_sha256 for the evaluator
guard, never the original R1000 or C1 manifest hash. Verify SHA256SUMS.txt first.

The four-parent matrix has continuation seeds 42, 137 and 2026 (12 E adapters).
A smaller package must be reported as partial; use the explicit inventory.
Reuse existing baseline results only when question hashes, checkpoint hashes,
prompt, decoding, checker and runtime contracts agree. Keep benchmarks/model
sizes separate and match parent and continuation seeds. Return predictions.jsonl,
the evaluator-emitted .config.json per run/benchmark, accuracy summaries, paired
C1-minus-E, E-minus-C0, E-minus-B and E-minus-parent comparisons, question-level
uncertainty and separate continuation-seed variation. E-minus-B requires the
previously shared matching B results. Final tests require separate authorization.

Adapter and provenance records retain their original paths. transfer_paths.csv
maps source paths to package-relative paths; do not edit original training records.
The package supplies final/run_config.json beside final/adapter for evaluator
metadata lookup. Original parent/C0/C1 weights and datasets already shared are
referenced, not duplicated. generation_provenance.json records generation cost
and cached outcomes; reuse was free for E, not free for the original experiment.
"""


def package(index_paths, destination):
    """Create-only transfer with relative checksums and a portable index."""
    rows = []
    for index_path in index_paths:
        with Path(index_path).open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    keys = [(r["model_size"], r["parent_seed"], r["continuation_seed"]) for r in rows]
    if not rows or len(keys) != len(set(keys)) or any(r.get("method") != "E" for r in rows):
        raise PipelineError("E packaging requires a nonempty, unique E run index")
    destination = Path(destination).resolve()
    if destination.exists():
        raise PipelineError("Use a new package output directory; existing transfers are preserved")
    files = {}
    mapping = []
    portable = []
    for row in rows:
        run = Path(row["run_directory"])
        handoff = read_json(row["handoff_path"])
        rec = read_json(run / "token_reconciliation.json")
        verified(row["handoff_path"], row["handoff_sha256"])
        verified(run / "e_provenance.json", row["e_provenance_sha256"])
        verified(run / "resolved_config.json", row["resolved_config_sha256"])
        provenance = read_json(run / "e_provenance.json")
        resolved = read_json(run / "resolved_config.json")
        selected_report = read_json(Path(row["selection_directory"]) / "allocation_report.json")
        immutable = [resolved["data"].get(k) for k in ("release", "inherited_release", "inherited_encoder")]
        immutable += [resolved.get("inputs", {}).get("parent_package"), str(run), row["selection_directory"]]
        if any(p and (destination == Path(p).resolve() or Path(p).resolve() in destination.parents) for p in immutable):
            raise PipelineError("Package output must be outside immutable inputs and completed runs/selections")
        manifest = Path(row["selection_directory"]) / "manifest.jsonl"
        if (handoff.get("method") != "E" or handoff.get("synthetic") is not False
                or handoff.get("checkpoint_sha256") != row["adapter_sha256"]
                or rec.get("method") != "E" or rec.get("valid_controlled_run") is not True
                or rec.get("training_smoke_only") is not False or rec.get("parent_unchanged") is not True
                or rec.get("parent_fingerprint") != row["parent_adapter_sha256"]
                or rec.get("continuation_seed") != int(row["continuation_seed"])
                or rec.get("actual_tokens") != int(row["actual_tokens"])
                or provenance.get("reconciliation_sha256") != sha(run / "token_reconciliation.json")
                or provenance.get("stage_fingerprint") != rec.get("stage_fingerprint")
                or handoff.get("selection_sha256") != rec.get("selection_sha256")
                or digest(selected_report) != rec.get("selection_sha256")
                or provenance.get("bindings") != handoff.get("bindings")
                or provenance.get("bindings") != selected_report.get("bindings")
                or resolved["active_profile"] != row["model_size"]
                or str(resolved["parent"]["training_seed"]) != row["parent_seed"]
                or resolved["parent"]["adapter_sha256"] != row["parent_adapter_sha256"]
                or resolved["training"]["seed"] != int(row["continuation_seed"])):
            raise PipelineError("Cannot package an unreconciled or mismatched E run")
        verified(run / "token_reconciliation.json", row["token_reconciliation_sha256"])
        for name, key in (("run_config.json", "run_config_sha256"), ("training_summary.json", "training_summary_sha256")):
            verified(run / name, rec[key])
        verified(Path(row["adapter_path"]) / "adapter_model.safetensors", row["adapter_sha256"])
        verified(Path(row["adapter_path"]) / "adapter_config.json", row["adapter_config_sha256"])
        verified(manifest, row["training_manifest_sha256"])
        if selected_report.get("manifest_sha256") != row["training_manifest_sha256"]:
            raise PipelineError("E package training manifest differs from its selected manifest")
        for name, key in (("selection_table.csv", "selection_table_sha256"), ("ordered_ids.json", "ordered_ids_sha256")):
            verified(Path(row["selection_directory"]) / name, selected_report[key])
        metadata = read_json(run / "run_config.json")
        if (metadata.get("model") != row["model"] or metadata.get("revision") != row["revision"]
                or metadata.get("arm") != "E" or metadata.get("seed") != int(row["continuation_seed"])
                or metadata.get("manifest_sha256") != row["training_manifest_sha256"]):
            raise PipelineError("E package evaluator metadata does not match the run index")
        prefix = f"E/{row['model_size']}/parent{row['parent_seed']}/seed{row['continuation_seed']}"

        def add(source, relative):
            path = Path(source)
            name = prefix + "/" + relative
            files[name] = path.read_bytes()
            mapping.append({"original_path": str(path.resolve()), "package_path": name})

        for name in ("run_config.json", "resolved_config.json", "training_summary.json", "token_reconciliation.json", "e_provenance.json"):
            add(run / name, name)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            add(Path(row["adapter_path"]) / name, "final/adapter/" + name)
        final_config = run / "final/run_config.json"
        if final_config.is_file():
            metadata = read_json(final_config)
            original = read_json(run / "run_config.json")
            for key in ("manifest_sha256", "model", "revision", "arm", "seed", "parent_adapter_sha256"):
                if metadata.get(key) != original.get(key):
                    raise PipelineError(f"Evaluator checkpoint metadata differs from training: {key}")
        add(final_config if final_config.is_file() else run / "run_config.json", "final/run_config.json")
        for path in sorted(run.glob("*")):
            if path.is_file() and ("loss" in path.name or path.suffix in (".log", ".out", ".err")):
                add(path, "logs/" + path.name)
        for path in sorted(Path(row["selection_directory"]).iterdir()):
            if path.is_file() and not path.name.endswith(".partial"):
                add(path, "selection/" + path.name)
        add(row["handoff_path"], "evaluation_handoff.json")
        files[prefix + "/generation_provenance.json"] = (json.dumps(provenance["generation_coverage"], sort_keys=True, indent=2) + "\n").encode()
        exported = dict(row)
        exported.update(adapter_path=prefix + "/final/adapter", run_directory=prefix,
                        selection_directory=prefix + "/selection", handoff_path=prefix + "/evaluation_handoff.json")
        portable.append(exported)
    files["e_run_index.csv"] = csv_bytes(portable)
    files["transfer_paths.csv"] = csv_bytes(mapping)
    files["MEGHA_E_EVALUATION_HANDOFF.md"] = INSTRUCTIONS.encode()
    inventory = {"E_runs": len(rows), "complete_twelve_run_matrix": set(keys) == {
        (size, str(parent), str(seed)) for size in ("0.5B", "1.5B") for parent in (17, 123) for seed in (42, 137, 2026)},
        "e_code": e.code_hashes(), "reused_code": e.source_hashes(), "evaluation_status": "pending"}
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
                              cwd=repo, capture_output=True, text=True, check=False)
    inventory["repository_revision"] = revision.stdout.strip() if revision.returncode == 0 else "unavailable; see source hashes"
    for relative, expected in inventory["e_code"].items():
        verified(repo / relative, expected)
        files["source/" + relative] = (repo / relative).read_bytes()
    files["inventory.json"] = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    sums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items()))
    files["SHA256SUMS.txt"] = sums.encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    write_bytes(destination / "E_evaluation.zip", buffer.getvalue())
    write_bytes(destination / "E_evaluation.zip.sha256", f"{sha(destination / 'E_evaluation.zip')}  E_evaluation.zip\n".encode())
    return {**inventory, "archive": str(destination / "E_evaluation.zip")}

"""Frozen group-disjoint candidates; the model never chooses its scoring universe."""

from collections import Counter, defaultdict
from pathlib import Path

from .allocation import quotas
from .audit import gate, parent_exposure
from .config import artifact_root, stage_fingerprint
from .person2_adapter import Person2Adapter
from .provenance import PipelineError, digest, read_json, read_rows, sha, source_hashes, unique_rows, write_json, write_rows
from .selection import stratum


def eligible_candidates(original, seen, overlap, max_total_tokens):
    exclusions, groups = [], defaultdict(list)
    for row in original:
        reason = "parent_seen_group" if row["problem_group_id"] in seen else "development_or_final_group" if row["problem_group_id"] in overlap else None
        if row["total_tokens"] > max_total_tokens:
            reason = reason or "whole_example_context_limit"
        if reason:
            exclusions.append({"row_id": row["row_id"], "problem_group_id": row["problem_group_id"], "reason": reason})
        else:
            groups[stratum(row)].append(row)
    return groups, exclusions


def prepare(cfg):
    if cfg.get("protection"):
        return prepare_v4(cfg)
    evidence = gate(cfg, "prepare")
    parent_rows, _ = parent_exposure(cfg)
    seen = {r["problem_group_id"] for r in parent_rows}
    overlap = set(evidence["overlap"]["excluded_group_ids"])
    for role in ("development", "final"):
        overlap.update(r["problem_group_id"] for r in read_rows(cfg["evaluation"][role + "_manifest"]) if r.get("problem_group_id"))
    original = read_rows(cfg["data"]["candidate_manifest"])
    groups, exclusions = eligible_candidates(original, seen, overlap, cfg["data"]["max_total_tokens"])
    count = cfg["data"]["candidate_count"]
    if type(count) is not int or count <= 0 or not groups:
        raise PipelineError("Positive candidate_count and an eligible fresh pool required")
    actual = min(count, sum(map(len, groups.values())))
    capacities = {k: len(v) for k, v in groups.items()}
    q = quotas(actual, capacities, capacities)
    seed = cfg["data"]["seed"]
    selected = []
    for key in sorted(groups):
        for row in sorted(groups[key], key=lambda r: digest([seed, r["row_id"]]))[:q[key]]:
            selected.append({**row, "canonical_text_hash": digest([row["problem"], row["completion"]]),
                             "split": "candidate", "synthetic": False})
    unique_rows(selected)
    # Real tokenization verifies all frozen candidates, independent of score utility.
    adapter = Person2Adapter(cfg)
    tokenizer = adapter.tokenizer()
    for row in selected:
        adapter.encode(row, tokenizer)
    root = artifact_root(cfg, "prepare") / "prepared"
    write_rows(root / "candidates.jsonl", selected)
    write_rows(root / "exclusions.jsonl", exclusions)
    report = {"synthetic": False, "config_fingerprint": cfg["config_fingerprint"], "code": source_hashes(),
              "stage_fingerprint": stage_fingerprint(cfg, "prepare"),
              "candidate_sha256": sha(root / "candidates.jsonl"), "tokenizer_fingerprint": adapter.tokenizer_fingerprint,
              "requested_rows": count, "actual_rows": actual, "shortfall": count - actual,
              "strata": dict(Counter(stratum(r) for r in selected)), "evidence": evidence,
              "parent_groups": len(seen), "excluded_rows": len(exclusions), "ids_hash": digest([r["row_id"] for r in selected]),
              "limitations": "Exact/near controls limited to the verified release and supplied overlap report"}
    write_json(root / "preparation.json", report)
    return report


def load_prepared(cfg):
    root = artifact_root(cfg, "prepare") / "prepared"
    report = read_json(root / "preparation.json")
    if (report.get("synthetic") is not False or report.get("stage_fingerprint") != stage_fingerprint(cfg, "prepare")
            or report.get("candidate_sha256") != sha(root / "candidates.jsonl")):
        raise PipelineError("Prepared candidates are synthetic, stale, or belong to another configuration/code version")
    rows = read_rows(root / "candidates.jsonl")
    unique_rows(rows)
    if report["actual_rows"] != len(rows):
        raise PipelineError("Prepared candidate coverage mismatch")
    return rows, report


def freeze_stratified(rows, parent_rows, count, seed):
    """Use parent row proportions and fail on deficits instead of resizing the pool."""
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    proportions = dict(Counter(stratum(r) for r in parent_rows))
    if type(count) is not int or count <= 0:
        raise PipelineError("Positive integer candidate_count required")
    # Compute desired quotas independent of availability, then report exact deficits.
    wanted = quotas(count, proportions, {k: count for k in proportions})
    deficits = {k: wanted[k] - len(groups[k]) for k in wanted if len(groups[k]) < wanted[k]}
    if deficits:
        raise PipelineError(f"Insufficient eligible candidate strata: {deficits}; declare a smaller common universe explicitly")
    selected = []
    for key in sorted(wanted):
        selected.extend(sorted(groups[key], key=lambda r: (digest([seed, r["row_id"]]), r["row_id"]))[:wanted[key]])
    unique_rows(selected)
    if len({r["duplicate_cluster_id"] for r in selected}) != len(selected):
        raise PipelineError("Normalized duplicate cluster survived candidate freezing")
    return selected, wanted


def prepare_v4(cfg):
    from .protection import load_protection, protect, write_csv
    from .bindings import write_bindings
    evidence = gate(cfg, "prepare")
    if (artifact_root(cfg, "prepare") / "prepared/preparation.json").exists():
        # Valid prepared candidates are reused unchanged; the report keeps its original
        # config snapshot even when settings of later stages (e.g. scoring) change.
        return load_prepared(cfg)[1]
    protection_path = artifact_root(cfg, "prepare") / "protection/summary.json"
    if not protection_path.exists():
        protect(cfg)
    eligible, protection = load_protection(cfg)
    parent_rows, parent = parent_exposure(cfg)
    maximum = cfg["data"]["max_total_tokens"]
    context_exclusions = [{"row_id": r["row_id"], "reason": "whole_example_context_limit"}
                          for r in eligible if r["total_tokens"] > maximum]
    eligible = [r for r in eligible if r["total_tokens"] <= maximum]
    selected, wanted = freeze_stratified(eligible, parent_rows, cfg["data"]["candidate_count"], cfg["data"]["seed"])
    selected = [{**r, "canonical_text_hash": digest([r["problem"], r["completion"]]),
                 "split": "candidate"} for r in selected]
    adapter = Person2Adapter(cfg)
    tokenizer = adapter.tokenizer()
    verification = []
    for role, rows in (("parent", parent_rows), ("candidate", selected)):
        for row in rows:
            encoded = adapter.encode(row, tokenizer)
            counts = encoded["counts"]
            for field in ("prompt_tokens", "completion_tokens", "supervised_tokens", "total_tokens"):
                if counts[field] != row[field]:
                    raise PipelineError(f"Actual {field} differs: {row['row_id']}")
            if encoded["labels"][-1] != tokenizer.eos_token_id or counts["total_tokens"] > maximum:
                raise PipelineError("EOS or whole-example context verification failed")
            verification.append({"role": role, "row_id": row["row_id"], **{k: counts[k] for k in
                                 ("prompt_tokens", "completion_tokens", "supervised_tokens", "total_tokens")},
                                 "encoded_sha256": digest([encoded["input_ids"], encoded["labels"]])})
    anchor = sum(r["supervised_tokens"] for r in verification if r["role"] == "parent")
    if anchor != parent["processed_tokens"] or anchor != cfg["allocation"].get("expected_replay_tokens", anchor):
        raise PipelineError("Re-encoded replay anchor differs from saved parent evidence/configuration")
    root = artifact_root(cfg, "prepare") / "prepared"
    write_rows(root / "candidates.jsonl", selected)
    write_rows(root / "eligible.jsonl", eligible)
    write_rows(root / "context_exclusions.jsonl", context_exclusions)
    write_json(root / "ordered_ids.json", [r["row_id"] for r in selected])
    write_csv(root / "token_verification.csv", verification, list(verification[0]))
    report = {"synthetic": False, "stage": "preparation", "config_fingerprint": cfg["config_fingerprint"],
              "stage_fingerprint": stage_fingerprint(cfg, "prepare"), "candidate_sha256": sha(root / "candidates.jsonl"),
              "tokenizer_fingerprint": adapter.tokenizer_fingerprint, "requested_rows": len(selected), "actual_rows": len(selected),
              "shortfall": 0, "strata": wanted, "parent_replay_tokens": anchor,
              "ids_hash": digest([r["row_id"] for r in selected]), "token_verification_sha256": sha(root / "token_verification.csv"),
              "protection_summary_sha256": sha(protection_path), "protection_counts": protection["counts"],
              "evidence": evidence, "code": source_hashes(), "evaluation_status": "pending"}
    write_bindings(cfg, evidence["release"], parent)
    write_json(root / "preparation.json", report)
    return report

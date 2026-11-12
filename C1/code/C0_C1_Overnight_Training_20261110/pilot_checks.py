"""Unchanged population, selection and consumption checks from the completed pilot."""
from collections import Counter, defaultdict
import hashlib
import math
TARGET = 318845
def unique(records):
    if not records:
        raise ValueError("Empty records")
    for key in ("row_id", "problem_group_id"):
        values = [r[key] for r in records]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f"Invalid/duplicate {key}")

def cell(row):
    return row["dataset_id"], row["length_bin"]

def bounds(target, numerator, denominator):
    return ((target * (denominator - numerator) + denominator - 1) // denominator,
            target * (denominator + numerator) // denominator)

def validate_population(original, retained, screen, decisions):
    unique(original)
    unique(retained)
    by_id = {r["row_id"]: r for r in original}
    held_ids = set(screen["held_out_ids"])
    if len(held_ids) != screen["held_out_rows"] or len(held_ids) != 1454:
        raise ValueError("Screen quarantine count differs")
    if len(decisions) != len(held_ids) or {d["row_id"] for d in decisions} != held_ids:
        raise ValueError("Quarantine decisions do not cover the declared IDs")
    for decision in decisions:
        row = by_id.get(decision["row_id"])
        if row is None or hashlib.sha256(row["problem"].encode()).hexdigest() != decision["question_sha256"]:
            raise ValueError("Quarantined question content changed")
        if decision.get("completion_sha256") and hashlib.sha256(row["completion"].encode()).hexdigest() != decision["completion_sha256"]:
            raise ValueError("Previously reviewed reference content changed")
    bad_groups = {by_id[rid]["problem_group_id"] for rid in held_ids}
    wanted = [r for r in original if r["problem_group_id"] not in bad_groups]
    if (retained != wanted or len(retained) != screen["retained_rows"]
            or len(retained) != 4955 or len(original) != 6409):
        raise ValueError("Retained bank differs from complete group-level quarantine")
    return True

def validate_selection(selected, bank, parent):
    unique(selected)
    by_id = {r["row_id"]: r for r in bank}
    if len(selected) != 1000 or any(by_id.get(r["row_id"]) != r for r in selected):
        raise ValueError("Selection membership/count/content differs")
    for key in ("row_id", "problem_group_id"):
        if {r[key] for r in selected} & {r[key] for r in parent}:
            raise ValueError(f"Selection intersects parent {key}")
    required = Counter(cell(r) for r in parent)
    if len(required) != 20 or set(required.values()) != {50} or Counter(cell(r) for r in selected) != required:
        raise ValueError("Selection must retain exactly 50 rows per source/length stratum")
    parent_tokens, selected_tokens = defaultdict(int), defaultdict(int)
    for r in parent:
        parent_tokens[cell(r)] += r["supervised_tokens"]
    for r in selected:
        selected_tokens[cell(r)] += r["supervised_tokens"]
    if sum(parent_tokens.values()) != TARGET:
        raise ValueError("Parent token anchor differs")
    for key, target in parent_tokens.items():
        low, high = bounds(target, 5, 100)
        if not low <= selected_tokens[key] <= high:
            raise ValueError("Selection stratum tokens exceed 5 percent")
    low, high = bounds(TARGET, 5, 1000)
    if not low <= sum(selected_tokens.values()) <= high:
        raise ValueError("Selection global tokens exceed 0.5 percent")
    def order(row):
        h = hashlib.sha256(("42|training-order|" + row["row_id"]).encode()).digest()
        return int.from_bytes(h[:4], "big"), row["row_id"]
    if selected != sorted(selected, key=order):
        raise ValueError("Selection training order differs")
    return sum(selected_tokens.values())

def validate_consumption(summary, selected, logs):
    tokens = sum(r["supervised_tokens"] for r in selected)
    updates = (len(selected) + 7) // 8
    if (summary.get("complete") is not True or summary.get("processed_tokens") != tokens
            or summary.get("unique_examples") != len(selected)
            or summary.get("unique_supervised_tokens") != tokens
            or summary.get("optimizer_updates") != updates
            or summary.get("parent_optimizer_updates") != 0
            or set(summary.get("seen_ids", [])) != {r["row_id"] for r in selected}
            or set(summary.get("seen_groups", [])) != {r["problem_group_id"] for r in selected}):
        raise ValueError("Training consumption, IDs, updates or optimizer isolation did not reconcile")
    if len(logs) != updates or sum(r["target_tokens"] for r in logs) != tokens:
        raise ValueError("Training log does not cover the complete token budget")
    cumulative = 0
    for update, record in enumerate(logs, 1):
        cumulative += record["target_tokens"]
        expected_rate = 1e-4 * min(update / 10.0, 1.0)
        if (record.get("optimizer_update") != update or record.get("cumulative_processed_tokens") != cumulative
                or not math.isclose(record["lr"], expected_rate, rel_tol=1e-12, abs_tol=1e-15)
                or not math.isfinite(record["loss"]) or not math.isfinite(record["grad_norm"])):
            raise ValueError("Training log update, token, warmup or finite-loss check failed")
    return tokens

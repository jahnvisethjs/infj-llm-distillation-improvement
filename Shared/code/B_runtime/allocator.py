"""CPU-only B selection and a shared priority/coverage/token allocator.

No model imports, downloads, scoring, or training occur on import or execution.
Other fresh selectors supply their own ascending priorities to allocate().
"""
import csv
import hashlib
import io
import json
import math
import platform
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

VERSION = "shared-priority-allocator-v1-fractional-targets"


class AllocationError(ValueError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def stable_key(domain, seed, identity):
    return sha(json.dumps([domain, seed, identity], separators=(",", ":")).encode())


def cell(row):
    return str(row["dataset_id"]), int(row["length_bin"])


def jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def csv_bytes(rows):
    if not rows:
        raise ValueError("Cannot export an empty audit table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def load_rows(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_rows(rows, max_total_tokens):
    ids, groups = set(), set()
    for row in rows:
        for name in ("row_id", "problem_group_id", "dataset_id", "problem", "completion", "final_answer"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Missing/empty {name}")
        if row["row_id"] in ids or row["problem_group_id"] in groups:
            raise ValueError("Repeated row ID or problem-group ID")
        ids.add(row["row_id"])
        groups.add(row["problem_group_id"])
        for name in ("prompt_tokens", "completion_tokens", "supervised_tokens", "total_tokens", "eos_tokens", "length_bin"):
            if type(row.get(name)) is not int or row[name] < 0:
                raise ValueError(f"Invalid {name}: {row['row_id']}")
        if row["prompt_tokens"] <= 0 or row["completion_tokens"] <= 0:
            raise ValueError("Empty recorded prompt/completion")
        if row["eos_tokens"] != 1 or row["supervised_tokens"] != row["completion_tokens"] + 1:
            raise ValueError("Stored completion/EOS accounting mismatch")
        if row["total_tokens"] != row["prompt_tokens"] + row["supervised_tokens"]:
            raise ValueError("Stored total-token accounting mismatch")
        if row["total_tokens"] > max_total_tokens or row.get("boundary_crossing_token") is True:
            raise ValueError("Overlength or boundary-crossing row; no silent filtering/truncation")
    if not rows:
        raise ValueError("Empty input rows")


def apportion(weights, count, capacities, quota_seed):
    """Exact largest-remainder allocation; capacity failure never changes weights."""
    if type(count) is not int or count < 1 or any(w <= 0 for w in weights.values()):
        raise AllocationError("Invalid count or stratum weights")
    denominator = sum(weights.values())
    ideals = {h: Fraction(count * weight, denominator) for h, weight in weights.items()}
    quotas = {h: math.floor(value) for h, value in ideals.items()}
    order = sorted(weights, key=lambda h: (
        -(ideals[h] - quotas[h]), stable_key("quota-tie-v1", quota_seed, list(h)), h
    ))
    for h in order[:count - sum(quotas.values())]:
        quotas[h] += 1
    if any(quotas[h] <= 0 for h in weights):
        raise AllocationError("Requested count cannot cover every source/length stratum")
    if any(quotas[h] > capacities.get(h, 0) for h in weights):
        raise AllocationError("Fixed-proportion quotas exceed capacity; enlarge the shared pool or lower the budget")
    return quotas


def random_priorities(rows, seed):
    # Full 256-bit hash priorities; lower keys have higher priority.
    return {row["row_id"]: int(stable_key("selector-tie-v1", seed, row["row_id"]), 16) for row in rows}


def repair_once(selected_ids, by_cell, priorities, current_tokens, target, window):
    """Best improving within-stratum swap, minimizing lost priority first."""
    best = None
    old_distance = abs(current_tokens - target)
    for h in sorted(by_cell):
        ordered = by_cell[h]
        chosen = [row for row in ordered if row["row_id"] in selected_ids]
        unchosen = [row for row in ordered if row["row_id"] not in selected_ids]
        for out in chosen[-window:]:
            for incoming in unchosen[:window]:
                total = current_tokens + incoming["supervised_tokens"] - out["supervised_tokens"]
                distance = abs(total - target)
                if distance >= old_distance:
                    continue
                cost = priorities[incoming["row_id"]] - priorities[out["row_id"]]
                key = (cost, distance, incoming["row_id"], out["row_id"])
                if best is None or key < best[0]:
                    best = (key, incoming, out, total)
    return best


def allocate(rows, priorities, parent_counts, target, tolerance="0.005", quota_seed=1729,
             tie_seed=42, row_count_adjustment_fraction="0.05", swap_window=24, max_swaps=8):
    """Use common counts, whole-row matching, and deterministic priority ties.

    C0/E/C1 use this exact function with model-specific ascending priorities.
    It is a bounded priority-preserving heuristic, not a uniform sampler or a
    proof of global constrained-utility optimality.
    """
    if type(target) not in (int, Fraction) or target <= 0:
        raise ValueError("Target must be a positive integer or exact Fraction")
    tol = Fraction(str(tolerance))
    adjustment = Fraction(str(row_count_adjustment_fraction))
    if not 0 <= tol < 1 or not 0 <= adjustment <= 1:
        raise ValueError("Invalid tolerance/count-adjustment fraction")
    if type(swap_window) is not int or type(max_swaps) is not int or swap_window < 1 or max_swaps < 0:
        raise ValueError("Invalid bounded-swap configuration")
    ids = {row["row_id"] for row in rows}
    if len(ids) != len(rows) or len({row["problem_group_id"] for row in rows}) != len(rows):
        raise ValueError("Candidate IDs/groups must be unique")
    if set(priorities) != ids:
        raise ValueError("Priority coverage must equal the whole common candidate universe")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in priorities.values()):
        raise ValueError("Missing/nonfinite numeric priority")
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[cell(row)].append(row)
    if set(by_cell) != set(parent_counts):
        raise ValueError("Candidate and parent source/length strata differ")
    for ordered in by_cell.values():
        ordered.sort(key=lambda row: (priorities[row["row_id"]], stable_key("selector-tie-v1", tie_seed, row["row_id"]), row["row_id"]))
    capacities = {h: len(ordered) for h, ordered in by_cell.items()}
    lower = math.ceil(target * (1 - tol))
    upper = math.floor(target * (1 + tol))
    mean = sum(Fraction(parent_counts[h], sum(parent_counts.values())) *
               Fraction(sum(row["supervised_tokens"] for row in by_cell[h]), len(by_cell[h]))
               for h in parent_counts)
    estimate = target / mean
    # Round exact fractions half-up, so the result is independent of float math.
    n0 = max(len(by_cell), min(len(rows), math.floor(estimate + Fraction(1, 2))))
    radius = max(len(by_cell), math.ceil(n0 * adjustment))
    counts = sorted(range(max(len(by_cell), n0 - radius), min(len(rows), n0 + radius) + 1),
                    key=lambda n: (abs(n - n0), n))
    attempts, nearest = [], None
    for count in counts:
        try:
            quotas = apportion(parent_counts, count, capacities, quota_seed)
        except AllocationError as error:
            attempts.append({"rows": count, "status": "quota_capacity_failure", "reason": str(error)})
            continue
        token_sorted = {h: sorted(row["supervised_tokens"] for row in ordered) for h, ordered in by_cell.items()}
        minimum = sum(sum(token_sorted[h][:quotas[h]]) for h in quotas)
        maximum = sum(sum(token_sorted[h][-quotas[h]:]) for h in quotas)
        selected_ids = {row["row_id"] for h, ordered in by_cell.items() for row in ordered[:quotas[h]]}
        chosen = [row for row in rows if row["row_id"] in selected_ids]
        total = sum(row["supervised_tokens"] for row in chosen)
        start = total
        swaps = []
        bounds_possible = minimum <= upper and maximum >= lower
        if bounds_possible:
            for _ in range(max_swaps):
                if lower <= total <= upper:
                    break
                best = repair_once(selected_ids, by_cell, priorities, total, target, swap_window)
                if best is None:
                    break
                _, incoming, out, new_total = best
                selected_ids.remove(out["row_id"])
                selected_ids.add(incoming["row_id"])
                swaps.append({"out_row_id": out["row_id"], "in_row_id": incoming["row_id"],
                              "token_delta": new_total - total})
                total = new_total
        attempt = {"rows": count, "initial_tokens": start, "actual_tokens": total,
                   "minimum_possible_tokens_for_quotas": minimum, "maximum_possible_tokens_for_quotas": maximum,
                   "token_bounds_intersect_target": bounds_possible, "swaps": swaps,
                   "status": "matched" if lower <= total <= upper else "not_matched"}
        attempts.append(attempt)
        if nearest is None or abs(total - target) < abs(nearest["actual_tokens"] - target):
            nearest = attempt
        if lower <= total <= upper:
            chosen = sorted((row for row in rows if row["row_id"] in selected_ids), key=lambda row: row["row_id"])
            return chosen, {"allocator_version": VERSION, "estimated_rows": n0,
                            "row_count_search_range": [min(counts), max(counts)],
                            "actual_rows": len(chosen), "actual_tokens": total,
                            "target_tokens": target.numerator if isinstance(target, Fraction) and target.denominator==1 else float(target) if isinstance(target, Fraction) else target,
                            "allowed_integer_token_range": [lower, upper],
                            "relative_token_difference": float(Fraction(total - target, target)),
                            "count_adjustment_from_estimate": count - n0,
                            "swap_count": len(swaps), "attempts": attempts,
                            "tolerance_relaxed": False, "coverage_relaxed": False}
    report = {"status": "no_match_found_in_declared_search", "target_tokens": float(target) if isinstance(target, Fraction) else target,
              "allowed_integer_token_range": [lower, upper], "nearest_attempt": nearest,
              "attempts": attempts, "tolerance_relaxed": False,
              "infeasibility_proven": bool(attempts) and all(
                  a["status"] == "quota_capacity_failure" or not a.get("token_bounds_intersect_target", False) for a in attempts),
              "proof_scope": "Only the declared row-count range and fixed source/length quotas.",
              "remedy": "Inspect nearest attempt and capacity bounds. Expand the shared bank or explicitly version the common search/budget policy for every fresh selector. Do not silently widen tolerance."}
    raise AllocationError("No budget-matched subset found; tolerance was not widened", report)


def safe_outputs(destination, outputs):
    destination = Path(destination)
    for filename, content in outputs.items():
        path = destination / filename
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Different output already exists: {path}; choose a new version")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        path = destination / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)


def run_b(root, config_path):
    root, config_path = Path(root), Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pool_dir = root / "outputs" / "common_pool_review_v1"
    pool_path = pool_dir / "candidate_pool.jsonl"
    parent_path = root / "data" / "train_n1000.jsonl"
    contract_path = root / "inputs" / "formatting_contract.json"
    edges_path = root / "inputs" / "length_bin_edges.json"
    pool_summary = json.loads((pool_dir / "pool_summary.json").read_text())
    for path, expected in [(pool_path, config["screened_pool_sha256"]),
                           (parent_path, config["parent_manifest_sha256"]),
                           (contract_path, config["formatting_contract_sha256"]),
                           (edges_path, config["length_bin_edges_sha256"])]:
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"Input hash mismatch: {path.name}; do not substitute another release")
    if pool_summary["pool_sha256"] != config["screened_pool_sha256"] or not pool_summary["review_queue_complete"]:
        raise ValueError("Screened-pool report is missing or incompatible")
    contract = json.loads(contract_path.read_text())
    parent, pool = load_rows(parent_path), load_rows(pool_path)
    validate_rows(parent, contract["max_total_tokens"])
    validate_rows(pool, contract["max_total_tokens"])
    if len(parent) != 1000 or len(pool) != 9909:
        raise ValueError("Unexpected verified input counts")
    target = sum(row["supervised_tokens"] for row in parent)
    if target != config["target_supervised_tokens"]:
        raise ValueError("Configured replay anchor differs from actual parent manifest")
    for field in ["row_id", "problem_group_id"]:
        if {row[field] for row in parent} & {row[field] for row in pool}:
            raise ValueError(f"Parent/pool overlap in {field}")
    for row in parent + pool:
        if row["tokenizer_revision"] != contract["tokenizer_revision"]:
            raise ValueError("Row tokenizer revision differs from formatting contract")
        if row["prompt_text"] != contract["prompt_template"].replace("{problem}", row["problem"]):
            raise ValueError("Stored prompt text differs from formatting contract")
    weights = Counter(cell(row) for row in parent)
    pool_cells = defaultdict(list)
    for row in pool:
        pool_cells[cell(row)].append(row)
    if set(pool_cells) != set(weights):
        raise ValueError("Screened bank lacks a reference stratum")
    universe_quotas = apportion(weights, config["candidate_universe_size"],
                               {h: len(rows) for h, rows in pool_cells.items()}, config["quota_seed"])
    universe = sorted((row for h, rows in sorted(pool_cells.items())
                       for row in sorted(rows, key=lambda r: (stable_key("common-universe-v1", config["universe_seed"], r["row_id"]), r["row_id"]))[:universe_quotas[h]]),
                      key=lambda row: row["row_id"])
    universe_bytes = jsonl_bytes(universe)
    universe_sha = sha(universe_bytes)
    priorities = random_priorities(universe, config["selection_seed"])
    output_dir = root / "outputs" / config["output_version"]
    try:
        selected, allocation = allocate(
            universe, priorities, weights, target, config["relative_token_tolerance"],
            config["quota_seed"], config["selection_seed"], config["row_count_adjustment_fraction"],
            config["swap_window"], config["max_priority_swaps"])
    except AllocationError as error:
        safe_outputs(output_dir, {"allocation_failure.json": json_bytes(error.report)})
        raise
    ordered = sorted(selected, key=lambda row: (stable_key("training-order-v1", config["training_order_seed"], row["row_id"]), row["row_id"]))
    training_bytes = jsonl_bytes(ordered)
    selected_ids = {row["row_id"] for row in selected}
    rank = {}
    for h in weights:
        group = [row for row in universe if cell(row) == h]
        for i, row in enumerate(sorted(group, key=lambda r: (priorities[r["row_id"]], r["row_id"])), 1):
            rank[row["row_id"]] = i
    audit = [{"row_id": row["row_id"], "problem_group_id": row["problem_group_id"],
              "source": row["dataset_id"], "length_bin": row["length_bin"],
              "prompt_tokens": row["prompt_tokens"], "supervised_tokens": row["supervised_tokens"],
              "total_tokens": row["total_tokens"],
              "random_priority_hex": f"{priorities[row['row_id']]:064x}",
              "within_stratum_priority_rank": rank[row["row_id"]],
              "selected": row["row_id"] in selected_ids} for row in universe]
    manifest = [{"training_order": index, **next(record for record in audit if record["row_id"] == row["row_id"])}
                for index, row in enumerate(ordered)]
    counts = Counter(cell(row) for row in selected)
    expected_quotas = apportion(weights, len(selected), {h: universe_quotas[h] for h in weights}, config["quota_seed"])
    coverage = []
    for h in sorted(weights):
        parent_tokens = sum(row["supervised_tokens"] for row in parent if cell(row) == h)
        selected_tokens = sum(row["supervised_tokens"] for row in selected if cell(row) == h)
        coverage.append({"source": h[0], "length_bin": h[1], "parent_rows": weights[h],
                         "screened_available_rows": len(pool_cells[h]), "shared_universe_rows": universe_quotas[h],
                         "selected_quota": expected_quotas[h], "selected_rows": counts[h],
                         "quota_deviation_rows": counts[h] - expected_quotas[h],
                         "parent_supervised_tokens": parent_tokens, "B_supervised_tokens": selected_tokens})
    if any(row["quota_deviation_rows"] for row in coverage):
        raise AssertionError("Allocator violated a source/length quota")
    sources = {}
    for source in sorted({row["dataset_id"] for row in parent}):
        subset = [row for row in selected if row["dataset_id"] == source]
        sources[source] = {"rows": len(subset), "supervised_tokens": sum(row["supervised_tokens"] for row in subset)}
        sources[source]["supervised_token_share"] = sources[source]["supervised_tokens"] / allocation["actual_tokens"]
    summary = {"method": "B_fresh_stratified_random", "allocator_version": VERSION,
               "screened_pool_sha256": config["screened_pool_sha256"],
               "common_universe_sha256": universe_sha, "common_universe_rows": len(universe),
               "common_universe_source_rows": dict(Counter(row["dataset_id"] for row in universe)),
               "selection_seed": config["selection_seed"], "universe_seed": config["universe_seed"],
               "training_order_seed": config["training_order_seed"],
               "selected_rows": len(selected), "unique_selected_groups": len({row["problem_group_id"] for row in selected}),
               "selected_supervised_tokens": allocation["actual_tokens"],
               "target_supervised_tokens": target, "difference_tokens": allocation["actual_tokens"] - target,
               "difference_percent": 100 * allocation["relative_token_difference"],
               "allowed_integer_token_range": allocation["allowed_integer_token_range"],
               "source_composition": sources, "selected_prompt_tokens": sum(row["prompt_tokens"] for row in selected),
               "selected_sequence_tokens": sum(row["total_tokens"] for row in selected),
               "source_length_groups": len(coverage), "quota_deviations": 0,
               "parent_row_id_overlap": 0, "parent_recorded_group_overlap": 0,
               "training_jsonl_sha256": sha(training_bytes),
               "parent_manifest_sha256": sha(parent_path.read_bytes()),
               "formatting_contract_sha256": sha(contract_path.read_bytes()),
               "length_bin_edges_sha256": sha(edges_path.read_bytes()),
               "selector_code_sha256": sha(Path(__file__).read_bytes()),
               "configuration_sha256": sha(config_path.read_bytes()),
               "exposures_per_selected_example": 1,
               "sampling_interpretation": "Stable random priorities under shared count/coverage/token constraints; not uniform over every feasible subset.",
               "whole_rows_preserved": True, "student_scores_used": False, "topic_quotas_used": False,
               "tolerance_relaxed": False, "data_selection_complete": True,
               "token_accounting": "Stored release counts checked structurally; trainer must reconcile actual labels before training.",
               "screening_scope": pool_summary["screening_scope"],
               "parent_related_development_groups_to_disclose": pool_summary["parent_related_development_groups"],
               "parent_checkpoint_path": None, "training_started": False,
               "benchmark_evaluation_run": False, "shared_protocol_adoption_confirmed": False,
               "remaining_training_requirements": ["Exact compatible saved parent adapter/checkpoint and identity", "All fresh selectors adopt this same universe, coverage rules, and allocator configuration", "Actual selected-row label counts reconciled with the trainer", "Shared continuation recipe with fresh optimizer and scheduler", "Shared frozen development/final evaluation implementation"]}
    shared_config = {"universe_sha256": universe_sha, "universe_rows": len(universe),
                     "screened_pool_sha256": config["screened_pool_sha256"],
                     "parent_manifest_sha256": config["parent_manifest_sha256"],
                     "universe_seed": config["universe_seed"], "quota_seed": config["quota_seed"],
                     "target_supervised_tokens": target, "relative_token_tolerance": config["relative_token_tolerance"],
                     "row_count_adjustment_fraction": config["row_count_adjustment_fraction"],
                     "swap_window": config["swap_window"], "max_priority_swaps": config["max_priority_swaps"],
                     "allocator_version": VERSION, "allocator_tie_seed": config["selection_seed"],
                     "strata": [{"dataset_id": h[0], "length_bin": h[1], "parent_weight": weights[h],
                                 "universe_rows": universe_quotas[h]} for h in sorted(weights)],
                     "shared_methods": ["B", "C0", "E", "C1"], "topic_quotas": None,
                     "universe_frozen_as_file": True, "team_adoption_confirmed": False}
    safe_outputs(output_dir, {
        "common_candidate_universe.jsonl": universe_bytes,
        "shared_allocation_config.json": json_bytes(shared_config),
        "train_B.jsonl": training_bytes,
        "selection_manifest.csv": csv_bytes(manifest),
        "candidate_random_priorities.csv": csv_bytes(audit),
        "selected_ids.txt": ("\n".join(sorted(selected_ids)) + "\n").encode(),
        "coverage_report.csv": csv_bytes(coverage),
        "budget_report.json": json_bytes(summary),
        "allocation_log.json": json_bytes(allocation),
        "selection_config.json": config_path.read_bytes(),
    })
    print("Approach B DATA SELECTION complete - no training started")
    print("Shared candidate universe:", len(universe))
    print("Universe hash:", universe_sha)
    print("Selected examples:", len(selected))
    print("Selected supervised tokens:", allocation["actual_tokens"])
    print("Target supervised tokens:", target)
    print("Difference:", summary["difference_tokens"], f"({summary['difference_percent']:+.6f}%)")
    print("Allowed token range:", summary["allowed_integer_token_range"])
    print("Source composition:", sources)
    print("Source/length groups:", len(coverage), "- quota deviations: 0")
    print("Budget-repair swaps:", allocation["swap_count"])
    print("Parent ID/group overlap: 0 / 0")
    print("Selected-data hash:", summary["training_jsonl_sha256"])
    print("Output folder:", output_dir)
    return summary

"""Deterministic quota allocation with whole examples and bounded boundary swaps."""

from collections import defaultdict
import math

from .provenance import PipelineError, unique_rows


class AllocationError(PipelineError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Allocation failed: {report['reason']}. Nearest found: {report.get('nearest')}. "
                         "Use a separately declared smaller budget, revise coverage explicitly, or increase search limits.")


def quotas(count, proportions, capacities):
    if not proportions or any(not math.isfinite(v) or v <= 0 for v in proportions.values()):
        raise PipelineError("Stratum proportions must be positive finite numbers")
    total = sum(proportions.values())
    exact = {k: count * v / total for k, v in proportions.items()}
    result = {k: math.floor(v) for k, v in exact.items()}
    order = sorted(exact, key=lambda k: (-(exact[k] - result[k]), k))
    for k in order[:count - sum(result.values())]:
        result[k] += 1
    if any(result[k] > capacities.get(k, 0) for k in result):
        return None
    return result


def allocate(rows, target, proportions, tolerance=0.005, boundary_window=16, max_swaps=32, max_row_counts=128):
    unique_rows(rows)
    if type(target) is not int or target <= 0 or not 0 <= tolerance < 1:
        raise PipelineError("Token target must be a positive integer; tolerance must be in [0,1)")
    if type(boundary_window) is not int or boundary_window < 1 or type(max_swaps) is not int or max_swaps < 0:
        raise PipelineError("Invalid bounded swap search settings")
    if type(max_row_counts) is not int or max_row_counts < 1:
        raise PipelineError("max_row_counts must be positive")
    groups = defaultdict(list)
    for r in rows:
        if type(r.get("supervised_tokens")) is not int or r["supervised_tokens"] <= 0:
            raise PipelineError("Allocation requires verified positive integer supervised_tokens")
        if not math.isfinite(r["utility"]):
            raise PipelineError("Nonfinite selection utility")
        groups[r["stratum"]].append(r)
    if set(proportions) != set(groups):
        raise PipelineError("Quota strata must exactly match the common candidate strata")
    for group in groups.values():
        group.sort(key=lambda r: (-r["utility"], r["tie_key"]))
    capacities = {k: len(v) for k, v in groups.items()}
    limit = target * tolerance
    nearest, best, evaluated = None, None, 0
    # All potentially feasible row counts; cheap min/max bounds avoid most searches.
    tokens = sorted(r["supervised_tokens"] for r in rows)
    mins, maxs = [0], [0]
    for token in tokens:
        mins.append(mins[-1] + token)
    for token in reversed(tokens):
        maxs.append(maxs[-1] + token)
    prefixes = {}
    for k, group in groups.items():
        prefixes[k] = [0]
        for row in group:
            prefixes[k].append(prefixes[k][-1] + row["supervised_tokens"])
    plans = []
    for count in range(1, len(rows) + 1):
        q = quotas(count, proportions, capacities)
        if q is None:
            continue
        total = sum(prefixes[k][q[k]] for k in groups)
        if nearest is None or abs(total - target) < abs(nearest["tokens"] - target):
            nearest = {"tokens": total, "rows": count, "quotas": q}
        if mins[count] > target + limit or maxs[count] < target - limit:
            continue
        plans.append((abs(total - target), count, q, total))
    # A feasible prefix has zero rank sacrifice and beats every swapped solution.
    prefix_feasible = [p for p in plans if p[0] <= limit]
    search = sorted(prefix_feasible or plans, key=lambda p: (p[0], p[1]))
    if not prefix_feasible:
        search = search[:max_row_counts]
    for _, count, q, total in search:
        chosen = {k: set(range(q[k])) for k in groups}
        evaluated += 1
        swaps = []
        for _ in range(max_swaps):
            if abs(total - target) <= limit:
                break
            options = []
            for k, group in groups.items():
                inside = sorted(chosen[k], reverse=True)[:boundary_window]
                outside = [i for i in range(len(group)) if i not in chosen[k]][:boundary_window]
                for old in inside:
                    for new in outside:
                        candidate = total - group[old]["supervised_tokens"] + group[new]["supervised_tokens"]
                        if abs(candidate - target) < abs(total - target):
                            options.append((abs(candidate - target), new - old, k, old, new, candidate))
            if not options:
                break
            _, _, k, old, new, total = min(options)
            chosen[k].remove(old)
            chosen[k].add(new)
            swaps.append({"stratum": k, "out": groups[k][old]["row_id"], "in": groups[k][new]["row_id"]})
        if nearest is None or abs(total - target) < abs(nearest["tokens"] - target):
            nearest = {"tokens": total, "rows": count, "quotas": q}
        if abs(total - target) > limit:
            continue
        selected = [dict(groups[k][i], priority_rank=i + 1) for k in sorted(groups) for i in sorted(chosen[k])]
        # Mean excess ordinal rank per row, then token error, then IDs: no utility magnitudes.
        excess = sum(sum(chosen[k]) - q[k] * (q[k] - 1) / 2 for k in chosen)
        objective = (excess / count, abs(total - target), tuple(r["row_id"] for r in selected))
        if best is None or objective < best[0]:
            best = (objective, selected, {"target_tokens": target, "actual_tokens": total,
                    "relative_error": abs(total - target) / target, "tolerance": tolerance,
                    "rows": count, "quotas": q, "quota_deviations": {k: 0 for k in q},
                    "swaps": swaps, "method": "largest-remainder-prefix-boundary-swaps-v1",
                    "coverage_relaxed": False, "search_exhaustive": False,
                    "max_row_counts": max_row_counts, "row_count_prefixes_checked": len(plans)})
    if best is None:
        reason = "proven total-capacity deficit" if sum(tokens) < target * (1 - tolerance) else "bounded search found no feasible allocation"
        raise AllocationError({"reason": reason, "nearest": nearest, "target_tokens": target,
                               "tolerance": tolerance, "row_counts_searched": evaluated})
    best[2]["row_counts_searched"] = evaluated
    return best[1], best[2]

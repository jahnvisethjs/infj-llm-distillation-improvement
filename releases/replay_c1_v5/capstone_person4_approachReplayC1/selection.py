"""Frozen random replay and cached-C1 fresh halves under token and coverage bounds."""

from collections import Counter, defaultdict
from fractions import Fraction

from . import METHOD, VERSION
from .common import TARGET, PilotError, checked, digest, read, rows, sha, unique, write, write_rows

FRESH_OBJECTIVE = {"student_scores_used": True, "random_utility": None,
                   "utility_definition": "round(1000000 * fit) * (1 + confirmed_mistake)"}


def cell(row):
    return row["dataset_id"], row["length_bin"]


def half_bounds(reference):
    from capstone_person4.b_select import bounds
    return bounds(Fraction(sum(r["supervised_tokens"] for r in reference), 2), "0.005")


def quotas(reference):
    counts = Counter(cell(r) for r in reference)
    if any(n % 2 for n in counts.values()):
        raise PilotError("Exact half-row quotas require even parent stratum counts")
    return {key: value // 2 for key, value in counts.items()}


def solve_replay(reference):
    """Select seen rows explicitly; never weaken the upstream fresh-only solver."""
    import ortools
    from ortools.sat.python import cp_model
    from capstone_person4.b_select import ORTOOLS_VERSION, bounds, tie_utility, utility, validate
    if ortools.__version__ != ORTOOLS_VERSION:
        raise PilotError("Use OR-Tools " + ORTOOLS_VERSION)
    validate(reference)
    required = quotas(reference)
    ordered = sorted(reference, key=lambda r: r["row_id"])
    grouped = defaultdict(list)
    for i, row in enumerate(ordered):
        grouped[cell(row)].append(i)
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"replay_{i}") for i in range(len(ordered))]
    for key, count in required.items():
        indexes = grouped[key]
        model.add(sum(variables[i] for i in indexes) == count)
        target = Fraction(sum(ordered[i]["supervised_tokens"] for i in indexes), 2)
        low, high = bounds(target, "0.05")
        tokens = sum(variables[i] * ordered[i]["supervised_tokens"] for i in indexes)
        model.add(tokens >= low)
        model.add(tokens <= high)
    low, high = half_bounds(reference)
    tokens = sum(v * r["supervised_tokens"] for v, r in zip(variables, ordered))
    model.add(tokens >= low)
    model.add(tokens <= high)
    primary = sum(v * utility(r["row_id"], 42) for v, r in zip(variables, ordered))
    model.maximize(primary)

    def solver(limit):
        result = cp_model.CpSolver()
        result.parameters.num_search_workers = 1
        result.parameters.random_seed = 42
        result.parameters.max_deterministic_time = limit
        return result

    first = solver(120)
    status = first.solve(model)
    report = {"status": first.status_name(status), "selection_seed": 42, "bounds": [low, high],
              "infeasibility_proven": status == cp_model.INFEASIBLE,
              "objective_definition": "B seeded hash priorities over verified parent examples"}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, report
    chosen = [i for i, v in enumerate(variables) if first.value(v)]
    objective = sum(utility(ordered[i]["row_id"], 42) for i in chosen)
    report.update(objective=objective, best_bound=first.best_objective_bound,
                  approximate_optimization=status != cp_model.OPTIMAL)
    model.add(primary == objective)
    model.minimize(sum(v * tie_utility(r["row_id"], 42) for v, r in zip(variables, ordered)))
    for i, variable in enumerate(variables):
        model.add_hint(variable, int(i in chosen))
    second = solver(10)
    tie_status = second.solve(model)
    report["tie_status"] = second.status_name(tie_status)
    if tie_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        chosen = [i for i, v in enumerate(variables) if second.value(v)]
    return [ordered[i] for i in chosen], report


def validate_half(reference, selected, universe, portion):
    from capstone_person4.b_select import bounds
    unique(selected)
    canonical = {r["row_id"]: r for r in universe}
    if any(canonical.get(r["row_id"]) != r for r in selected):
        raise PilotError(f"{portion} rows differ from their canonical input")
    # Parent bins in its historical JSONL may differ from B's frozen intervals.
    # Use the rebinned reference for coverage, retaining original row contents.
    bins = {r["row_id"]: cell(r) for r in reference} if portion == "replay" else {
        r["row_id"]: cell(r) for r in universe}
    if dict(Counter(bins[r["row_id"]] for r in selected)) != quotas(reference):
        raise PilotError(f"{portion} source/length quotas differ from half the parent")
    total = sum(r["supervised_tokens"] for r in selected)
    low, high = half_bounds(reference)
    if not low <= total <= high:
        raise PilotError(f"{portion} supervised-token half-budget failed")
    strata = []
    for key, count in sorted(quotas(reference).items()):
        parent_tokens = sum(r["supervised_tokens"] for r in reference if cell(r) == key)
        actual = sum(r["supervised_tokens"] for r in selected if bins[r["row_id"]] == key)
        low, high = bounds(Fraction(parent_tokens, 2), "0.05")
        if not low <= actual <= high:
            raise PilotError(f"{portion} stratum-token bound failed: {key}")
        strata.append({"source": key[0], "length_bin": key[1], "rows": count, "tokens": actual})
    return {"rows": len(selected), "tokens": total, "strata": strata}


def validate_mix(bundle, replay, fresh):
    unique(replay + fresh)
    for key in ("row_id", "problem_group_id"):
        if {r[key] for r in fresh} & {r[key] for r in bundle.parent}:
            raise PilotError("Fresh portion overlaps original parent " + key)
    parts = {"replay": validate_half(bundle.rebinned, replay, bundle.parent, "replay"),
             "fresh": validate_half(bundle.rebinned, fresh, bundle.bank, "fresh")}
    total = sum(p["tokens"] for p in parts.values())
    if abs(total - TARGET) > TARGET * Fraction(5, 1000):
        raise PilotError("Combined token budget failed")
    return {"portions": parts, "actual_tokens": total,
            "replay_token_fraction": parts["replay"]["tokens"] / total, "target_tokens": TARGET}


def ledger(bundle, combined, replay_ids):
    reference = {r["row_id"]: cell(r) for r in bundle.rebinned}
    return [{"row_id": r["row_id"], "problem_group_id": r["problem_group_id"],
             "portion": "replay" if r["row_id"] in replay_ids else "fresh",
             "source": r["dataset_id"], "length_bin": reference[r["row_id"]][1]
             if r["row_id"] in replay_ids else r["length_bin"],
             "supervised_tokens": r["supervised_tokens"]} for r in combined]


def select(bundle):
    from capstone_person4.b_select import hash32, solve
    from capstone_person4.selection import fit_priorities
    from capstone_person4_approachC1.generation_scoring import c1_utilities
    dest = bundle.root / "selection"
    if (dest / "selection.json").is_file():
        return load_selection(bundle)[1]
    replay_binned, replay_solver = solve_replay(bundle.rebinned)
    if replay_binned is None:
        write(dest / "replay_failure.json", replay_solver)
        raise PilotError("Replay solve unresolved/infeasible; bounds were not relaxed")
    # Preserve native parent record contents; quota labels live in the ledger.
    parent_by_id = {r["row_id"]: r for r in bundle.parent}
    replay = [parent_by_id[r["row_id"]] for r in replay_binned]
    fit = fit_priorities(bundle.bank, bundle.scores, 42)
    mistakes = {r["row_id"]: r["mistake"] for r in bundle.outcomes}
    utilities = c1_utilities(fit, mistakes, 1000000, Fraction(1))
    fresh, fresh_solver = solve(bundle.rebinned, bundle.bank, Fraction(1, 2), seed=42, utilities=utilities)
    if fresh is None:
        write(dest / "fresh_failure.json", fresh_solver)
        raise PilotError("Fresh C1 solve unresolved/infeasible; bounds were not relaxed")
    # The shared allocator's default receipt describes B's random objective,
    # even when integer utilities are supplied. Describe the actual objective;
    # this changes reporting only, never utilities, constraints or chosen rows.
    fresh_solver.update(FRESH_OBJECTIVE)
    order = lambda r: (hash32("42|training-order|" + r["row_id"]), r["row_id"])
    replay, fresh = sorted(replay, key=order), sorted(fresh, key=order)
    combined = sorted(replay + fresh, key=order)
    totals = validate_mix(bundle, replay, fresh)
    members = ledger(bundle, combined, {r["row_id"] for r in replay})
    for name, values in (("replay", replay), ("fresh", fresh), ("train", combined), ("membership", members)):
        write_rows(dest / (name + ".jsonl"), values)
    report = {"version": VERSION, "method": METHOD, "identity": bundle.identity,
              "binding_sha256": digest(bundle.binding), **totals, "rows": len(combined),
              "replay_solver": replay_solver, "fresh_solver": fresh_solver,
              "manifests": {name: sha(dest / (name + ".jsonl")) for name in ("replay", "fresh", "train", "membership")}}
    write(dest / "selection.json", report)
    load_selection(bundle)
    return report


def load_selection(bundle):
    dest = bundle.root / "selection"
    report = read(dest / "selection.json")
    if (report.get("identity") != bundle.identity or report.get("binding_sha256") != digest(bundle.binding)
            or report.get("method") != METHOD or report.get("version") != VERSION):
        raise PilotError("Selection belongs to changed inputs, code or another method")
    if any(report.get("fresh_solver", {}).get(k) != v for k, v in FRESH_OBJECTIVE.items()):
        raise PilotError("Fresh solver report does not identify the C1 objective")
    data = {}
    for name in ("replay", "fresh", "train", "membership"):
        data[name] = rows(checked(dest / (name + ".jsonl"), report["manifests"][name]))
    totals = validate_mix(bundle, data["replay"], data["fresh"])
    unique(data["train"])
    combined_map = {r["row_id"]: r for r in data["replay"] + data["fresh"]}
    if len(data["train"]) != len(combined_map) or any(combined_map.get(r["row_id"]) != r for r in data["train"]):
        raise PilotError("Combined manifest is not the two validated portions")
    expected_ledger = ledger(bundle, data["train"], {r["row_id"] for r in data["replay"]})
    if data["membership"] != expected_ledger or any(report.get(k) != v for k, v in totals.items()):
        raise PilotError("Membership ledger or reported budget changed")
    return data, report

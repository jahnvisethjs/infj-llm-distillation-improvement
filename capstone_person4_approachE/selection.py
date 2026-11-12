"""E's binary mistake objective over the unchanged common candidate universe."""

from collections import Counter
import csv
import io
from pathlib import Path

from capstone_person4 import b_select
from capstone_person4.audit import gate
from capstone_person4.config import artifact_root as upstream_root, stage_fingerprint, without_paths
from capstone_person4.continuation import load_selection as load_control
from capstone_person4.data import coverage_parent_rows, load_prepared
from capstone_person4.provenance import (
    PipelineError, digest, read_json, read_rows, sha, source_hashes, unique_rows,
    write_bytes, write_json, write_rows,
)
from capstone_person4.selection import random_priorities, stratum
from capstone_person4_approachC1 import generation_scoring as generation

VERSION = "mistake-only-v1-cpsat"


def code_hashes():
    package = Path(__file__).parent
    repo = package.parent
    files = list(package.glob("*.py")) + list((repo / "scripts/sol").glob("e_*.sh"))
    files += list((repo / "scripts/sol").glob("e_*.sbatch"))
    submission = repo / "scripts/sol/submit_e_pipeline.sh"
    if submission.is_file():
        files.append(submission)
    return {p.relative_to(repo).as_posix(): sha(p) for p in sorted(set(files))}


def inputs(cfg):
    rows, preparation = load_prepared(cfg)
    outcomes, coverage = generation.load_generation(cfg)
    ids = [o.get("row_id") for o in outcomes]
    if len(ids) != len(set(ids)) or set(ids) != {r["row_id"] for r in rows}:
        raise PipelineError("E outcomes must cover every candidate exactly once")
    for outcome in outcomes:
        status = outcome.get("status")
        if (status not in generation.STATUSES or status == "execution_error"
                or type(outcome.get("mistake")) is not bool
                or outcome["mistake"] != (status == "incorrect")
                or type(outcome.get("correct")) is not bool
                or outcome["correct"] != (status == "correct")
                or type(outcome.get("generated_tokens")) is not int
                or outcome["generated_tokens"] < 0 or not isinstance(outcome.get("flags"), list)):
            raise PipelineError(f"Invalid or unresolved execution outcome: {outcome.get('row_id')}")
    if coverage.get("outcomes_sha256") != sha(generation.generation_root(cfg) / "outcomes.jsonl"):
        raise PipelineError("E outcome content changed")
    return rows, preparation, outcomes, coverage


def binding(cfg):
    _, preparation, _, coverage = inputs(cfg)
    return {"version": VERSION, "candidate_sha256": preparation["candidate_sha256"],
            "preparation_fingerprint": stage_fingerprint(cfg, "prepare"),
            "upstream_select_fingerprint": stage_fingerprint(cfg, "select"),
            "parent_fingerprint": cfg["parent"]["adapter_sha256"],
            "generation_key": coverage["generation_key"], "outcomes_sha256": coverage["outcomes_sha256"],
            "generation_code_sha256": sha(generation.__file__), "e_code": code_hashes(),
            "reused_code": source_hashes(), "allocation": without_paths(cfg["allocation"])}


def fingerprint(cfg, stage="select", seedless=False):
    payload = binding(cfg)
    if stage == "train":
        selection = read_json(selection_root(cfg) / "allocation_report.json")
        recipe = dict(cfg["training"])
        if seedless:
            recipe.pop("seed", None)
        payload.update(selection_sha256=digest(selection), training=recipe)
    elif stage != "select":
        raise PipelineError(f"Unknown E stage: {stage}")
    return digest(payload)


def root(cfg, stage="train"):
    return Path(cfg["output_root"]) / cfg["active_profile"] / ("e-" + fingerprint(cfg, stage)[:16])


def selection_root(cfg):
    return root(cfg, "select") / "selection/E"


def priorities(rows, outcomes, seed, required_rows):
    """Lexicographic mistake count, B random objective, then B solver tie hash."""
    unique_rows(rows)
    ids = [o.get("row_id") for o in outcomes]
    if len(ids) != len(set(ids)) or set(ids) != {r["row_id"] for r in rows}:
        raise PipelineError("Explicit mistakes required for the entire universe exactly once")
    mistakes = {o["row_id"]: o.get("mistake") for o in outcomes}
    if any(type(v) is not bool for v in mistakes.values()):
        raise PipelineError("Mistake indicators must be booleans")
    random = {r["row_id"]: b_select.utility(r["row_id"], seed) for r in rows}
    multiplier = required_rows * max(random.values(), default=0) + 1
    # Fixed row quotas make a constant mistake term irrelevant. Normalize it
    # away so the solver gets B's exact objective even under finite time limits.
    offset = int(all(mistakes.values()))
    # CP-SAT validates the sum of all variable domains, not just feasible quotas.
    if sum(multiplier * int(mistakes[i]) + random[i] for i in random) >= 2**62:
        raise PipelineError("E integer objective would exceed the safe CP-SAT range")
    result = [{**r, "mistake": mistakes[r["row_id"]], "utility": int(mistakes[r["row_id"]]),
               "random_utility": random[r["row_id"]],
               "utility_integer": multiplier * (int(mistakes[r["row_id"]]) - offset) + random[r["row_id"]],
               "random_order": r["utility"]} for r in random_priorities(rows, seed)]
    return result, multiplier


def order(rows):
    # Within a constant indicator vector this is exactly the existing B manifest order.
    return sorted(rows, key=lambda r: (r["stratum"], -int(r["mistake"]), -r["random_order"], r["tie_key"]))


def validate_budget(parent, chosen, settings):
    unique_rows(chosen)
    quotas = {f"{s}|{b}": n for (s, b), n in b_select.quotas(parent, 1).items() if n}
    if dict(Counter(stratum(r) for r in chosen)) != quotas:
        raise PipelineError("E source/length coverage differs from parent quotas")
    target = sum(r["supervised_tokens"] for r in parent)
    actual = sum(r["supervised_tokens"] for r in chosen)
    lo, hi = b_select.bounds(target, str(settings["tolerance"]))
    if not lo <= actual <= hi:
        raise PipelineError("E global supervised-token budget failed")
    for key in quotas:
        reference = sum(r["supervised_tokens"] for r in parent if stratum(r) == key)
        tokens = sum(r["supervised_tokens"] for r in chosen if stratum(r) == key)
        lo, hi = b_select.bounds(reference, str(settings["stratum_tolerance"]))
        if not lo <= tokens <= hi:
            raise PipelineError(f"E stratum token bound failed: {key}")
    return target, actual, quotas


def controls(cfg, rows):
    candidates = {r["row_id"]: r for r in rows}
    result = {}
    for method in ("B", "C0", "C1"):
        selected, report = load_control(cfg, method)
        if any(candidates.get(r["row_id"]) != r for r in selected):
            raise PipelineError(f"{method} uses another candidate universe")
        result[method] = (selected, report)
    # C1's legacy selection report binds generation_key but not outcome content.
    # Its batch gate is the evidence that the completed arm used these outcomes.
    record = read_json(upstream_root(cfg, "select") / "selection/C1/batch_gate.json")
    _, coverage = generation.load_generation(cfg)
    expected = {"candidate_sha256": inputs(cfg)[1]["candidate_sha256"],
                "generation_key": coverage["generation_key"], "outcomes_sha256": coverage["outcomes_sha256"],
                "c0_selection_sha256": digest(result["C0"][1]), "generation_code_sha256": sha(generation.__file__)}
    if (any(record.get("bindings", {}).get(k) != v for k, v in expected.items())
            or record.get("selection_sha256") != digest(result["C1"][1])
            or record.get("diagnostics", {}).get("proceed_to_training") is not True
            or any(record.get("diagnostics", {}).get("degeneracy_flags", {}).values())):
        raise PipelineError("Completed C1 gate does not bind the current candidate grading/selection")
    return result


def select(cfg):
    gate(cfg, "select")
    rows, prep, outcomes, coverage = inputs(cfg)
    directory = selection_root(cfg)
    if (directory / "allocation_report.json").is_file():
        return load_selection(cfg)[1]
    parent = coverage_parent_rows(cfg)
    target = sum(r["supervised_tokens"] for r in parent)
    if target != prep.get("parent_replay_tokens") or target != cfg["allocation"]["expected_replay_tokens"]:
        raise PipelineError("Prepared parent budget does not match verified parent rows")
    competing = controls(cfg, rows)
    settings = cfg["allocation"]
    weighted, multiplier = priorities(rows, outcomes, settings["seed"], len(parent))
    utilities = {r["row_id"]: r["utility_integer"] for r in weighted}
    chosen, solver = b_select.solve(
        parent, rows, 1, seed=settings["seed"], deterministic_limit=settings["deterministic_limit"],
        tie_limit=settings["tie_limit"], global_tolerance=str(settings["tolerance"]),
        stratum_tolerance=str(settings["stratum_tolerance"]), utilities=utilities)
    if chosen is None:
        write_json(directory / "failures" / (digest(solver)[:16] + ".json"), solver)
        raise PipelineError(f"E solve {solver.get('status')}; quotas and tolerances were not relaxed")
    solver = {**solver, "student_scores_used": True, "student_signals": ["confirmed_mistake"]}
    ids = {r["row_id"] for r in chosen}
    ordered = order(weighted)
    ranks = Counter()
    table = []
    for r in ordered:
        ranks[r["stratum"]] += 1
        if r["row_id"] in ids:
            table.append({k: r[k] for k in ("row_id", "problem_group_id", "stratum", "mistake", "utility",
                                          "random_utility", "utility_integer", "tie_key", "supervised_tokens")} |
                         {"priority_rank": ranks[r["stratum"]]})
    by_id = {r["row_id"]: r for r in rows}
    manifest = [by_id[r["row_id"]] for r in table]
    target, actual, quotas = validate_budget(parent, manifest, settings)
    status = {o["row_id"]: o["status"] for o in outcomes}
    diagnostics = {}
    for key in sorted(quotas):
        subset = [r for r in rows if stratum(r) == key]
        selected = [r for r in manifest if stratum(r) == key]
        diagnostics[key] = {"required_rows": quotas[key], "candidate_status_counts": dict(Counter(status[r["row_id"]] for r in subset)),
                            "selected_status_counts": dict(Counter(status[r["row_id"]] for r in selected))}
    write_rows(directory / "manifest.jsonl", manifest)
    write_json(directory / "ordered_ids.json", [r["row_id"] for r in manifest])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    write_bytes(directory / "selection_table.csv", stream.getvalue().encode("utf-8"))
    report = {"synthetic": False, "method": "E", "version": VERSION, "bindings": binding(cfg),
              "stage_fingerprint": fingerprint(cfg), "candidate_sha256": prep["candidate_sha256"],
              "parent_fingerprint": cfg["parent"]["adapter_sha256"], "generation_key": coverage["generation_key"],
              "outcomes_sha256": coverage["outcomes_sha256"], "rows": len(manifest),
              "target_tokens": target, "actual_tokens": actual, "tolerance": settings["tolerance"],
              "stratum_tolerance": settings["stratum_tolerance"], "quotas": quotas, "coverage_relaxed": False,
              "manifest_sha256": sha(directory / "manifest.jsonl"), "selected_data_hash": digest(manifest),
              "selection_table_sha256": sha(directory / "selection_table.csv"),
              "ordered_ids_sha256": sha(directory / "ordered_ids.json"),
              "utility_definition": "M * confirmed_mistake + Method B seeded random utility",
              "constant_mistake_term_omitted": len({o["mistake"] for o in outcomes}) == 1,
              "dominance_multiplier": multiplier, "mistakes_in_universe": sum(o["mistake"] for o in outcomes),
              "mistakes_selected": sum(r["mistake"] for r in table), "status_by_stratum": diagnostics,
              "cpsat": solver, "comparisons": {
                  m: {"overlap_rows": len(ids & {r["row_id"] for r in selected}),
                      "jaccard": len(ids & {r["row_id"] for r in selected}) / len(ids | {r["row_id"] for r in selected}),
                      "control_tokens": rec["actual_tokens"], "E_tokens": actual,
                      "absolute_token_difference": abs(actual - rec["actual_tokens"]),
                      "control_selection_sha256": digest(rec)} for m, (selected, rec) in competing.items()},
              "evaluation_status": "pending"}
    write_json(directory / "allocation_report.json", report)
    return load_selection(cfg)[1]


def load_selection(cfg):
    directory = selection_root(cfg)
    report = read_json(directory / "allocation_report.json")
    manifest = read_rows(directory / "manifest.jsonl")
    rows, prep, outcomes, _ = inputs(cfg)
    by_id = {r["row_id"]: r for r in rows}
    if (report.get("synthetic") is not False or report.get("method") != "E"
            or report.get("bindings") != binding(cfg) or report.get("stage_fingerprint") != fingerprint(cfg)
            or report.get("candidate_sha256") != prep["candidate_sha256"]
            or report.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]
            or report.get("manifest_sha256") != sha(directory / "manifest.jsonl")
            or report.get("selected_data_hash") != digest(manifest)
            or any(by_id.get(r["row_id"]) != r for r in manifest)):
        raise PipelineError("E selection is stale, synthetic, or belongs to other inputs")
    for name, key in (("selection_table.csv", "selection_table_sha256"), ("ordered_ids.json", "ordered_ids_sha256")):
        if sha(directory / name) != report.get(key):
            raise PipelineError(f"E selection artifact changed: {name}")
    if read_json(directory / "ordered_ids.json") != [r["row_id"] for r in manifest]:
        raise PipelineError("E selection order changed")
    target, actual, quotas = validate_budget(coverage_parent_rows(cfg), manifest, cfg["allocation"])
    if (report.get("actual_tokens") != actual or report.get("target_tokens") != target or report.get("quotas") != quotas
            or report.get("tolerance") != cfg["allocation"]["tolerance"] or report.get("rows") != len(manifest)
            or report.get("mistakes_selected") != sum(o["mistake"] for o in outcomes if o["row_id"] in {r["row_id"] for r in manifest})):
        raise PipelineError("E token, coverage, or mistake reconciliation failed")
    return manifest, report

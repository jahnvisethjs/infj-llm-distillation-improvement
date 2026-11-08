"""Question-only generation outcomes for the C1 (fit x mistake) and E (mistake-only) selectors.

Person 3 executes the generation under the frozen evaluation contract; this module only
exports the question-only request, verifies and ingests the returned predictions, derives
the confirmed-mistake indicator, and turns it into CP-SAT utilities. It is deliberately
outside STAGE_SOURCES so improving it never moves the fingerprints of the Phase 0 C0 runs;
C1 selections are bound to the outcomes through ``generation_key`` instead.
"""

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from capstone_person4.bindings import verified
from capstone_person4.config import artifact_root, require, without_paths
from capstone_person4.data import load_prepared
from capstone_person4.provenance import PipelineError, digest, read_json, read_rows, sha, write_json, write_rows

INTERFACE = "person4-generation-request-v1"
STATUSES = ("correct", "incorrect", "no_answer", "truncated", "unresolved_equivalence", "execution_error")
# Fields of a candidate row that must never reach Person 3's inference input.
REFERENCE_FIELDS = ("completion", "solution", "solution_raw", "gold_solution", "answer_evidence_text", "upstream_fields_json")
REQUEST_FIELDS = ("problem_id", "benchmark", "source", "split", "question", "gold_answer", "subject", "level")


def generation_key(cfg, preparation):
    """Immutable inputs of an outcome: parent, universe, prompt/decoding/checker policy. Not code."""
    return digest({"parent": without_paths(cfg["parent"]), "candidate_sha256": preparation["candidate_sha256"],
                   "generation": without_paths(require(cfg, "generation")), "interface": INTERFACE})


def generation_root(cfg):
    return artifact_root(cfg, "select") / "generation"


def request_name(cfg):
    return f"{cfg['active_profile']}-parentseed{cfg['parent'].get('training_seed')}"


def export_request(cfg):
    """Write Person 3's question-file schema: question and gold answer only, no solution text."""
    rows, preparation = load_prepared(cfg)
    verified(cfg["generation"]["generation_contract"], cfg["generation"].get("contract_sha256"))
    root = generation_root(cfg) / "request" / request_name(cfg)
    questions = []
    for row in rows:
        question = row.get("problem")
        answer = row.get("final_answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise PipelineError(f"Candidate lacks question text or extracted final answer: {row['row_id']}")
        questions.append({"problem_id": row["row_id"], "benchmark": row["dataset_id"], "source": row.get("original_source"),
                          "split": "candidate", "question": question, "gold_answer": answer, "subject": "None", "level": "None"})
    for record in questions:
        if set(record) != set(REQUEST_FIELDS):
            raise PipelineError("Request schema drift")
        if any(field in record for field in REFERENCE_FIELDS):
            raise PipelineError("Reference text must not enter the generation request")
    write_rows(root / "questions.jsonl", questions)
    key = generation_key(cfg, preparation)
    generation = cfg["generation"]
    request = {"interface": INTERFACE, "synthetic": False, "request_name": request_name(cfg),
               "generation_key": key, "candidate_sha256": preparation["candidate_sha256"], "rows": len(questions),
               "questions_sha256": sha(root / "questions.jsonl"),
               "model": cfg["parent"]["model"], "revision": cfg["parent"]["revision"],
               "adapter": cfg["parent"]["adapter"], "adapter_sha256": cfg["parent"]["adapter_sha256"],
               "adapter_mode": generation["adapter_mode"],
               "contract_sha256": generation["contract_sha256"], "prompt_sha256": generation["prompt_sha256"],
               "decoding": {k: generation[k] for k in ("precision", "decoding", "max_new_tokens", "batch_size",
                                                       "eos_token_id", "add_special_tokens", "question_text")},
               "parser_version": generation["parser_version"], "checker": generation["checker"],
               "gold_answer_role": "grading only; the inference prompt is built from `question` alone under the contract template",
               "return": "one prediction row per problem_id in Person 3's evaluate.py output format, plus the run's config.json"}
    write_json(root / "request.json", request)
    return {**request, "request_directory": str(root)}


def classify(row):
    """Frozen checker policy -> one status plus retained flags. Only `incorrect` is a mistake."""
    flags = []
    if row.get("finish_reason") == "length":
        flags.append("hit_length_limit")
    correct = row.get("correct")
    if isinstance(correct, str):
        correct = {"true": True, "false": False}.get(correct.strip().lower())
    parse = row.get("parse_status")
    if correct is True:
        status = "correct"
    elif parse == "ok" and correct is False:
        status = "incorrect"
    elif parse == "ambiguous":
        status = "unresolved_equivalence"
    elif parse in ("no_answer", "unparseable_number", "unparseable_math"):
        status = "truncated" if "hit_length_limit" in flags else "no_answer"
    elif parse in ("error", "execution_error"):
        status = "execution_error"
    else:
        raise PipelineError(f"Unknown checker outcome for {row.get('problem_id')}: parse_status={parse!r} correct={correct!r}")
    return status, correct is True, flags


def ingest(cfg, predictions_path, report_path):
    """Verify Person 3's predictions against the request and freeze the outcomes."""
    rows, preparation = load_prepared(cfg)
    root = generation_root(cfg)
    request = read_json(root / "request" / request_name(cfg) / "request.json")
    key = generation_key(cfg, preparation)
    if request.get("generation_key") != key or request.get("candidate_sha256") != preparation["candidate_sha256"]:
        raise PipelineError("Request does not belong to this parent/universe/generation policy; export it again")
    report = read_json(report_path)
    generation = cfg["generation"]
    expected = {"checkpoint_sha256": request["adapter_sha256"], "prompt_sha256": generation["prompt_sha256"],
                "parser_version": generation["parser_version"], "precision": generation["precision"],
                "decoding": generation["decoding"], "max_new_tokens": generation["max_new_tokens"],
                "eos_token_id": generation["eos_token_id"], "add_special_tokens": generation["add_special_tokens"],
                "adapter_mode": generation["adapter_mode"]}
    mismatched = {k: (report.get(k), v) for k, v in expected.items() if report.get(k) != v}
    if mismatched:
        raise PipelineError(f"Prediction run config differs from the request: {mismatched}")
    model_revision = f"{request['model']}@{request['revision']}"
    if report.get("model") != request["model"] or report.get("revision") != request["revision"]:
        raise PipelineError("Prediction run config names another base model/revision")
    predictions = read_rows(predictions_path)
    ids = [p.get("problem_id") for p in predictions]
    universe = {r["row_id"] for r in rows}
    if len(ids) != len(set(ids)) or set(ids) != universe:
        raise PipelineError(f"Predictions must cover the request exactly once: {len(set(ids) & universe)} of {len(universe)} rows, "
                            f"{len(ids) - len(set(ids))} duplicates, {len(set(ids) - universe)} unknown")
    outcomes, errors = [], []
    for p in predictions:
        if p.get("model_revision") not in (None, model_revision) or p.get("checkpoint_sha256") not in (None, request["adapter_sha256"]):
            raise PipelineError(f"Prediction row for {p['problem_id']} names another model or checkpoint")
        if p.get("parser_version") not in (None, generation["parser_version"]):
            raise PipelineError(f"Prediction row for {p['problem_id']} used another parser version")
        status, correct, flags = classify(p)
        if status == "execution_error":
            errors.append(p["problem_id"])
            continue
        tokens = p.get("generated_tokens")
        tokens = int(tokens) if isinstance(tokens, str) and tokens.isdigit() else tokens
        if type(tokens) is not int or tokens < 0:
            raise PipelineError(f"Invalid generated token count for {p['problem_id']}")
        outcomes.append({"row_id": p["problem_id"], "status": status, "correct": correct, "mistake": status == "incorrect",
                         "extracted_answer": p.get("parsed_answer"), "generated_tokens": tokens,
                         "stop_reason": p.get("finish_reason"), "flags": flags, "parse_status": p.get("parse_status")})
    if errors:
        raise PipelineError(f"{len(errors)} execution errors must be retried by Person 3 before ingest; they are never mistakes: "
                            + ", ".join(errors[:5]))
    write_rows(root / "outcomes.jsonl", outcomes)
    counts = dict(Counter(o["status"] for o in outcomes))
    coverage = {"synthetic": False, "generation_key": key, "candidate_sha256": preparation["candidate_sha256"],
                "parent_fingerprint": cfg["parent"]["adapter_sha256"], "rows": len(outcomes), "status_counts": counts,
                "mistakes": sum(o["mistake"] for o in outcomes), "flag_counts": dict(Counter(f for o in outcomes for f in o["flags"])),
                "predictions_sha256": sha(predictions_path), "report_sha256": sha(report_path),
                "predictions_path": str(Path(predictions_path).resolve()), "report_path": str(Path(report_path).resolve()),
                "outcomes_sha256": sha(root / "outcomes.jsonl"), "parser_version": generation["parser_version"],
                "checker": generation["checker"], "mistake_policy": generation["mistake_policy"],
                "generated_tokens": sum(o["generated_tokens"] for o in outcomes), "executor": generation["executor"]}
    write_json(root / "coverage.json", coverage)
    return coverage


def load_generation(cfg):
    rows, preparation = load_prepared(cfg)
    root = generation_root(cfg)
    coverage = read_json(root / "coverage.json")
    if (coverage.get("synthetic") is not False or coverage.get("generation_key") != generation_key(cfg, preparation)
            or coverage.get("candidate_sha256") != preparation["candidate_sha256"]
            or coverage.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]
            or coverage.get("outcomes_sha256") != sha(root / "outcomes.jsonl")):
        raise PipelineError("Generation outcomes are missing, stale, or belong to another parent/universe/policy")
    outcomes = read_rows(root / "outcomes.jsonl")
    if len(outcomes) != len(rows) or {o["row_id"] for o in outcomes} != {r["row_id"] for r in rows}:
        raise PipelineError("Generation outcomes do not cover the frozen universe exactly once")
    for o in outcomes:
        if o["status"] not in STATUSES or type(o["mistake"]) is not bool or o["mistake"] != (o["status"] == "incorrect"):
            raise PipelineError(f"Corrupt outcome record: {o['row_id']}")
    return outcomes, coverage


def mistake_bonus(cfg):
    value = Fraction(str(require(cfg, "selection.mistake_bonus")))
    if value < 0:
        raise PipelineError("selection.mistake_bonus must be a nonnegative fraction")
    return value


def c1_utilities(fit_rows, mistakes, scale, bonus):
    """Integer CP-SAT utilities: u_C0 * (den + num*w). lambda=0 or constant w reproduces C0 exactly."""
    base = {r["row_id"]: int(round(r["fit"] * scale)) for r in fit_rows}
    if set(mistakes) != set(base) or any(type(v) is not bool for v in mistakes.values()):
        raise PipelineError("Explicit confirmed-mistake booleans required for the entire universe")
    return {i: u * (bonus.denominator + bonus.numerator * int(mistakes[i])) for i, u in base.items()}


def priorities(cfg, method, rows, seed):
    """Rows with fit, utility and integer CP-SAT utility for C1; E is Phase 2."""
    from capstone_person4.reference_scoring import load_scores
    from capstone_person4.selection import fit_priorities
    if method != "C1":
        raise PipelineError(f"{method} selection is Phase 2 (mistake-only ablation); not implemented yet")
    scores, coverage = load_scores(cfg)
    outcomes, generation = load_generation(cfg)
    fit = fit_priorities(rows, scores, seed)
    mistakes = {o["row_id"]: o["mistake"] for o in outcomes}
    bonus = mistake_bonus(cfg)
    integers = c1_utilities(fit, mistakes, require(cfg, "allocation.utility_scale"), bonus)
    return [{**r, "mistake": mistakes[r["row_id"]], "utility": r["fit"] * (1 + float(bonus) * mistakes[r["row_id"]]),
             "utility_integer": integers[r["row_id"]], "score_fingerprint": coverage["measurement_key"],
             "generation_key": generation["generation_key"], "mistake_bonus": str(bonus)} for r in fit]


def correctness_diagnostics(cfg, rows, scores):
    """Fit versus independent correctness, C0/C1 overlap at the budget, degeneracy flags."""
    from capstone_person4.continuation import budget_and_coverage, cpsat_select
    from capstone_person4.selection import fit_priorities, stratum
    outcomes, coverage = load_generation(cfg)
    by_id = {o["row_id"]: o for o in outcomes}
    seed = cfg["allocation"]["seed"]
    fit = fit_priorities(rows, scores, seed)
    quartile = {}
    for key in sorted({r["stratum"] for r in fit}):
        group = sorted((r for r in fit if r["stratum"] == key), key=lambda r: (-r["fit"], r["row_id"]))
        for i, r in enumerate(group):
            quartile[r["row_id"]] = min(3, 4 * i // len(group))
    mistake_by_quartile = defaultdict(lambda: [0, 0])
    status_by_stratum = defaultdict(Counter)
    for r in fit:
        o = by_id[r["row_id"]]
        mistake_by_quartile[quartile[r["row_id"]]][0] += o["mistake"]
        mistake_by_quartile[quartile[r["row_id"]]][1] += 1
        status_by_stratum[r["stratum"]][o["status"]] += 1
    share = coverage["mistakes"] / coverage["rows"]
    strong_fit_mistakes = sum(1 for r in fit if r["fit"] >= 0.5 and by_id[r["row_id"]]["mistake"])
    target, _, _ = budget_and_coverage(cfg, rows)
    c0_rows, _ = cpsat_select(cfg, "C0", rows, fit, target, None)
    c1_rows, _ = cpsat_select(cfg, "C1", rows, priorities(cfg, "C1", rows, seed), target, None)
    c0_ids, c1_ids = {r["row_id"] for r in c0_rows}, {r["row_id"] for r in c1_rows}
    overlap = len(c0_ids & c1_ids) / len(c0_ids | c1_ids)
    flags = {"mistakes_rare": share < 0.05, "mistakes_near_universal": share > 0.95, "c1_equals_c0": overlap >= 0.95}
    return {"generation_key": coverage["generation_key"], "status_counts": coverage["status_counts"],
            "mistake_share": share, "status_by_stratum": {k: dict(v) for k, v in status_by_stratum.items()},
            "mistake_rate_by_fit_quartile": {f"q{q + 1}": {"mistakes": m, "rows": n, "rate": m / n}
                                             for q, (m, n) in sorted(mistake_by_quartile.items())},
            "confirmed_mistakes_with_fit_at_least_half": strong_fit_mistakes,
            "c0_c1_overlap": {"c0_rows": len(c0_ids), "c1_rows": len(c1_ids), "shared": len(c0_ids & c1_ids),
                              "jaccard": overlap, "c1_mistakes_selected": sum(by_id[i]["mistake"] for i in c1_ids),
                              "c0_mistakes_selected": sum(by_id[i]["mistake"] for i in c0_ids)},
            "degeneracy_flags": flags, "proceed_to_training": not any(flags.values()),
            "mistake_bonus": str(mistake_bonus(cfg)),
            "interpretation": "One greedy generation under the frozen contract is an observation, not a mastery estimate"}

"""Pinned classifier, for standalone CPU tests only."""
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

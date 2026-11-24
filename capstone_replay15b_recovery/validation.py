"""A validation adapter; prediction text, statuses and correctness stay untouched."""

from contextlib import contextmanager

NORMAL_STATUSES = frozenset(("ok", "ambiguous", "no_answer", "empty", "empty_marker",
                             "unparseable_number", "unparseable_math"))
ERROR_STATUSES = frozenset(("verify_error", "error", "execution_error"))
ALIASES = {"empty": "no_answer", "empty_marker": "no_answer"}


class CheckerExecutionError(ValueError):
    """A checker exception requires a clean evaluation attempt, never a wrong score."""


def validate_predictions(base_validator, records, config, expected, benchmark, question_ids):
    view = []
    errors = []
    for row in records:
        status = row.get("parse_status")
        if type(row.get("correct")) is not bool:
            raise ValueError("Evaluation correctness must be explicit booleans")
        if status in ERROR_STATUSES:
            errors.append({"problem_id": row.get("problem_id"), "parse_status": status})
        elif status not in NORMAL_STATUSES:
            raise ValueError(f"Unknown parser status for {row.get('problem_id')}: {status!r}")
        if status in NORMAL_STATUSES and status != "ok" and row["correct"]:
            raise ValueError("A non-ok parser status cannot be a correct answer")
        if status in ALIASES and row.get("parsed_answer") not in (None, ""):
            raise ValueError("An empty answer status contains a parsed answer")
        for key, value in (("checkpoint_sha256", expected["checkpoint_sha256"]),
                           ("train_manifest_sha256", expected["manifest_sha256"]),
                           ("model_revision", config["model"] + "@" + config["revision"])):
            if key in row and row[key] != value:
                raise ValueError("Prediction row belongs to another " + key)
        # Only this ephemeral validation view uses the historical classifier's
        # no_answer spelling. Raw and canonical rows retain their exact status.
        if status in ERROR_STATUSES:
            # Audit the original checkpoint/runtime/question contract before
            # reporting a retryable checker error. These provisional values
            # are never returned or saved: the batch is rejected below.
            view.append({**row, "parse_status": "no_answer", "correct": False})
        else:
            view.append({**row, "parse_status": ALIASES[status]} if status in ALIASES else row)
    outcomes = base_validator(view, config, expected, benchmark, question_ids)
    if errors:
        error = CheckerExecutionError("Checker execution errors require a clean evaluation: " + repr(errors[:10]))
        error.records = errors
        raise error
    return outcomes


@contextmanager
def parser_status_support(evaluation):
    """Install support only for evaluation validation, then restore the frozen API."""
    original = evaluation.validate_predictions
    evaluation.validate_predictions = lambda *args: validate_predictions(original, *args)
    try:
        yield
    finally:
        evaluation.validate_predictions = original

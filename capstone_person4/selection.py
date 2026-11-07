"""C0 fit ranks and B random priorities over a common universe."""

import math
from collections import defaultdict

from .provenance import PipelineError, digest, unique_rows


def stratum(row):
    return f"{row['dataset_id']}|{row['length_bin']}"


def fit_priorities(rows, scores, seed):
    unique_rows(rows)
    if len(scores) != len(rows) or {s["row_id"] for s in scores} != {r["row_id"] for r in rows}:
        raise PipelineError("Reference scores must cover the complete common candidate universe exactly once")
    by_id = {s["row_id"]: s for s in scores}
    if len(by_id) != len(scores):
        raise PipelineError("Duplicate row in score cache")
    groups = defaultdict(list)
    for row in rows:
        score = by_id[row["row_id"]]
        loss = score.get("mean_nll")
        if score.get("status") != "ok" or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
            raise PipelineError(f"Invalid reference score for {row['row_id']}")
        if score.get("supervised_tokens") != row["supervised_tokens"]:
            raise PipelineError(f"Scored token count differs for {row['row_id']}")
        groups[stratum(row)].append((loss, row))
    result = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: (x[0], x[1]["row_id"]))
        start = 0
        while start < len(group):
            end = start + 1
            while end < len(group) and group[end][0] == group[start][0]:
                end += 1
            average_rank = (start + 1 + end) / 2
            fit = 1 - (average_rank - 0.5) / len(group)
            for loss, row in group[start:end]:
                result.append({**row, "fit": fit, "utility": fit, "mean_nll": loss,
                               "stratum": key, "tie_key": digest([seed, row["row_id"]])})
            start = end
    return result


def random_priorities(rows, seed):
    unique_rows(rows)
    return [{**row, "utility": int(digest([seed, row["row_id"]]), 16) / 2**256,
             "stratum": stratum(row), "tie_key": digest([seed, row["row_id"]])} for row in rows]


def combined_priorities(fit_rows, mistakes, bonus=1.0):
    """Extension seam only: callers must provide confirmed-mistake booleans."""
    if bonus < 0 or not math.isfinite(bonus):
        raise PipelineError("Mistake bonus must be finite and nonnegative")
    if set(mistakes) != {r["row_id"] for r in fit_rows} or any(type(v) is not bool for v in mistakes.values()):
        raise PipelineError("Explicit confirmed-mistake booleans required for the entire universe")
    return [{**r, "utility": r["fit"] * (1 + bonus * mistakes[r["row_id"]])} for r in fit_rows]

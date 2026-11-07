"""Hash-bound instance protection with conservative, traceable dispositions."""

import ast
from collections import Counter, defaultdict
import csv
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata

from .bindings import contained, verified
from .config import artifact_root, stage_fingerprint
from .provenance import PipelineError, digest, read_json, read_rows, sha, write_bytes, write_json, write_rows

RULE_VERSION = "v4-selection-protection-v1"
SEVERE = {"EXACT", "LIKELY_SAME"}


def canonical_question(row):
    """Keep original question text and references in separate fields."""
    text = row.get("question", row.get("problem"))
    if not isinstance(text, str) or not text.strip() or not row.get("problem_id") or not row.get("benchmark"):
        raise PipelineError("Protected questions require original text, problem_id and benchmark")
    return {**row, "problem": text,
            "final_answer": row.get("gold_answer", row.get("final_answer", row.get("answer")))}


def load_definitions(path, names, namespace):
    """Compile only hash-verified pure definitions; never execute rebuild/import setup."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise PipelineError(f"Supplied matching definitions missing: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def matching_rules(cfg):
    policy = cfg["protection"]
    verified(policy["matcher_source"], policy["matcher_sha256"])
    verified(policy["checker_source"], policy["checker_sha256"])
    namespace = {"re": re, "unicodedata": unicodedata, "defaultdict": defaultdict}
    checker = load_definitions(policy["checker_source"], ("grams", "mask_numbers", "sim"), dict(namespace))
    # Read the supplied threshold assignment rather than replacing its values.
    tree = ast.parse(Path(policy["checker_source"]).read_text(encoding="utf-8-sig"))
    names = ("SAME_J", "SAME_C", "SIMILAR_J", "NUM_J", "NUM_MIN_TOKENS")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) and
                tuple(getattr(n, "id", None) for n in t.elts) == names for t in node.targets):
            checker.update(zip(names, ast.literal_eval(node.value)))
    if tuple(checker.get(k) for k in names) != (0.85, 0.95, 0.60, 0.90, 20):
        raise PipelineError("Supplied matcher thresholds differ from frozen V4 policy")
    namespace = load_definitions(policy["matcher_source"], ("normalize_question", "ProtectedMatcher"), namespace)
    return SimpleNamespace(checker=SimpleNamespace(**{k: checker[k] for k in (*names, "grams", "mask_numbers", "sim")}),
                           normalize=namespace["normalize_question"], matcher_class=namespace["ProtectedMatcher"],
                           fingerprint=digest({"version": RULE_VERSION, "matcher": policy["matcher_sha256"],
                                               "checker": policy["checker_sha256"]}))


def question_sources(cfg):
    sources, combined, seen = [], [], {}
    for spec in cfg["protection"]["question_manifests"]:
        verified(spec["manifest"], spec["sha256"])
        if spec["role"] not in ("development", "fast-development", "final"):
            raise PipelineError("Unknown protected question role")
        rows = [canonical_question(r) for r in read_rows(spec["manifest"])]
        identities = [(r["benchmark"], r["problem_id"]) for r in rows]
        if len(set(identities)) != len(rows):
            raise PipelineError("Duplicate protected identities within source")
        source = {**spec, "rows": len(rows)}
        sources.append(source)
        for row, identity in zip(rows, identities):
            previous = seen.get(identity)
            if previous and (previous["problem"] != row["problem"] or previous["final_answer"] != row["final_answer"]):
                raise PipelineError("Conflicting nested question text/reference")
            if previous and (previous["role"] == "final") != (spec["role"] == "final"):
                raise PipelineError("Development and final identities overlap")
            seen[identity] = {**row, "role": spec["role"]}
            combined.append({**row, "role": spec["role"], "source_manifest": spec["manifest"],
                             "source_sha256": spec["sha256"], "source_name": spec["name"]})
    full = {(r["benchmark"], r["problem_id"]) for r in combined if r["role"] == "development"}
    fast = {(r["benchmark"], r["problem_id"]) for r in combined if r["role"] == "fast-development"}
    if not fast <= full:
        raise PipelineError("Fast development is not nested in full development")
    return combined, sources


def check_inputs(cfg):
    rules = matching_rules(cfg)
    combined, sources = question_sources(cfg)
    root = Path(cfg["data"]["release"])
    meta = read_json(root / "metadata/release_manifest.json")
    expected_names = {"dev_gsm", "dev_math", "dev_fast", "test_gsm", "test_math500"}
    if {s["name"] for s in sources} != expected_names or len(sources) != 5:
        raise PipelineError("V4 protection requires all five separate frozen question manifests")
    for spec in sources:
        relative = "inputs/evaluation/data/" + spec["name"] + ".jsonl"
        if Path(spec["manifest"]) != root / relative or spec["sha256"] != meta["files"][relative]["sha256"]:
            raise PipelineError("Protection source differs from immutable V4 question manifest")
        expected_role = "fast-development" if spec["name"] == "dev_fast" else "development" if spec["name"].startswith("dev_") else "final"
        if spec["role"] != expected_role:
            raise PipelineError("Protected source role differs")
    reserved = Path(cfg["protection"]["reserved_manifest"])
    if reserved != root / "manifests/loss_dev.jsonl":
        raise PipelineError("Reserved examples must be the supplied loss_dev manifest")
    verified(reserved, cfg["protection"]["reserved_sha256"])
    for field, relative in (("reserved_sha256", "manifests/loss_dev.jsonl"),
                            ("matcher_sha256", "code/capstone_person1/decontamination.py"),
                            ("checker_sha256", "inputs/overlap_evidence/overlap_check.py")):
        if cfg["protection"][field] != meta["files"][relative]["sha256"]:
            raise PipelineError(f"Protection rule/input not bound to V4: {field}")
    normalized_roles = defaultdict(set)
    group_roles = defaultdict(set)
    for row in combined:
        role = "final" if row["role"] == "final" else "development"
        normalized_roles[rules.normalize(row["problem"])].add(role)
        if row.get("problem_group_id"):
            group_roles[row["problem_group_id"]].add(role)
    if any(len(v) > 1 for v in list(normalized_roles.values()) + list(group_roles.values())):
        raise PipelineError("Development and final question texts/groups overlap")
    return {"sources": sources, "unique_questions": len(normalized_roles), "source_rows": len(combined),
            "reserved_sha256": sha(reserved), "rules_fingerprint": rules.fingerprint,
            "readiness_scope": "Question protection only; evaluator and answer checker are deferred"}


def make_matcher(references, rules):
    """Populate the inherited matcher index with any declared comparison boundary."""
    matcher = rules.matcher_class.__new__(rules.matcher_class)
    matcher.checker = rules.checker
    matcher.questions = []
    matcher.exact, matcher.index, matcher.masked_index = defaultdict(set), defaultdict(set), defaultdict(set)
    for idx, row in enumerate(references):
        norm = rules.normalize(row["problem"])
        masked = rules.checker.mask_numbers(norm)
        matcher.questions.append({"eval_set": row["boundary"], "eval_id": row["identity"], "eval_text": row["problem"],
                                  "norm": norm, "grams": rules.checker.grams(norm),
                                  "masked_grams": rules.checker.grams(masked)})
        matcher.exact[norm].add(idx)
        for gram in rules.checker.grams(norm, 5):
            matcher.index[gram].add(idx)
        for gram in rules.checker.grams(masked, 5):
            matcher.masked_index[gram].add(idx)
    return matcher


def dispositions(candidates, references, rules, reviews=(), progress=None):
    """Union group reasons, quarantine ambiguity, and collapse normalized duplicate clusters."""
    ordered = sorted(candidates, key=lambda r: r["row_id"])
    if len({r["row_id"] for r in ordered}) != len(ordered):
        raise PipelineError("Duplicate candidate row IDs")
    matcher = make_matcher(references, rules)
    by_identity = {(r["boundary"], r["identity"]): r for r in references}
    decisions = {}
    for review in reviews:
        key = review.get("pair_key")
        if not key or key in decisions or review.get("decision") not in ("distinct", "same") or not review.get("rationale"):
            raise PipelineError("Reviews need unique pair keys, distinct/same decisions and rationale")
        decisions[key] = review
    reasons, flagged = defaultdict(list), []
    seen_groups = defaultdict(set)
    for row in references:
        if row["boundary"] != "internal" and row.get("problem_group_id"):
            seen_groups[row["problem_group_id"]].add(row["boundary"])
    for position, row in enumerate(ordered, 1):
        rid = row["row_id"]
        for boundary in sorted(seen_groups.get(row["problem_group_id"], ())):
            reasons[rid].append({"disposition": "exclude", "reason": "reserved_or_seen_group", "boundary": boundary})
        # V4's synthetic field describes upstream Orca provenance, not a mock run.
        if (row.get("eligible") is not True or row.get("group_excluded") is True
                or row.get("quality_flags") or row.get("review_flags") or not row.get("problem", "").strip()
                or not row.get("completion", "").strip()):
            reasons[rid].append({"disposition": "exclude", "reason": "quality_or_reference"})
        for hit in matcher.matches({**row, "source": row.get("source", row.get("dataset_id", ""))}, position):
            boundary, identity = hit["eval_set"], hit["eval_id"]
            if boundary == "internal" and identity == rid:
                continue
            counterpart = by_identity[boundary, identity]
            evidence = {"row_id": rid, "counterpart_id": identity, "boundary": boundary,
                        "candidate_text_sha256": digest(row["problem"]), "counterpart_text_sha256": digest(hit["eval_text"]),
                        "rules_fingerprint": rules.fingerprint, "kind": hit["kind"]}
            key = digest(evidence)
            review = decisions.get(key)
            accepted = bool(review and all(review.get(k) == v for k, v in evidence.items()))
            # Exact internal duplicates are resolved as clusters, not arbitrary pair exclusions.
            if boundary == "internal" and hit["kind"] == "EXACT":
                disposition = "cluster"
            elif hit["kind"] in SEVERE:
                disposition = "quarantine" if boundary == "internal" else "exclude"
            elif accepted and review["decision"] == "distinct":
                disposition = "admit_reviewed"
            else:
                disposition = "exclude" if accepted and review["decision"] == "same" and boundary != "internal" else "quarantine"
            record = {**evidence, "pair_key": key, "disposition": disposition, "review_applied": accepted,
                      "candidate_text": row["problem"], "counterpart_text": hit["eval_text"],
                      "jaccard": hit["jaccard"], "containment": hit["containment"],
                      "jaccard_numbers_masked": hit["jaccard_numbers_masked"],
                      "counterpart_provenance": {k: counterpart.get(k) for k in ("source_manifest", "source_sha256", "role")}}
            flagged.append(record)
            if disposition in ("exclude", "quarantine"):
                reason = {"disposition": disposition, "reason": "instance_match", "pair_key": key,
                          "boundary": boundary, "kind": hit["kind"]}
                reasons[rid].append(reason)
                if boundary == "internal":
                    reasons[identity].append(reason)
        if progress and position % 1000 == 0:
            progress(position, len(ordered))
    clusters = defaultdict(list)
    stored_groups = defaultdict(list)
    for row in ordered:
        clusters[rules.normalize(row["problem"])].append(row)
        stored_groups[row["problem_group_id"]].append(row)
    # Propagate protected/ambiguous relationships across stored groups and exact-text clusters.
    changed = True
    while changed:
        changed = False
        for members in list(stored_groups.values()) + list(clusters.values()):
            implicated = any(reasons[r["row_id"]] for r in members)
            if implicated and len(members) > 1:
                for row in members:
                    if not reasons[row["row_id"]]:
                        reasons[row["row_id"]].append({"disposition": "quarantine", "reason": "implicated_duplicate_or_group"})
                        changed = True
    duplicate_records = []
    for norm, members in sorted(clusters.items()):
        cluster = digest(["normalized-question", norm])
        eligible = [r for r in members if not reasons[r["row_id"]]]
        representative = eligible[0]["row_id"] if eligible else None
        if len(members) > 1:
            duplicate_records.append({"duplicate_cluster_id": cluster, "row_ids": [r["row_id"] for r in members],
                                      "representative": representative, "implicated": not bool(eligible)})
            for row in eligible[1:]:
                reasons[row["row_id"]].append({"disposition": "exclude", "reason": "internal_normalized_duplicate", "representative": representative})
    eligible, excluded, quarantine = [], [], []
    # Multiple rows in a stored group also require one stable representative.
    retained_groups = set()
    for row in ordered:
        rid = row["row_id"]
        if not reasons[rid] and row["problem_group_id"] in retained_groups:
            reasons[rid].append({"disposition": "exclude", "reason": "internal_stored_group_duplicate"})
        if reasons[rid]:
            disposition = "exclude" if any(r["disposition"] == "exclude" for r in reasons[rid]) else "quarantine"
            (excluded if disposition == "exclude" else quarantine).append({"row_id": rid, "problem_group_id": row["problem_group_id"], "reasons": reasons[rid]})
        else:
            retained_groups.add(row["problem_group_id"])
            eligible.append({**row, "duplicate_cluster_id": digest(["normalized-question", rules.normalize(row["problem"])])})
    if len(eligible) + len(excluded) + len(quarantine) != len(ordered):
        raise PipelineError("Protection dispositions do not reconcile")
    return {"eligible": eligible, "exclusions": excluded, "quarantine": quarantine,
            "flagged_pairs": flagged, "duplicate_clusters": duplicate_records,
            "unused_review_keys": sorted(set(decisions) - {r["pair_key"] for r in flagged if r["review_applied"]})}


def write_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: r.get(k) for k in fields} for r in rows)
    write_bytes(path, buffer.getvalue().encode("utf-8"))


def protect(cfg):
    from .audit import gate, parent_exposure
    from .bindings import write_bindings
    evidence = gate(cfg, "prepare")
    parent_rows, parent = parent_exposure(cfg)
    rules = matching_rules(cfg)
    candidates = read_rows(cfg["data"]["candidate_manifest"])
    questions, sources = question_sources(cfg)
    references = [{**r, "boundary": "protected", "identity": r["source_name"] + ":" + r["problem_id"]} for r in questions]
    for boundary, rows in (("parent", parent_rows), ("reserved", read_rows(cfg["protection"]["reserved_manifest"])), ("internal", candidates)):
        references.extend({**r, "boundary": boundary, "identity": r["row_id"]} for r in rows)
    reviews = []
    for spec in cfg["protection"].get("reviews", []):
        verified(spec["manifest"], spec["sha256"])
        reviews.extend(read_rows(spec["manifest"]))
    result = dispositions(candidates, references, rules, reviews,
                          progress=lambda current, total: print(f"Protection scan: {current}/{total}", flush=True))
    root = artifact_root(cfg, "prepare") / "protection"
    for key in ("eligible", "exclusions", "quarantine", "duplicate_clusters", "flagged_pairs"):
        write_rows(root / (key + ".jsonl"), result[key])
    fields = ["pair_key", "row_id", "counterpart_id", "boundary", "kind", "disposition", "review_applied",
              "candidate_text_sha256", "counterpart_text_sha256", "rules_fingerprint", "candidate_text", "counterpart_text", "jaccard", "containment", "jaccard_numbers_masked"]
    write_csv(root / "flagged_pairs.csv", result["flagged_pairs"], fields)
    write_csv(root / "review_queue.csv", [r for r in result["flagged_pairs"] if r["disposition"] == "quarantine"], fields)
    summary = {"synthetic": False, "stage": "protection", "stage_fingerprint": stage_fingerprint(cfg, "prepare"),
               "input_rows": len(candidates), "counts": {k: len(result[k]) for k in ("eligible", "exclusions", "quarantine", "duplicate_clusters", "flagged_pairs")},
               "rules_fingerprint": rules.fingerprint, "question_sources": sources, "protection_inputs": evidence["overlap"],
               "parent_binding": parent, "unused_review_keys": result["unused_review_keys"],
               "boundary_flags": dict(Counter(r["boundary"] for r in result["flagged_pairs"])),
               "artifacts": {p.name: sha(p) for p in sorted(root.glob("*")) if p.suffix in (".jsonl", ".csv")},
               "limitations": "Supplied bounded five-gram retrieval; approximate recall; quarantine is eligibility policy, not proof of equivalence"}
    write_json(root / "summary.json", summary)
    write_bindings(cfg, evidence["release"], parent)
    return summary


def load_protection(cfg):
    root = artifact_root(cfg, "prepare") / "protection"
    report = read_json(root / "summary.json")
    if report.get("synthetic") is not False or report.get("stage_fingerprint") != stage_fingerprint(cfg, "prepare"):
        raise PipelineError("Protection evidence is stale or synthetic")
    for relative, expected in report["artifacts"].items():
        verified(contained(root, relative), expected)
    if report["counts"]["eligible"] == 0:
        raise PipelineError("Protection left no eligible candidates")
    rows = read_rows(root / "eligible.jsonl")
    if len(rows) != report["counts"]["eligible"]:
        raise PipelineError("Protection eligible coverage differs")
    return rows, report

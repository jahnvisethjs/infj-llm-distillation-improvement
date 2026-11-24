"""Question-only family screening and fixed held-out routing splits."""

import ast
from collections import defaultdict
import csv
from pathlib import Path
import re
import unicodedata

from .common import (PARENTS, SEEDS, benchmark, bundle, checked_spec, digest, identity, keyed,
                     question, read, require, rows, seal, sha, write, write_csv, write_rows)
from .inputs import load_registry

SOURCES = {"gsm": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
           "math": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3")}
ROLES = {"parent_training", "adapter_training", "selection", "reserved", "development", "official_test"}


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, i, j):
        self.parents[self.find(i)] = self.find(j)


def matcher(spec):
    """Reuse only pure, hash-bound definitions, never the overlap script's CLI."""
    path = checked_spec(spec)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"norm", "mask_numbers", "grams", "sim"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, "Frozen overlap definitions are missing")
    namespace = {"re": re, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def related(a, b, rules):
    if a["norm"] == b["norm"] or a["masked"] == b["masked"] or a["group"] and a["group"] == b["group"]:
        return True
    jac, containment = rules["sim"](a["grams"], b["grams"])
    masked, _ = rules["sim"](a["masked_grams"], b["masked_grams"])
    short = min(len(a["norm"].split()), len(b["norm"].split()))
    return jac >= .60 or containment >= .95 and short >= 12 or masked >= .90 and short >= 20


class QuestionIndex:
    """All shared trigrams are indexed so candidate blocking cannot miss a .60 pair."""
    def __init__(self, rules):
        self.rules = rules
        self.records = []
        self.index = defaultdict(set)

    def describe(self, row):
        norm = self.rules["norm"](question(row))
        masked = self.rules["mask_numbers"](norm)
        return {"norm": norm, "masked": masked, "grams": self.rules["grams"](norm),
                "masked_grams": self.rules["grams"](masked), "group": row.get("problem_group_id", row.get("family_id"))}

    def keys(self, row):
        return [("n", row["norm"]), ("m", row["masked"])] + \
            ([("g", row["group"])] if row["group"] else []) + \
            [("t", g) for g in row["grams"]] + [("mt", g) for g in row["masked_grams"]]

    def matches(self, record):
        candidates = set()
        for key in self.keys(record):
            candidates.update(self.index[key])
        return [i for i in sorted(candidates) if related(record, self.records[i], self.rules)]

    def add(self, record):
        i = len(self.records)
        self.records.append(record)
        for key in self.keys(record):
            self.index[key].add(i)
        return i


def source_rows(spec, download=False):
    bench = benchmark(spec["benchmark"])
    require((spec["dataset"], spec["revision"]) == SOURCES[bench] and spec.get("split") == "train",
            "Router sources must be the pinned GSM8K/MATH training splits")
    if spec.get("path"):
        return rows(checked_spec(spec)), {"path": spec["path"], "sha256": spec["sha256"],
                                       "dataset": spec["dataset"], "revision": spec["revision"]}
    require(download, "Missing local source export; explicitly pass --download to fetch pinned datasets")
    from datasets import load_dataset
    configurations = [spec["configuration"]] if spec.get("configuration") else ["main"] if bench == "gsm" else [
        "algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    records = []
    for configuration in configurations:
        dataset = load_dataset(spec["dataset"], configuration, revision=spec["revision"], split="train")
        records.extend({**dict(row), "problem_id": f"{bench}_{configuration}_train_{i:05d}"} for i, row in enumerate(dataset))
    return records, {"dataset": spec["dataset"], "revision": spec["revision"], "content_sha256": digest(records)}


def exposure_rows(spec):
    path = checked_spec(spec)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    return rows(path)


def audit_exposures(specs, registry):
    require(specs and {s["role"] for s in specs} == ROLES, "All exposure roles must be explicitly inventoried")
    candidates = registry["candidates"]
    expected = {identity(p, m, s) for p in PARENTS for s in SEEDS for m in ("A", "B", "C1")}
    require(set(candidates) == expected, "Router dataset requires all 18 declared checkpoints")
    for cid, cp in candidates.items():
        for role, expected_sha in (("adapter_training", cp["train_manifest_sha256"]),
                                   ("parent_training", cp["parent_manifest_sha256"])):
            matches = [s for s in specs if s["role"] == role and cid in s.get("checkpoint_ids", [])
                       and s.get("sha256") == expected_sha]
            require(matches, f"Missing hash-bound {role} exposure for {cid}")
        if cp["method"] != "A":
            require(any(s["role"] == "selection" and cid in s.get("checkpoint_ids", []) for s in specs),
                    f"Missing candidate selection/scoring exposure for {cid}")
    for role in ("development", "official_test"):
        require({benchmark(s["benchmark"]) for s in specs if s["role"] == role} == {"gsm", "math"},
                f"Both benchmark exposures are required for {role}")


def split_families(records, seed=42):
    groups = defaultdict(list)
    for row in records:
        groups[row["family_id"]].append(row)
    require(len(groups) >= 3, "At least three independent families are required")
    totals = {b: sum(r["benchmark"] == b for r in records) for b in ("gsm", "math")}
    targets = {s: {b: totals[b] * f for b in totals} for s, f in (("train", .6), ("validation", .2), ("test", .2))}
    actual = {s: {b: 0 for b in totals} for s in targets}
    assignments = {}
    ordered = sorted(groups, key=lambda g: (-len(groups[g]), digest([seed, g])))
    for family in ordered:
        counts = {b: sum(r["benchmark"] == b for r in groups[family]) for b in totals}
        # Minimize the increase in normalized squared target error, with stable ties.
        def cost(split):
            return sum(((actual[split][b] + counts[b] - targets[split][b]) ** 2
                        - (actual[split][b] - targets[split][b]) ** 2) / max(1, targets[split][b]) for b in totals)
        selected = min(targets, key=lambda s: (cost(s), s))
        assignments[family] = selected
        for b in totals:
            actual[selected][b] += counts[b]
    result = [{**r, "split": assignments[r["family_id"]]} for r in records]
    require(all(any(r["split"] == s and r["benchmark"] == b for r in result)
                for s in targets for b in totals), "Family structure prevents usable benchmark-balanced splits")
    return result


def screen_families(population, rules):
    """Propagate any direct exposure through the entire question-family component."""
    keyed(population)
    family_index = QuestionIndex(rules)
    uf = UnionFind(len(population))
    for i, row in enumerate(population):
        record = family_index.describe(row)
        for j in family_index.matches(record):
            uf.union(i, j)
        family_index.add(record)
    families = defaultdict(list)
    for i, row in enumerate(population):
        families[uf.find(i)].append(row["benchmark"] + "/" + row["problem_id"])
    family_ids = {i: digest(sorted(ids)) for i, ids in families.items()}
    tainted = {uf.find(i) for i, row in enumerate(population) if row["exposure_matches"]}
    candidates, excluded = [], []
    for i, row in enumerate(population):
        if not row["gold_answer"] or uf.find(i) in tainted:
            excluded.append({"benchmark": row["benchmark"], "problem_id": row["problem_id"],
                             "reason": "missing_gold" if not row["gold_answer"] else "exposure_or_transitive_family",
                             "direct_matches": row["exposure_matches"], "family_id": family_ids[uf.find(i)]})
        else:
            candidates.append({k: v for k, v in {**row, "family_id": family_ids[uf.find(i)]}.items() if k != "exposure_matches"})
    return candidates, excluded


def prepare_pool(cfg, registry_root, output, download=False):
    registry = load_registry(registry_root)
    specs = cfg.get("exposures", [])
    require(cfg.get("exposure_inventory_complete") is True, "Explicit complete exposure inventory is required")
    audit_exposures(specs, registry)
    rules = matcher(cfg["matcher"])
    excluded_index = QuestionIndex(rules)
    for spec in specs:
        for row in exposure_rows(spec):
            excluded_index.add(excluded_index.describe(row))
    from .live import parser_module
    parser = parser_module(cfg["evaluator"])
    population, candidates, excluded, source_provenance = [], [], [], []
    require({benchmark(s["benchmark"]) for s in cfg["sources"]} == {"gsm", "math"}, "Need both router sources")
    for spec in cfg["sources"]:
        bench = benchmark(spec["benchmark"])
        source, provenance = source_rows(spec, download)
        source_provenance.append(provenance)
        for pos, raw in enumerate(source):
            text = question(raw)
            answer = parser.gsm_gold(str(raw["answer"])) if bench == "gsm" else parser.math_gold(raw)
            rid = str(raw.get("problem_id", f"{bench}_train_{pos:05d}"))
            row = {"benchmark": bench, "problem_id": rid, "question": text,
                   "problem_group_id": raw.get("problem_group_id"), "gold_answer": answer,
                   "source_dataset": spec["dataset"], "source_revision": spec["revision"]}
            matches = excluded_index.matches(excluded_index.describe(row))
            population.append({**row, "exposure_matches": len(matches)})
    candidates, excluded = screen_families(population, rules)
    selected = []
    count = cfg.get("pool_size", 3000)
    require(type(count) is int and count >= 30 and count % 2 == 0, "Pool size must be an even integer >= 30")
    for bench in ("gsm", "math"):
        eligible = [(i, r) for i, r in enumerate(candidates) if r["benchmark"] == bench]
        require(len(eligible) >= count // 2, f"Insufficient eligible {bench} questions: {len(eligible)}")
        for i, row in sorted(eligible, key=lambda pair: digest([42, bench, pair[1]["problem_id"]]))[:count // 2]:
            selected.append(row)
    selected = split_families(selected)
    root = Path(output)
    for split in ("train", "validation", "test"):
        subset = [r for r in selected if r["split"] == split]
        write_rows(root / (split + ".questions.jsonl"),
                   [{k: r[k] for k in ("benchmark", "problem_id", "question", "family_id", "split")} for r in subset])
        write_rows(root / "grading" / (split + ".jsonl"), subset)
    write_csv(root / "exclusions.csv", excluded)
    write(root / "provenance.json", {"sources": source_provenance, "exposures": specs, "matcher": cfg["matcher"],
          "candidate_population": len(candidates), "pool_size": len(selected), "seed": 42,
          "family_counts": {s: len({r["family_id"] for r in selected if r["split"] == s}) for s in ("train", "validation", "test")},
          "split_counts": {s: sum(r["split"] == s for r in selected) for s in ("train", "validation", "test")},
          "limitations": "No known overlap under the frozen question matcher; approximate family coverage and base pretraining exposure are not proven."})
    seal(root, "pool", [registry_root], registry["synthetic"])
    return read(root / "provenance.json")


def pool_questions(root, splits=("train", "validation", "test")):
    bundle(root, "pool")
    all_rows = [r for s in ("train", "validation", "test") for r in rows(Path(root) / (s + ".questions.jsonl"))]
    keyed(all_rows)
    families = defaultdict(set)
    for row in all_rows:
        require(row.get("family_id") and row.get("split") in ("train", "validation", "test"), "Missing split/family")
        families[row["family_id"]].add(row["split"])
    require(all(len(s) == 1 for s in families.values()), "Question family crosses router splits")
    allowed = {"benchmark", "problem_id", "question", "family_id", "split"}
    require(all(set(r) == allowed for r in all_rows), "Router question manifest contains non-feature fields")
    return [r for r in all_rows if r["split"] in splits]

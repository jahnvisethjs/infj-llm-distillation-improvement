"""Explicit frozen-evaluator execution/import, separate from the live answer API."""

from pathlib import Path
import subprocess
import sys

from .common import (EVALUATOR_SHA, PARSER_SHA, PARENTS, SEEDS, benchmark, bundle, checked,
                     checked_spec, digest, identity, keyed, qkey, read, require, rows,
                     seal, sha, write, write_bytes, write_rows)
from .inputs import load_registry, prediction_records
from .live import verify_candidate
from .pool import pool_questions


def depends_on(root, target):
    root, target = Path(root).resolve(), Path(target).resolve()
    if root == target:
        return True
    return any(depends_on(s["path"], target) for s in read(root / "receipt.json")["upstream"])


def label(cfg, registry_root, pool_root, split, output, authorize_gpu=False, frozen_root=None, import_index=None, reuse_root=None):
    require(split in ("train", "validation", "test"), "Unknown router split")
    registry = load_registry(registry_root)
    expected = {identity(p, m, s) for p in PARENTS for s in SEEDS for m in ("A", "B", "C1")}
    require(set(registry["candidates"]) == expected, "All 18 candidates are required for matched labeling")
    frozen_hash = None
    upstream = [registry_root, pool_root]
    if split == "test":
        require(frozen_root is not None, "Final-test labeling requires the complete frozen experiment set")
        bundle(frozen_root, "frozen")
        require(depends_on(frozen_root, pool_root) and depends_on(frozen_root, registry_root), "Frozen routers use another pool/registry")
        frozen_hash = sha(Path(frozen_root) / "receipt.json")
        upstream.append(frozen_root)
    questions = pool_questions(pool_root, (split,))
    grading = keyed(rows(Path(pool_root) / "grading" / (split + ".jsonl")))
    require(set(grading) == set(keyed(questions)), "Grading and router question coverage differs")
    imports = read(import_index) if import_index else []
    require(authorize_gpu or imports, "Generation requires explicit --authorize-gpu; imports do not")
    evaluator = checked(cfg["evaluator"]["path"], EVALUATOR_SHA)
    checked(cfg["evaluator"]["parser_path"], PARSER_SHA)
    require(evaluator.parent / "answer_parser.py" == Path(cfg["evaluator"]["parser_path"]),
            "Evaluator must import the verified sibling answer_parser.py")
    root = Path(output)
    if (root / "receipt.json").exists():
        bundle(root, "labels")
        metadata = read(root / "labels.json")
        require(metadata["split"] == split and metadata["pool_receipt_sha256"] == sha(Path(pool_root) / "receipt.json")
                and metadata["frozen_receipt_sha256"] == frozen_hash
                and metadata["checkpoint_hashes"] == {cid: c["checkpoint_sha256"] for cid, c in registry["candidates"].items()},
                "Existing label output belongs to another experiment")
        return metadata
    coverage = []
    for cid, candidate in sorted(registry["candidates"].items()):
        outcomes = {}
        for bench in ("gsm", "math"):
            subset = [r for r in questions if r["benchmark"] == bench]
            full = [grading[qkey(r)] for r in subset]
            require(all(g["question"] == q["question"] and g.get("gold_answer") not in (None, "")
                        for g, q in zip(full, subset)), "Grading inputs differ from question manifest")
            choices = [s for s in imports if s["candidate"] == cid and benchmark(s["benchmark"]) == bench and s["split"] == split]
            require(len(choices) <= 1, "Ambiguous imported label attempts")
            if choices:
                item = choices[0]
                source_questions = keyed(rows(checked_spec(item["questions"])))
                require(set(source_questions) == set(keyed(full)), "Imported question coverage differs")
                for row in full:
                    original = source_questions[qkey(row)]
                    require(original["question"] == row["question"] and original["gold_answer"] == row["gold_answer"],
                            "Imported question/reference differs")
                values, report = prediction_records({**item, "questions_sha256": item["questions"]["sha256"]},
                                                     candidate, keyed(full))
                coverage.append({"candidate": cid, "benchmark": bench, "imported": True,
                                 "source_predictions": item["predictions"], "source_config": item["config"],
                                 "source_questions": item["questions"], **report})
                outcomes.update(values)
                continue
            require(authorize_gpu, f"No imported labels for {cid}/{bench}; GPU generation was not authorized")
            verify_candidate(candidate)
            chunk_size = cfg.get("label_chunk_size", 250)
            require(type(chunk_size) is int and chunk_size > 0, "Invalid label chunk size")
            for offset in range(0, len(full), chunk_size):
                chunk = full[offset:offset + chunk_size]
                directory = root / cid / "chunks" / f"{bench}_{offset:05d}"
                question_path = directory / "questions.jsonl"
                predictions = directory / "predictions.jsonl"
                config_path = directory / "predictions.config.json"
                if (directory / "receipt.json").is_file():
                    bundle(directory, "label_chunk")
                else:
                    require(not directory.exists(), f"Incomplete chunk preserved: {directory}; choose a new output attempt")
                    source_chunk = Path(reuse_root) / cid / "chunks" / f"{bench}_{offset:05d}" if reuse_root else None
                    if source_chunk and (source_chunk / "receipt.json").is_file():
                        reused = bundle(source_chunk, "label_chunk")
                        require(reused["synthetic"] == registry["synthetic"], "Mixed synthetic/real reused chunks")
                        require(rows(source_chunk / "questions.jsonl") == chunk, "Reused chunk has other questions/references/split")
                        for name in ("questions.jsonl", "predictions.jsonl", "predictions.config.json"):
                            write_bytes(directory / name, (source_chunk / name).read_bytes())
                        coverage.append({"candidate": cid, "benchmark": bench, "offset": offset,
                                         "reused_chunk": str(source_chunk.resolve()),
                                         "source_receipt_sha256": sha(source_chunk / "receipt.json")})
                    else:
                        write_rows(question_path, chunk)
                        command = [sys.executable, "-B", str(evaluator), "--eval_file", str(question_path),
                                   "--out", str(predictions), "--run_id", f"routing_{split}_{cid}_{bench}_{offset}",
                                   "--model", candidate["model"], "--revision", candidate["revision"],
                                   "--adapter", candidate["checkpoint"], "--expect_train_manifest_sha256", candidate["train_manifest_sha256"],
                                   "--precision", "fp32", "--batch_size", "16", "--max_new_tokens", "1024", "--question_text", "p1_clean"]
                        if split == "test":
                            command.append("--final_test")
                        subprocess.run(command, check=True)
                require(rows(question_path) == chunk, "Cached label chunk belongs to other questions")
                spec = {"predictions": str(predictions), "config": str(config_path), "questions_sha256": sha(question_path)}
                values, report = prediction_records(spec, candidate, keyed(chunk))
                if not (directory / "receipt.json").is_file():
                    seal(directory, "label_chunk", synthetic=registry["synthetic"])
                coverage.append({"candidate": cid, "benchmark": bench, "offset": offset, "imported": False, **report})
                outcomes.update(values)
        write_rows(root / cid / "outcomes.jsonl", [{"benchmark": q["benchmark"], "problem_id": q["problem_id"],
                   "question_sha256": digest(q["question"]), "correct": outcomes[qkey(q)]} for q in questions])
    metadata = {"split": split, "questions": len(questions), "checkpoint_hashes": {
        cid: c["checkpoint_sha256"] for cid, c in registry["candidates"].items()},
        "pool_receipt_sha256": sha(Path(pool_root) / "receipt.json"), "frozen_receipt_sha256": frozen_hash,
        "coverage": coverage, "official_final_benchmarks_used": False,
        "evaluator_sha256": EVALUATOR_SHA, "parser_sha256": PARSER_SHA}
    write(root / "labels.json", metadata)
    seal(root, "labels", upstream, registry["synthetic"])
    return metadata

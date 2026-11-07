"""Teacher-forced completion NLL with causal shifting and raw per-token records."""

from collections import defaultdict
from contextlib import nullcontext
import math
from pathlib import Path
import time

from .audit import gate
from .config import artifact_root, stage_fingerprint, without_paths, STAGE_SOURCES
from .data import load_prepared
from .person2_adapter import Person2Adapter
from .provenance import PipelineError, digest, environment, read_gzip, read_json, sha, source_hashes, write_gzip, write_json, write_rows
from .selection import stratum


def token_losses(logits, input_ids, labels, attention_mask, chunk_size=64):
    """Return per-example post-shift losses. Inputs are [batch, sequence, ...]."""
    import torch
    import torch.nn.functional as F
    if logits.ndim != 3 or input_ids.shape != labels.shape or labels.shape != attention_mask.shape or logits.shape[:2] != labels.shape:
        raise PipelineError("Incompatible logits/labels/attention shapes")
    if (labels[:, 0] != -100).any():
        raise PipelineError("The first token has no causal predictor and must be ignored")
    if ((attention_mask == 0) & (labels != -100)).any():
        raise PipelineError("Padding labels must be ignored by position")
    if ((labels != -100) & (labels != input_ids)).any():
        raise PipelineError("Supervised labels differ from input tokens")
    outputs = []
    for b in range(labels.shape[0]):
        records = []
        for start in range(1, labels.shape[1], chunk_size):
            stop = min(start + chunk_size, labels.shape[1])
            targets = labels[b, start:stop]
            values = F.cross_entropy(logits[b, start - 1:stop - 1].float(), targets, ignore_index=-100, reduction="none")
            # Transfer once per chunk; per-token CUDA synchronization is prohibitively slow.
            cpu_targets = targets.detach().cpu().tolist()
            cpu_values = values.detach().cpu().tolist()
            for offset, target in enumerate(cpu_targets):
                if target == -100:
                    continue
                value = float(cpu_values[offset])
                if not math.isfinite(value) or value < 0:
                    raise PipelineError("Nonfinite or negative token NLL")
                pos = start + offset
                records.append({"position": pos, "token_id": target, "nll": value})
        if not records:
            raise PipelineError("No effective supervised next-token targets")
        total = math.fsum(t["nll"] for t in records)
        outputs.append({"mean_nll": total / len(records), "total_nll": total,
                        "supervised_tokens": len(records), "tokens": records})
    return outputs


def validate_score(record, expected_count=None):
    tokens = record["tokens"]
    count = record["supervised_tokens"]
    if count <= 0 or count != len(tokens) or (expected_count is not None and count != expected_count):
        raise PipelineError("Per-token count does not match row/scoring manifest")
    positions = [t["position"] for t in tokens]
    if positions != sorted(set(positions)) or any(t["nll"] < 0 or not math.isfinite(t["nll"]) for t in tokens):
        raise PipelineError("Invalid per-token positions/NLL")
    total = math.fsum(t["nll"] for t in tokens)
    if not math.isclose(total, record["total_nll"], rel_tol=1e-7, abs_tol=1e-7) or not math.isclose(total / count, record["mean_nll"], rel_tol=1e-7, abs_tol=1e-7):
        raise PipelineError("Per-token NLL does not reconstruct row sum/mean")


def score_batch(model, encoded, pad_id, dtype=None):
    import torch
    device = next(model.parameters()).device
    width = max(len(e["input_ids"]) for e in encoded)
    ids, labels, masks = [], [], []
    for e in encoded:
        padding = width - len(e["input_ids"])
        ids.append(e["input_ids"] + [pad_id] * padding)
        labels.append(e["labels"] + [-100] * padding)
        masks.append([1] * len(e["input_ids"]) + [0] * padding)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    mask = torch.tensor(masks, dtype=torch.long, device=device)
    model.eval()
    context = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype else nullcontext()
    with torch.inference_mode(), context:
        logits = model(input_ids=ids, attention_mask=mask).logits
        result = token_losses(logits, ids, labels, mask)
    return result


def model_stamp(model):
    """Track every parameter version plus exact LoRA contents before/after inference."""
    import torch
    versions, adapters = {}, {}
    for name, param in model.named_parameters():
        versions[name] = param._version
        if "lora_" in name:
            value = param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            adapters[name] = __import__("hashlib").sha256(value).hexdigest()
    return {"versions": versions, "adapter_hashes": adapters}


def measurement_key(cfg, preparation, runtime=None):
    return digest({"parent": without_paths(cfg["parent"]), "prepared": preparation["candidate_sha256"],
                   "tokenizer": preparation["tokenizer_fingerprint"], "scoring": cfg["scoring"],
                   "format": preparation["evidence"]["release"]["format_sha256"],
                   "runtime": runtime if runtime is not None else environment(),
                   "code": {k: v for k, v in source_hashes().items() if k in STAGE_SOURCES["score-reference"]},
                   "log_base": "e", "units": "nats"})


def diagnostic_subset(rows, count):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset_id"]].append(row)
    for values in groups.values():
        values.sort(key=lambda r: (r.get("total_tokens", r["supervised_tokens"]), r["row_id"]))
    result, seen = [], set()
    rounds = max(1, (count + len(groups) - 1) // len(groups))
    for i in range(rounds):
        for key in sorted(groups):
            values = groups[key]
            index = round(i * (len(values) - 1) / max(1, rounds - 1))
            row = values[index]
            if row["row_id"] not in seen:
                result.append(row)
                seen.add(row["row_id"])
            if len(result) == count:
                return result
    for row in sorted(rows, key=lambda r: r["row_id"]):
        if len(result) == count:
            break
        if row["row_id"] not in seen:
            result.append(row)
            seen.add(row["row_id"])
    return result


def shared_cache_root(cfg):
    return Path(cfg["output_root"]) / "reference_cache" / cfg["active_profile"]


def smoke_paths(cfg, key):
    # Keyed per runtime so a smoke on another GPU/runtime never collides with this one.
    return (artifact_root(cfg, "score-reference") / "reference" / "smoke" / (key[:16] + ".json"),
            shared_cache_root(cfg) / "smoke" / (key + ".json"))


def smoke_proof(cfg, key):
    for path in smoke_paths(cfg, key):
        if path.exists():
            proof = read_json(path)
            if (proof.get("measurement_key") == key and proof.get("passed") is True
                    and proof.get("synthetic") is False and proof.get("parent_unchanged") is True):
                return proof
    return None


def score_reference(cfg, smoke=False):
    stage_started = time.perf_counter()
    evidence = gate(cfg, "score-reference")
    rows, preparation = load_prepared(cfg)
    runtime = evidence["runtime"]
    key = measurement_key(cfg, preparation, runtime)
    root = artifact_root(cfg, "score-reference") / "reference"
    smoke_path, shared_smoke = smoke_paths(cfg, key)
    if not smoke and smoke_proof(cfg, key) is None:
        # The smoke must pass on this exact GPU/runtime: run it inside this allocation
        # instead of trusting a proof from another node type or package set.
        score_reference(cfg, smoke=True)
        if smoke_proof(cfg, key) is None:
            raise PipelineError("Run the matching score-reference --smoke before candidate-wide scoring")
    scoring = cfg["scoring"]
    batch_size, smoke_rows = scoring["batch_size"], scoring["smoke_rows"]
    if type(batch_size) is not int or batch_size < 1 or type(smoke_rows) is not int or not 2 <= smoke_rows <= 32:
        raise PipelineError("batch_size must be positive and smoke_rows must be 2–32")
    if smoke:
        rows = diagnostic_subset(rows, smoke_rows)
    adapter = Person2Adapter(cfg)
    tokenizer = adapter.tokenizer()
    if adapter.tokenizer_fingerprint != preparation["tokenizer_fingerprint"]:
        raise PipelineError("Actual tokenizer serialization changed after preparation")
    encoded = [adapter.encode(r, tokenizer) for r in rows]
    import torch
    torch.cuda.reset_peak_memory_stats()
    model = adapter.load_parent(trainable=False)
    before = model_stamp(model)
    dtype = getattr(torch, scoring["dtype"])
    worst = {"token_nll_abs": 0.0, "row_mean_nll_abs": 0.0}
    if smoke:
        # Check all smoke rows individually and in genuinely padded batches. bfloat16
        # kernels differ by batch shape, and on Sol single tokens differed by up to 0.44
        # nats while row means differed by <0.02, so the gate is the per-row mean NLL
        # (the quantity C0 ranks on); masking/padding bugs shift rows by whole nats.
        # The per-token maximum is recorded for inspection only.
        singles = [score_batch(model, [e], tokenizer.pad_token_id, dtype)[0] for e in encoded]
        batches = []
        for offset in range(0, len(rows), max(2, batch_size)):
            batches.extend(score_batch(model, encoded[offset:offset + max(2, batch_size)], tokenizer.pad_token_id, dtype))
        for single, batch in zip(singles, batches):
            if single["supervised_tokens"] != batch["supervised_tokens"]:
                raise PipelineError("Padded batching changed the effective supervised-token count")
            a = torch.tensor([t["nll"] for t in single["tokens"]])
            b = torch.tensor([t["nll"] for t in batch["tokens"]])
            worst["token_nll_abs"] = max(worst["token_nll_abs"], (a - b).abs().max().item())
            worst["row_mean_nll_abs"] = max(worst["row_mean_nll_abs"], abs(single["mean_nll"] - batch["mean_nll"]))
            if not math.isclose(single["mean_nll"], batch["mean_nll"], abs_tol=scoring["atol"], rel_tol=scoring["rtol"]):
                raise PipelineError(f"Unbatched/padded-batch row mean NLL differs beyond declared tolerance: {worst}")
    started = time.perf_counter()
    fresh_seconds, fresh_tokens, fresh_rows, cache_hits = 0.0, 0, 0, 0
    scores = []
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset:offset + batch_size]
        batch_enc = encoded[offset:offset + batch_size]
        missing = []
        records = {}
        for i, (row, enc) in enumerate(zip(batch_rows, batch_enc)):
            row_key = digest([key, row, enc])
            path = shared_cache_root(cfg) / key / (row_key + ".json.gz")
            if path.exists():
                record = read_gzip(path)
                if (record.get("cache_key") != row_key or record.get("row_id") != row["row_id"]
                        or record.get("measurement_key") != key
                        or record.get("rendered_input_hash") != digest([enc["input_ids"], enc["labels"]])):
                    raise PipelineError("Cache entry input fingerprint differs")
                validate_score(record, row["supervised_tokens"])
                records[i] = record
                cache_hits += 1
            else:
                missing.append((i, row, enc, row_key, path))
        if missing:
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            values = score_batch(model, [item[2] for item in missing], tokenizer.pad_token_id, dtype)
            torch.cuda.synchronize()
            fresh_seconds += time.perf_counter() - forward_start
            fresh_tokens += sum(item[1]["supervised_tokens"] for item in missing)
            fresh_rows += len(missing)
            for (i, row, enc, row_key, path), value in zip(missing, values):
                record = {**value, "row_id": row["row_id"], "status": "ok", "measurement_key": key,
                          "parent_fingerprint": cfg["parent"]["adapter_sha256"], "cache_key": row_key,
                          "rendered_input_hash": digest([enc["input_ids"], enc["labels"]]),
                          "precision": scoring["dtype"], "log_base": "e", "units": "nats",
                          "per_token_artifact": str(path)}
                validate_score(record, row["supervised_tokens"])
                write_gzip(path, record)
                records[i] = record
        for i in range(len(batch_rows)):
            scores.append({k: v for k, v in records[i].items() if k != "tokens"})
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if model_stamp(model) != before or sha(Path(cfg["parent"]["adapter"]) / "adapter_model.safetensors") != cfg["parent"]["adapter_sha256"]:
        raise PipelineError("Parent parameters/artifacts changed during reference scoring")
    report = {"synthetic": False, "measurement_key": key, "passed": True, "rows": len(rows),
              "supervised_tokens": sum(s["supervised_tokens"] for s in scores),
              "config_fingerprint": cfg["config_fingerprint"], "stage_fingerprint": stage_fingerprint(cfg, "score-reference"),
              "candidate_sha256": preparation["candidate_sha256"],
              "code": source_hashes(), "runtime": runtime, "parent_unchanged": True, "atol": scoring["atol"], "rtol": scoring["rtol"],
              "padded_batch_max_difference": worst if smoke else None}
    if smoke:
        write_json(smoke_path, report)
        write_json(shared_smoke, {k: v for k, v in report.items() if k != "config_fingerprint"})
    else:
        write_rows(root / "scores.jsonl", scores)
        report["scores_sha256"] = sha(root / "scores.jsonl")
        write_json(root / "coverage.json", report)
    # Runtime observations are distinct from immutable scientific cache records.
    runtime_path = root / "runtime" / (__import__("uuid").uuid4().hex + ".json")
    write_json(runtime_path, {"smoke": smoke, "seconds": seconds, "end_to_end_wall_seconds": time.perf_counter() - stage_started,
                             "rows": len(rows),
                             "newly_scored_rows": fresh_rows, "cache_hits": cache_hits,
                             "fresh_forward_seconds": fresh_seconds, "fresh_scored_supervised_tokens": fresh_tokens,
                             "fresh_supervised_tokens_per_second": fresh_tokens / fresh_seconds if fresh_seconds else None,
                             "projected_full_pool_forward_seconds": sum(r["supervised_tokens"] for r in load_prepared(cfg)[0]) * fresh_seconds / fresh_tokens if fresh_tokens else None,
                             "peak_memory_bytes": torch.cuda.max_memory_allocated(), "environment": environment(),
                             "note": "Forward projection excludes load, tokenization and disk overhead; resume/cold costs are distinct"})
    return {**report, "runtime_report": str(runtime_path)}


def load_scores(cfg):
    rows, prep = load_prepared(cfg)
    root = artifact_root(cfg, "score-reference") / "reference"
    coverage = read_json(root / "coverage.json")
    from .provenance import read_rows
    scores = read_rows(root / "scores.jsonl")
    if (coverage.get("synthetic") is not False or coverage.get("passed") is not True or coverage.get("parent_unchanged") is not True
            or coverage.get("measurement_key") != measurement_key(cfg, prep, coverage.get("runtime"))
            or coverage.get("scores_sha256") != sha(root / "scores.jsonl") or coverage.get("rows") != len(rows)):
        raise PipelineError("Reference cache is partial, stale or belongs to another parent/runtime")
    if len(scores) != len(rows) or {s["row_id"] for s in scores} != {r["row_id"] for r in rows}:
        raise PipelineError("Reference cache has duplicate/missing/unexpected candidate IDs")
    counts = {r["row_id"]: r["supervised_tokens"] for r in rows}
    for score in scores:
        full = read_gzip(score["per_token_artifact"])
        validate_score(full, counts[score["row_id"]])
        if (full.get("measurement_key") != coverage["measurement_key"]
                or full.get("parent_fingerprint") != cfg["parent"]["adapter_sha256"]
                or full.get("precision") != cfg["scoring"]["dtype"] or full.get("status") != "ok"):
            raise PipelineError("Per-token reference record belongs to a different parent/measurement/precision")
        if {k: v for k, v in full.items() if k != "tokens"} != score:
            raise PipelineError("Reference index differs from its per-token cache artifact")
    return scores, coverage


def diagnose(cfg):
    rows, _ = load_prepared(cfg)
    scores, coverage = load_scores(cfg)
    by_id = {s["row_id"]: s for s in scores}
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(by_id[row["row_id"]]["mean_nll"])
    report = {"synthetic": False, "coverage": coverage, "strata": {
        k: {"rows": len(v), "min_nll": min(v), "mean_nll": math.fsum(v) / len(v), "max_nll": max(v)}
        for k, v in sorted(groups.items())}, "interpretation": "Reference fit only; independent correctness is unmeasured"}
    write_json(artifact_root(cfg, "score-reference") / "diagnostics/reference_fit.json", report)
    return report

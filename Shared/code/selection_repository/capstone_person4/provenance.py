"""Content-bound artifacts. Outputs are additive; no cleanup/deletion is performed."""

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid


class PipelineError(ValueError):
    """An actionable validation failure, rather than a successful partial experiment."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix == ".csv":
            rows = list(csv.DictReader(stream))
        else:
            rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise PipelineError(f"Empty manifest: {path}")
    return rows


def write_bytes(path, content):
    """Atomic creation; identical reruns are allowed, different existing files are preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise PipelineError(f"Output already exists with different content: {path}. Use a new output directory.")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temp.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    # Never replace an existing artifact, including in a competing worker.
    os.link(temp, path)
    # The retained .partial hard link is harmless and deliberately not deleted.


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def write_rows(path, rows):
    write_bytes(path, ("".join(canonical(row) + "\n" for row in rows)).encode("utf-8"))


def write_gzip(path, value):
    write_bytes(path, gzip.compress(canonical(value).encode("utf-8"), mtime=0))


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def unique_rows(rows):
    ids, groups = set(), set()
    for row in rows:
        rid, gid = row.get("row_id"), row.get("problem_group_id")
        if not isinstance(rid, str) or not rid or not isinstance(gid, str) or not gid:
            raise PipelineError("Every canonical row needs nonempty row_id and problem_group_id")
        if rid in ids or gid in groups:
            raise PipelineError(f"Duplicate row or problem group: {rid}, {gid}")
        ids.add(rid)
        groups.add(gid)
    return ids, groups


def environment():
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "determinism_environment": {k: os.environ.get(k) for k in
                ("MKL_THREADING_LAYER", "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")}}


def source_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}

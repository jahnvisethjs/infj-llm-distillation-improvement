"""Small immutable artifact protocol used only by the routing experiment."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

from . import VERSION

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SEEDS = (42, 137, 2026)
PARENTS = (17, 123)
METHODS = ("A", "B", "C1")
EVALUATOR_SHA = "8f2bc66ddfe3a0000bfc1e08ee325a0f520263f057a0e2fc7576d1467776d3d1"
PARSER_SHA = "3361e3c81864072961026fca4d7459a17c7544ed5ffec144575ab20eeac93d5d"
PROMPT_SHA = "22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db"
QUESTIONS = {"gsm": "3cb2c71007d23e01cb70b6370c1ad82ad5bcea3527c7f3258b4814cf2e3d1cc6",
             "math": "641460fdc55beff591daad0cc7530a3f933eb263c3dfdf8a0534fcfaf8196416"}
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SETTINGS = {"model": MODEL, "revision": REVISION, "precision": "fp32", "max_new_tokens": 1024,
            "batch_size": 16, "stop_at_question": False, "question_text": "p1_clean",
            "prompt_sha256": PROMPT_SHA, "add_special_tokens": False, "decoding": "greedy",
            "eos_token_id": 151643, "adapter_mode": "merged", "parser_version": "p3-parser-v6",
            "cut_at_next_question": True, "math_verify": "0.9.0", "torch": "2.6.0+cu124",
            "transformers": "4.57.1", "peft": "0.17.1", "cuda": "12.4",
            "gpu": "NVIDIA A100-SXM4-80GB", "evaluate_py_sha256": EVALUATOR_SHA,
            "answer_parser_sha256": PARSER_SHA}


class RoutingError(ValueError):
    """Missing evidence or an experiment invariant that cannot be relaxed silently."""


def require(condition, message):
    if not condition:
        raise RoutingError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checked(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, f"Missing SHA256: {path}")
    require(sha(path) == expected, f"Hash mismatch: {path}")
    return Path(path)


def checked_spec(spec):
    return checked(spec["path"], spec.get("sha256"))


def write_bytes(path, value):
    """Create only; interrupted outputs are retained and need a new destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == value, f"Existing artifact preserved; choose a new output: {path}")
        return
    with path.open("xb") as stream:
        stream.write(value)


def write(path, value):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())


def write_rows(path, records):
    write_bytes(path, "".join(canonical(row) + "\n" for row in records).encode())


def write_csv(path, records, fields=None):
    records = list(records)
    fields = fields or list(dict.fromkeys(k for r in records for k in r))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    write_bytes(path, output.getvalue().encode("utf-8"))


def code_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def seal(root, kind, upstream=(), synthetic=False):
    root = Path(root).resolve()
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p != root / "receipt.json"}
    receipt = {"version": VERSION, "kind": kind, "synthetic": synthetic, "files": files,
               "upstream": [{"path": str(Path(p).resolve()), "sha256": sha(Path(p) / "receipt.json")}
                            for p in upstream], "code": code_hashes()}
    write(root / "receipt.json", receipt)
    return receipt


def bundle(root, kind=None, _visited=None, _verified=None):
    root = Path(root).resolve()
    visited = set() if _visited is None else _visited
    verified = {} if _verified is None else _verified
    require(root not in visited, "Cyclic artifact dependencies")
    if root in verified:
        receipt = verified[root]
        require(kind is None or receipt.get("kind") == kind, f"Wrong routing artifact: {root}")
        return receipt
    visited.add(root)
    receipt = read(root / "receipt.json")
    require(receipt.get("version") == VERSION and (kind is None or receipt.get("kind") == kind),
            f"Wrong routing artifact: {root}")
    for name, expected in receipt["files"].items():
        path = (root / name).resolve()
        require(root in path.parents, "Artifact path escapes its directory")
        checked(path, expected)
    for spec in receipt["upstream"]:
        checked(Path(spec["path"]) / "receipt.json", spec["sha256"])
        bundle(spec["path"], _visited=visited.copy(), _verified=verified)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "receipt.json"}
    require(actual == set(receipt["files"]), f"Unreceipted files in sealed artifact: {root}")
    verified[root] = receipt
    return receipt


def load_config(path):
    path = Path(path).resolve()
    cfg = read(path)
    require(cfg.get("version") == VERSION, "Expected routing-v1 configuration")
    path_keys = {"path", "parser_path", "checkpoint", "run_config", "training_summary", "parent_checkpoint",
                 "parent_run_config", "predictions", "config", "receipt", "a", "b", "c1"}

    def resolve(value):
        if isinstance(value, dict):
            return {k: str((path.parent / v).resolve()) if k in path_keys and isinstance(v, str)
                    else resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(cfg)


def destination(output, inputs):
    target = Path(output).resolve()
    for value in inputs:
        if value is None:
            continue
        source = Path(value).resolve()
        require(target != source and target not in source.parents and source not in target.parents,
                f"Output overlaps an immutable input: {source}")
    return target


def identity(parent, method, seed):
    require(type(parent) is int and parent in PARENTS and method in METHODS
            and type(seed) is int and seed in SEEDS, "Unknown parent, policy, or continuation seed")
    return f"p{parent}_{method}_t{seed}"


def benchmark(value):
    require(value in ("gsm", "math", "dev_gsm", "dev_math"), f"Unknown benchmark: {value}")
    return value.removeprefix("dev_")


def question(row):
    text = row.get("question", row.get("problem"))
    require(isinstance(text, str) and text.strip(), "Question text is missing")
    return text


def qkey(row):
    value = row.get("problem_id", row.get("row_id"))
    require(isinstance(value, str) and value, "Question ID is missing")
    return benchmark(row.get("benchmark", row.get("dataset_id"))), value


def keyed(records):
    result = {}
    for row in records:
        key = qkey(row)
        require(key not in result, f"Duplicate question: {key}")
        result[key] = row
    return result

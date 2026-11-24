"""Small, stdlib-only artifact interface, independent of frozen upstream imports."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid

from . import VERSION

BANK_SHA = "ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f"
PARENT_SHA = "11a9de70df8b102e57676fb1c4b8ed61f9f0d1e7f524c6c0f0169368ea17b1cb"
MANIFEST_SHA = "efa258c9c4a8bffa85bb3be2026c187e9035f3918fe044967c4dafadcfa7f303"
RUNNER_SHA = "50809b9e374fcc05a9b637c381396b46223ab85abcda3ca07cd4b6a5a4af882f"
TARGET = 318845
SEEDS = (42, 137, 2026)
RECIPE = {"epochs": 1, "accumulation": 8, "learning_rate": 0.0001,
          "warmup_updates": 10, "microbatch_size": 1, "packing": False}
EVAL_PACKAGES = {"torch": "2.6.0+cu124", "transformers": "4.57.1",
                 "peft": "0.17.1", "math-verify": "0.9.0"}
QUESTIONS = {
    "dev_gsm": "3cb2c71007d23e01cb70b6370c1ad82ad5bcea3527c7f3258b4814cf2e3d1cc6",
    "dev_math": "641460fdc55beff591daad0cc7530a3f933eb263c3dfdf8a0534fcfaf8196416"}
EVALUATOR_SHA = "8f2bc66ddfe3a0000bfc1e08ee325a0f520263f057a0e2fc7576d1467776d3d1"
PARSER_SHA = "3361e3c81864072961026fca4d7459a17c7544ed5ffec144575ab20eeac93d5d"
PROMPT_SHA = "22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db"


class PilotError(ValueError):
    """A missing input or failed scientific/operational invariant."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_sha(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    if not expected or sha(path) != expected:
        raise PilotError(f"Hash mismatch: {path}")
    return Path(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_bytes(path, content):
    """Create immutably. Retain atomic-write hard links, without cleanup/deletion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise PilotError(f"Existing artifact preserved: {path}; use a new output root/attempt")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)


def write(path, value):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_rows(path, records):
    write_bytes(path, ("".join(canonical(r) + "\n" for r in records)).encode())


def unique(records):
    for key in ("row_id", "problem_group_id"):
        values = [r.get(key) for r in records]
        if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise PilotError(f"Empty, invalid, or repeated {key}")


def attempt_name(value):
    if not value or value in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise PilotError("Attempt names must contain only letters, digits, underscores and hyphens")
    return value


def code_hashes():
    package = Path(__file__).parent
    repo = package.parent
    files = list(package.glob("*.py")) + list((repo / "scripts/sol").glob("replay_c1*"))
    return {p.relative_to(repo).as_posix(): sha(p) for p in sorted(files) if p.is_file()}


def load_config(path):
    path = Path(path).resolve()
    cfg = read(path)
    if cfg.get("version") != VERSION or cfg.get("model") != "0.5B" or cfg.get("parent_seed") != 17:
        raise PilotError("This pilot is fixed to version v1, 0.5B and parent17")
    if cfg.get("continuation_seeds") != list(SEEDS) or cfg.get("selection_seed") != 42:
        raise PilotError("Freeze continuation seeds 42/137/2026 and selection seed42")
    for key in ("work", "b_root", "native_runner", "c1_profile", "output_root", "training_python",
                "solver_python", "a_eval_root", "c1_eval_root"):
        value = cfg.get(key)
        if not isinstance(value, str) or not value or "$" in value or "~" in value:
            raise PilotError(f"Set an explicit, expanded path for {key}")
        # venv/bin/python is often a symlink to system Python. Resolving it
        # changes sys.prefix and loses the frozen environment on Linux.
        cfg[key] = (os.path.abspath(path.parent / value) if key in ("training_python", "solver_python")
                    else str((path.parent / value).resolve()))
    if cfg.get("a_policy"):
        cfg["a_policy"] = str((path.parent / cfg["a_policy"]).resolve())
    output = Path(cfg["output_root"])
    immutable = [Path(cfg[k]) for k in ("b_root", "c1_profile", "a_eval_root", "c1_eval_root")]
    immutable += [Path(cfg["work"]) / "repo", Path(cfg["native_runner"]).parent,
                  Path(cfg["work"]) / "outputs/C0_C1_independent_screen_preview_v1"]
    if any(output == p or p in output.parents or output in p.parents for p in immutable):
        raise PilotError("Output must be separate from immutable inputs, source, caches and control results")
    cfg["config_path"] = str(path)
    return cfg

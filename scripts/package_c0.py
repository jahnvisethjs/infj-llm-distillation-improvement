"""Build a small, checksum-indexed source transfer bundle without data or weights."""

import argparse
import hashlib
import io
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capstone_person4.bindings import verify_release
from capstone_person4.config import load_config
from capstone_person4.provenance import PipelineError, canonical, write_bytes


def bundle(cfg, output):
    evidence = verify_release(cfg)
    output = Path(output).resolve()
    immutable = [Path(cfg["data"][key]) for key in ("release", "inherited_release", "inherited_encoder")]
    immutable.append(Path(cfg["inputs"]["parent_package"]))
    if any(output == root or root in output.parents for root in immutable):
        raise PipelineError("Source bundle must be outside immutable packages")
    files = {}
    patterns = ("capstone_person4/*.py", "capstone_person4_approachC1/*.py", "capstone_person4_approachC1/tests/*.py",
                "capstone_person4_approachC1/docs/*.md", "scripts/*.py", "scripts/sol/*.sh", "scripts/sol/*.sbatch",
                "scripts/sol/*.py", "tests/person4/*.py", "docs/C0*.md", "requirements-person4.txt")
    for pattern in patterns:
        for path in sorted(REPO.glob(pattern)):
            files["source/" + path.relative_to(REPO).as_posix()] = path.read_bytes()
    for path in (REPO / "configs").glob("*.json"):
        if ".local" not in path.name:
            files["source/configs/" + path.name] = path.read_bytes()
    for relative in ("README.md", "AGENTS.md", "requirements-person4.txt", "Person2_SOL_Seeds17_123_UpTo4000.ipynb"):
        files["source/" + relative] = (REPO / relative).read_bytes()
    inherited = Path(cfg["data"]["inherited_release"])
    files["inherited/metadata/release_manifest.json"] = (inherited / "metadata/release_manifest.json").read_bytes()
    files["inherited/configs/formatting_contract.json"] = (inherited / "configs/formatting_contract.json").read_bytes()
    encoder = Path(cfg["data"]["inherited_encoder"])
    for relative in evidence["encoder_files"]:
        files["inherited/code/" + relative] = (encoder / relative).read_bytes()
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}
    files["SHA256SUMS.json"] = (canonical(manifest) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    write_bytes(output, buffer.getvalue())
    return {"bundle": str(output), "files": len(files), "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "bytes": len(buffer.getvalue()), "includes_data_or_weights": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(canonical(bundle(load_config(args.config), args.output)))


if __name__ == "__main__":
    main()

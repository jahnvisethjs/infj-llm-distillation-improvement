"""Verify an extracted C0 source bundle before generating a SOL-local config."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Bundle path escapes root: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle checksum mismatch: {relative}")
    return {"verified_files": len(manifest), "root": str(root)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    print(json.dumps(verify(parser.parse_args().root)))

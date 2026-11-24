"""Build an additive, deterministic routing-only deployment ZIP; never overwrite."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def package(output):
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo / "capstone_routing").glob("*.py")) + [
        repo / "requirements-routing.txt", repo / "configs/routing.sol.example.json",
        repo / "docs/ROUTING_RUN_GUIDE.md", repo / "docs/ROUTING_PILOT_GUIDE.md",
        repo / "scripts/sol/routing_pilot_worker.sh", repo / "scripts/sol/routing_pilot_setup.sh",
        repo / "scripts/sol/routing_pilot_submit.sh", Path(__file__).resolve()]
    payload = {p.relative_to(repo).as_posix(): p.read_bytes() for p in files}
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(payload.items())}
    payload["ROUTING_SHA256SUMS.json"] = (json.dumps(checksums, indent=2, sort_keys=True) + "\n").encode()
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    target = Path(output).resolve()
    if target.suffix != ".zip":
        raise ValueError("Deployment output must end in .zip")
    data = content.getvalue()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("Existing deployment ZIP preserved; choose a new output name")
    else:
        with target.open("xb") as stream:
            stream.write(data)
    return {"path": str(target), "sha256": hashlib.sha256(data).hexdigest(), "source_files": len(checksums),
            "gpu_jobs_submitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))

"""Create a portable V4 configuration from explicitly supplied package locations."""

import argparse
import os
from pathlib import Path

from .config import PATH_FIELDS
from .provenance import PipelineError, read_json, sha, write_json


def configuration(release, inherited, parents, output, seed, notebook, tokenizer_assets=None, profile="0.5B"):
    release, inherited, parents = map(lambda p: Path(p).resolve(), (release, inherited, parents))
    repo = Path(__file__).resolve().parent.parent
    cfg = read_json(repo / "configs/c0.example.json")
    if profile not in cfg["profiles"]:
        raise PipelineError(f"Unknown model profile: {profile}")
    if type(seed) is not int or seed not in (17, 123):
        raise PipelineError("Select parent training seed 17 or 123")
    cfg["active_profile"] = profile
    cfg["output_root"] = str(Path(output).resolve())
    cfg["integration"] = {"notebook": str(Path(notebook).resolve()), "notebook_sha256": sha(notebook)}
    cfg["inputs"] = {"parent_package": str(parents), "parent_index": str(parents / "parent_index.json"),
                     "parent_index_sha256": sha(parents / "parent_index.json"),
                     "parent_checksums_sha256": sha(parents / "SHA256SUMS.json")}
    from .bindings import resolve_parent
    cfg["profiles"] = {name: resolve_parent({"inputs": cfg["inputs"], "parent": {
        "model": profile["model"], "revision": profile["revision"], "training_seed": seed}})
        for name, profile in cfg["profiles"].items()}
    cfg["data"].update(release=str(release), release_sha256=sha(release / "metadata/release_manifest.json"),
                       checksums_sha256=sha(release / "SHA256SUMS.txt"), inherited_release=str(inherited),
                       inherited_encoder=str(inherited / "code"), contract_root=str(inherited / "code"),
                       candidate_manifest=str(release / "manifests/selection_bank.jsonl"))
    for key in ("overlap_report", "overlap_sha256", "contract_root"):
        cfg["data"].pop(key)
    if tokenizer_assets:
        cfg["data"]["tokenizer_assets"] = str(Path(tokenizer_assets).resolve())
    sources = []
    for name in ("dev_gsm", "dev_math", "dev_fast", "test_gsm", "test_math500"):
        path = release / "inputs/evaluation/data" / (name + ".jsonl")
        sources.append({"name": name, "manifest": str(path), "sha256": sha(path),
                        "role": "fast-development" if name == "dev_fast" else "development" if name.startswith("dev_") else "final"})
    matcher = release / "code/capstone_person1/decontamination.py"
    checker = release / "inputs/overlap_evidence/overlap_check.py"
    reserved = release / "manifests/loss_dev.jsonl"
    cfg["protection"] = {"question_manifests": sources, "reserved_manifest": str(reserved), "reserved_sha256": sha(reserved),
                         "matcher_source": str(matcher), "matcher_sha256": sha(matcher),
                         "checker_source": str(checker), "checker_sha256": sha(checker), "reviews": []}
    cfg["allocation"].update(budget_kind="replay", expected_replay_tokens=318845)
    cfg["training"].update(learning_rate=1e-4, warmup_updates=10)
    cfg["sol"] = {"python": None, "account": None, "partition": None, "qos": None, "walltime": None,
                  "note": "Resolve on SOL; measure GPU smoke throughput before full requests"}
    return cfg


def relative_paths(value, base):
    """Keep generated configurations portable and relative to their own location."""
    base = Path(base).resolve()
    if isinstance(value, dict):
        return {k: os.path.relpath(v, base).replace(os.sep, "/") if k in PATH_FIELDS and isinstance(v, str)
                else relative_paths(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [relative_paths(v, base) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "inherited-release", "parent-package", "output-root", "config-output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--parent-seed", type=int, choices=(17, 123), default=17)
    parser.add_argument("--profile", choices=("0.5B", "1.5B"), default="0.5B")
    parser.add_argument("--notebook", default=str(Path(__file__).resolve().parent.parent / "Person2_SOL_Seeds17_123_UpTo4000.ipynb"))
    parser.add_argument("--tokenizer-assets")
    args = parser.parse_args()
    cfg = configuration(args.release, args.inherited_release, args.parent_package, args.output_root,
                        args.parent_seed, args.notebook, args.tokenizer_assets, args.profile)
    cfg = relative_paths(cfg, Path(args.config_output).resolve().parent)
    write_json(args.config_output, cfg)
    print(str(Path(args.config_output).resolve()))


if __name__ == "__main__":
    main()

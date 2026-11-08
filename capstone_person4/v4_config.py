"""Create a portable V4 configuration from explicitly supplied package locations."""

import argparse
import os
from pathlib import Path

from .config import PATH_FIELDS
from .provenance import PipelineError, read_json, sha, write_json


B_SELECT_ORIGIN_SHA256 = "5c9241231efc449fccbe464ce3198b87af20de8041aa699149f9c702f83bf7de"


def bind_b_bank(cfg, b_root):
    """Freeze Method B's bank, bins and selected manifest as the explicit common universe."""
    b_root = Path(b_root).resolve()
    bank = b_root / "shared/mixed_candidate_bank.jsonl"
    bins = b_root / "shared/parent_length_bins.json"
    selected = b_root / "mixed/budget_1x/1x/train_B.jsonl"
    for path in (bank, bins, selected):
        if not path.is_file():
            raise PipelineError(f"Method B artifact missing: {path}")
    cfg["data"].update(universe_mode="explicit", universe_manifest=str(bank), universe_sha256=sha(bank),
                       universe_origin="Method B main mixed bank (Suraj Shinde, method-b-document-alignment-v3): "
                                       "6,409 rows after reserving 1,000 development and 2,500 transfer rows",
                       length_bins=str(bins), length_bins_sha256=sha(bins), candidate_count=6409)
    cfg["allocation"].update(selector="method-b-cpsat-v3", b_select_origin_sha256=B_SELECT_ORIGIN_SHA256,
                             deterministic_limit=120, tie_limit=10, stratum_tolerance="0.05", utility_scale=1000000,
                             b_selected_manifest=str(selected), b_selected_sha256=sha(selected))
    return cfg


def configuration(release, inherited, parents, output, seed, notebook, tokenizer_assets=None, profile="0.5B", b_root=None):
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
    if b_root:
        bind_b_bank(cfg, b_root)
    bind_generation_contract(cfg, release / "inputs/evaluation/handoff/eval_contract.json")
    return cfg


def bind_generation_contract(cfg, contract_path):
    """Question-only generation for C1/E follows Person 3's frozen contract, executed by Person 3."""
    contract_path = Path(contract_path).resolve()
    if not contract_path.is_file():
        raise PipelineError(f"Evaluation contract missing: {contract_path}")
    contract = read_json(contract_path)
    settings = contract["frozen_generation_settings"]
    cfg["generation"] = {
        "generation_contract": str(contract_path), "contract_sha256": sha(contract_path),
        "contract_schema": contract["schema_version"], "prompt_template": contract["prompt_template"],
        "prompt_sha256": contract["prompt_template_sha256"], "precision": settings["precision"],
        "decoding": settings["decoding"], "max_new_tokens": settings["max_new_tokens"], "batch_size": settings["batch_size"],
        "eos_token_id": settings["eos_token_id"], "add_special_tokens": settings["add_special_tokens"],
        "question_text": settings["question_text"], "adapter_mode": "merged",
        "parser_version": contract["parser_version"], "checker": "math-verify " + contract["software_versions"]["math-verify"],
        "executor": "person3-external",
        "mistake_policy": "status incorrect only: parse_status ok and correct false; no_answer/unparseable/ambiguous/truncated are not mistakes"}
    cfg["selection"] = {"mistake_bonus": "1", "note": "lambda as an exact fraction string; C1 utility = round(fit*scale) * (den + num*w)"}
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
    parser.add_argument("--b-bank-root", help="Method B's B_method_B_document_v3 folder: binds its bank, bins and selection as the explicit universe")
    args = parser.parse_args()
    cfg = configuration(args.release, args.inherited_release, args.parent_package, args.output_root,
                        args.parent_seed, args.notebook, args.tokenizer_assets, args.profile, args.b_bank_root)
    cfg = relative_paths(cfg, Path(args.config_output).resolve().parent)
    write_json(args.config_output, cfg)
    print(str(Path(args.config_output).resolve()))


if __name__ == "__main__":
    main()

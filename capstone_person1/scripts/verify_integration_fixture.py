"""Read-only validation of the packaged diagnostic fixture and reference labels."""
import json
from pathlib import Path

from capstone_person1.person1_data_contract import (
    PINNED_TOKENIZER_REVISION, TOKENIZER_ID, validate_manifest,
)
from capstone_person1.pipeline_controls import file_sha256
from capstone_person1.tokenizer_adapter import SnapshotTokenizer


def main():
    project = Path(__file__).resolve().parents[1]
    root = project / "diagnostics/integration_smoke"
    path = root / "debug_32_pilot.jsonl"
    summary = json.loads((root / "debug_32_pilot_summary.json").read_text(encoding="utf-8"))
    formatting = json.loads((project / "configs/formatting_contract.json").read_text(encoding="utf-8"))
    assets = project / "cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots" / PINNED_TOKENIZER_REVISION
    for name, digest in formatting["tokenizer_asset_sha256"].items():
        if file_sha256(assets / name) != digest:
            raise ValueError("Pinned tokenizer asset hash mismatch: " + name)
    tokenizer = SnapshotTokenizer.from_pretrained(TOKENIZER_ID, revision=PINNED_TOKENIZER_REVISION)
    result = validate_manifest(path, {
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": PINNED_TOKENIZER_REVISION,
        "manifest_sha256": summary["sha256"],
        "rows": summary["rows"],
        "supervised_tokens": summary["total_supervised_tokens"],
    }, tokenizer)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids_path = root / "debug_32_pilot_ids.txt"
    if file_sha256(ids_path) != summary["ids_file_sha256"]:
        raise ValueError("Fixture ID file hash mismatch")
    if ids_path.read_text(encoding="utf-8").splitlines() != [row["row_id"] for row in records]:
        raise ValueError("Fixture ID order mismatch")
    ready = json.loads((root / "handoff_readiness.json").read_text(encoding="utf-8"))
    if ready["research_ready"] is not False:
        raise ValueError("Diagnostic fixture must not claim research readiness")
    print(json.dumps(dict(result, status="DIAGNOSTIC_ONLY", research_ready=False,
                          tokenizer_backend="SnapshotTokenizer; pinned assets hash-checked"), indent=2))


if __name__ == "__main__":
    main()

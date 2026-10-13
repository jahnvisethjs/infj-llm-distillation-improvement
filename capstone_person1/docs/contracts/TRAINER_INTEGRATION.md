# Person 2 integration inputs

The data package can be `DATA_READY_FOR_TRAINER` before these inputs arrive. The following are evidence from actual training, supplied by Person 2. Templates, reference labels and planned seen-group lists do not establish completed training.

## Label reconciliation

Export CSV with `row_id,actual_supervised_tokens`. Submit one complete scope at a time: all 32 IDs from `debug_32`, then all 1,000 IDs from `train_n1000`. Counts come from actual collator labels: `sum(label != -100 for label in labels)`. IDs must be unique and exactly cover the corresponding manifest. Empty/partial reports, unexpected or duplicate IDs, nonpositive/noninteger counts and unequal counts fail.

Supply accompanying JSON:

```json
{
  "trainer_version": "your actual code/config version",
  "contract_version": "person1-plain-text-labels-v3",
  "manifest_sha256": "the actual corresponding release manifest SHA-256",
  "prompt_mask_verified": true,
  "eos_label_verified": true,
  "padding_by_position_verified": true,
  "confirmed_by": "actual verifier identity",
  "confirmed_utc": "actual ISO-8601 timestamp with timezone"
}
```

The three booleans attest to actual inspected trainer behavior: prompt labels are -100 while attention remains 1; the genuine appended EOS remains supervised; padding is masked by position/attention, preserving EOS when padding and EOS share ID 151643.

```powershell
python -m capstone_person1.integration_checks tokens --release RELEASE --scope debug_32 --actual ACTUAL.csv --metadata ACTUAL.json --output RESULT_OUTSIDE_RELEASE.json
```

Repeat with `--scope train_n1000` before the parent run. Keep integration results outside the frozen release.

## Actual parent binding

Supply the exact JSONL training manifest used for the actual parent, CSV of unique `problem_group_id` values seen during training, and JSON with these fields:

```json
{
  "actual_run_id": "the actual training run ID",
  "checkpoint_id": "reloadable actual checkpoint path or immutable reference",
  "checkpoint_sha256": "actual 64-character SHA-256",
  "actual_manifest_sha256": "actual training-manifest SHA-256",
  "confirmed_by": "actual verifier identity",
  "confirmed_utc": "actual ISO-8601 timestamp with timezone"
}
```

```powershell
python -m capstone_person1.integration_checks parent --release RELEASE --actual-manifest ACTUAL.jsonl --seen ACTUAL_SEEN.csv --metadata ACTUAL_PARENT.json --output RESULT_OUTSIDE_RELEASE.json
```

The checker requires the actual manifest hash to match the corrected planned R1000 manifest, exact seen-group coverage, and disjointness from the selection bank. Training outside that manifest requires coordinated freshness revalidation and a new version. A checkpoint digest identifies the supplied checkpoint; Person 2 still owns executing and documenting save/reload verification.

## Readiness ownership

| State | Additional evidence |
|---|---|
| `DATA_READY_FOR_TRAINER` | Person 1 full, reviewed, validated package |
| `MAIN_RUN_READY` | Actual trainer-label verification and Person 3 frozen split/evaluator contract |
| `CONTINUATION_READY` | Actual parent binding and verified selection-bank freshness |

Person 2 runs the QLoRA smoke test and independent learning-curve training. Person 3 creates/freezes its own development and final evaluation definitions and validates generation/scoring. Person 4 owns student-fit scoring, budget-matched continuation arms and research analysis. Person 1 supplies the common prompt/encoding contract, manifests, group mapping and detected-overlap reports.

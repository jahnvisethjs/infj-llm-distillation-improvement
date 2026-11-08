# Running C0 reference-fit selection

V4 implementation update, 5 November 2026: use the [V4 run guide](C0_V4_RUN_GUIDE.md) for the supplied data and R1000 parents. V4 inheritance, protection, stage-specific dependencies and approach-only execution are implemented. The commands and release-local encoder requirements below describe the earlier configuration format; the V4 guide supersedes them for V4 runs.

The implementation is in [capstone_person4](../capstone_person4/). It reuses Person 2's model loader, encoder wrapper, training update loop, checkpoint saving and reload behavior through a source-hashed notebook adapter. Importing it never runs notebook setup, downloads, historical training or sweeps.

The comparison is untouched parent versus fresh random continuation (B) versus reference-fit continuation (C0). C0 scores the supplied reference completion; it does not use independent answer correctness. Lower reference loss is not a guarantee of better held-out accuracy. See the [implementation plan](C0_IMPLEMENTATION_PLAN.md).

## Start here in the current Windows workspace

Run from the repository root. The shell's default `python` points to a missing environment. The installed Anaconda executable works for the model-independent and CPU tensor checks:

```powershell
& 'C:/softwares/pythonAnaconda/python.exe' -B -m capstone_person4 --help
& 'C:/softwares/pythonAnaconda/python.exe' -B -m capstone_person4 dry-run --output outputs/person4/synthetic-demo-current
& 'C:/softwares/pythonAnaconda/python.exe' -B -m capstone_person4 audit --config configs/c0.example.json
```

`dry-run` writes explicitly synthetic scores/selections. It cannot satisfy real stage gates. `audit` writes both readiness reports, an environment record and an inventory of saved training/release handoffs. **Exit code 2 is expected while inputs are unresolved.** Inventory entries and historical readiness declarations are leads, not new verification.

Run the focused tests with:

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
& 'C:/softwares/pythonAnaconda/python.exe' -B scripts/validate_c0.py
```

This Anaconda installation needed the process-local MKL setting to avoid its observed duplicate OpenMP initialization error during CPU PyTorch imports. The validation script retains logs and a machine-readable result under `.test_artifacts/verification/`. Its synthetic CPU training test exercises Person 2's actual update loop; it does not train or certify a real Qwen parent.

For real model work, use Person 2's compatible isolated NVIDIA/CUDA environment. [requirements-person4.txt](../requirements-person4.txt) preserves the notebook's proposed package versions; choose a driver-compatible PyTorch separately. No model dependencies were installed or changed by this implementation.

## Resolve an experiment configuration

Copy [c0.example.json](../configs/c0.example.json) to a local configuration and fill actual inputs. All paths are relative to the configuration file, not the shell's working directory. `output_root` must be outside the frozen release. Outputs are separated by model profile and resolved configuration fingerprint.

The notebook path and SHA256 are populated from the supplied repository. Other missing inputs remain `null`. Resolve these from existing handoffs before running dependent stages:

| Input | Required content |
| --- | --- |
| Parent adapter directory | Matching `adapter_model.safetensors` and `adapter_config.json`, with hashes |
| Parent run evidence | `run_config.json`, `training_summary.json`, their hashes and actual run ID |
| Complete exposure evidence | Every fine-tuning exposure manifest, its hash and epochs, bound to the run, summary and adapter |
| Immutable Person 1 release | Release-manifest hash, canonical selection JSONL, verified formatting contract and `code/` root |
| Frozen question manifests | JSONL with unique `problem_id`, `benchmark`, `problem`, and `final_answer` or `answer`; optional mapped `problem_group_id` |
| Overlap report | Exact candidate/release/dev/final hashes, excluded groups, controls and limitations |
| Continuation recipe | Explicit learning rate, warmup, seed and supported batch/precision settings |
| Evaluation contract/source | Frozen generation/checker policy, source dependencies and callable adapter |

[c0.inputs.template.json](../configs/c0.inputs.template.json) contains unresolved schemas for the evidence files. Split them into separate files and replace nulls with actual evidence. A planned manifest, a filename such as R500, or an adapter path copied from an old evaluation config is insufficient by itself.

Parent exposure paths inside an evidence file are relative to that evidence file. Its `exposures` must include the manifest matching the parent run's `manifest_sha256`; summary seen IDs/groups and row/epoch counts must agree. Earlier fine-tuning manifests belong in the same history. The replay anchor uses all unique verified parent rows once, distinguishing historical repeats.

The release candidate manifest and encoder code must be inside the configured immutable release and covered by its recorded file hashes. Do not combine the v3 workspace data with a seeds-17/123 checkpoint from a different release merely because filenames look similar.

The overlap report binds both frozen question manifests and the candidate/release hashes. Its excluded groups must come from the project's actual exact/near-duplicate controls. An empty exclusion list is valid only when those controls support it; the list is not a substitute for running or recovering the controls. Development and final questions/groups must be disjoint. Dataset references remain mathematically unverified corpus-wide.

## Stage sequence

The commands below assume `python` resolves to the chosen compatible execution environment. On this Windows machine, substitute the explicit executable shown above. `<...>` denotes an input you must resolve; it is not a literal runnable argument.

```bash
python -m capstone_person4 audit --config configs/c0.local.json
python -m capstone_person4 prepare --config configs/c0.local.json
python -m capstone_person4 score-reference --config configs/c0.local.json --smoke
```

Preparation re-encodes the frozen candidates with the actual tokenizer and enforces whole-example context limits. It does not load model weights. The real scoring smoke checks padded versus unbatched token losses and unchanged parent parameters before allowing the full reference pass.

Execute the full scoring pass as a separate experiment stage:

```bash
python -m capstone_person4 score-reference --config configs/c0.local.json
python -m capstone_person4 diagnose --config configs/c0.local.json
python -m capstone_person4 select --config configs/c0.local.json --method B
python -m capstone_person4 select --config configs/c0.local.json --method C0
```

Default preparation requests 5,000 eligible fresh examples. Any shortfall is explicit. Reference caches retain raw nats, per-token losses/positions/IDs, parent/render fingerprints and complete coverage evidence. Interrupted scoring can resume valid per-row cache entries. A different parent, text, tokenizer, mask, precision, source version or runtime invalidates affected measurements. Cache validation during model-independent selection uses the recorded scoring runtime, so reading a GPU-produced cache on a CPU host does not pretend to rescore it.

Per-token cache entries and matching scoring-smoke proofs are shared under `<output_root>/reference_cache/<profile>/<measurement-key>/` (proofs use a separate `smoke/` directory). Changing only the continuation seed can reuse verified reference measurements. Each resolved experiment still writes its own complete coverage index and reports fresh rows versus cache hits separately.

The allocator uses the same frozen strata, largest-remainder quotas, rank-based priority policy and whole-example budget logic for B and C0. It checks every row-count prefix efficiently, then performs bounded swaps only if no prefix meets the tolerance. Defaults: 0.5% budget tolerance, 16 boundary rows, 32 swaps and 128 row-count swap searches. Exhausting this search is reported separately from proven capacity infeasibility. It does not widen tolerance, relax quotas or repeat fresh rows automatically.

Choose `allocation.budget_kind`:

- `replay`: derive the target by re-encoding each verified unique parent row once.
- `pilot`: require explicit integer `target_tokens`; outputs label the anchor provisional.

Default stratum proportions come from the actual parent rows. If candidate capacities cannot support them, selection fails and requests an explicit common alternative. Source row proportions and source token shares are reported separately. The actual selected row count can differ between B and C0.

Before full continuation, run a disposable tiny training smoke:

```bash
python -m capstone_person4 train --config configs/c0.local.json --method C0 --attempt smoke01 --smoke
```

Its manifest contains at most four selected rows. It writes only to `training_smoke/` and can never be labeled a controlled full arm. A successful smoke with matching parent/configuration/code is required before full continuation:

```bash
python -m capstone_person4 train --config configs/c0.local.json --method B --attempt pilot01
python -m capstone_person4 train --config configs/c0.local.json --method C0 --attempt pilot01
```

Each call loads the same trained adapter and creates a fresh AdamW/scaler/schedule. Supported recipe: one epoch, microbatch one, no packing, token-weighted accumulation, linear warmup followed by constant learning rate. The adapter preserves the notebook's actual update loop. It changes parent initialization and continuation schedule explicitly; historical notebook behavior remains unchanged.

Training includes the final incomplete accumulation group. Completion requires exact reconciliation of consumed tokens, selected IDs/groups and the manifest schedule, and unchanged parent artifact bytes. A completed run writes `token_reconciliation.json`; an unreconciled checkpoint is not eligible for controlled evaluation.

To recover an interrupted continuation, supply its own periodic checkpoint and a new attempt name:

```bash
python -m capstone_person4 train --config configs/c0.local.json --method C0 --attempt pilot01_resume01 --resume <interrupted-arm/checkpoint-000100>
```

Only that arm's own resume restores optimizer/RNG state. Parent optimizer state is never restored when starting B or C0.

## Person 3 evaluation adapter

Recover the existing evaluator and checker source. The interface accepts an explicit adapter around that implementation, rather than guessing CLI flags or recreating its mathematical checker.

Configure a Python file exporting:

```python
def evaluate(request):
    # Call the existing Person 3 evaluator and translate its results.
    # request['questions'] contains only problem_id and problem.
    # request['references'] is separate gold data for checking, never a prompt.
    # request also supplies model/revision, adapter path/hash and frozen contract.
    return predictions
```

Each prediction must provide `problem_id`, `generated_text`, `extracted_answer`, `generated_tokens`, `stop_reason`, `flags`, boolean `correct`, and a status from `correct`, `incorrect`, `no_answer`, `truncated`, `unresolved_equivalence`, `execution_error`. Preserve additional flags, including abnormal stopping after an extracted correct answer. Retry execution errors before comparison. Unsupported equivalence remains explicit even when overall benchmark accuracy counts the answer as unsuccessful.

The contract must use `adapter_interface: person4-evaluate-v1` and bind greedy decoding, output cap, stopping, precision, prompt hash and checker version. Record actual evaluator/checker source files in `evaluator_dependencies` as `{path, sha256}` entries; relative paths resolve against the contract file. Supply hash-bound `checker_validation` evidence recording passed fixtures, supported answer forms, checked-case count, the same version/source dependency list, and unresolved ambiguity handling. The input template includes that schema. The adapter can also export `runtime_metadata()` to report actual evaluation hardware and memory. Caller wall time and generated token totals are retained separately.

```bash
python -m capstone_person4 evaluate --config configs/c0.local.json --method parent --attempt dev01
python -m capstone_person4 evaluate --config configs/c0.local.json --method B --run <B-run-directory> --attempt dev01
python -m capstone_person4 evaluate --config configs/c0.local.json --method C0 --run <C0-run-directory> --attempt dev01
python -m capstone_person4 compare --config configs/c0.local.json --attempt dev01 --reports <parent-evaluation-report.json> <B-evaluation-report.json> <C0-evaluation-report.json>
```

Development is the default split. For final evaluation, supply a hash-bound method-freeze file with `frozen: true` and the actual `method_definition(cfg)` from `capstone_person4.evaluation`, then use `--split final`. The definition includes selection/scoring/training and evaluator/checker fingerprints, while allowing training seed replication.

Comparison requires the exact shared questions, parent, protocol and method. It reports GSM/MATH separately, C0–B and parent contrasts, paired bootstrap intervals, extraction/truncation diagnostics and evaluation costs. Repeat with configurations differing only in `training.seed`; provide matched B/C0 reports plus one parent report to compute separate training-seed variability. Changing selection/data seeds is a different experiment.

## Artifact and recovery behavior

Reports, selected manifests, reference caches and runtime observations live under:

```text
<output_root>/<model-profile>/<config-fingerprint-prefix>/
```

Scientific artifacts are content-bound and additive. Identical writes are idempotent; conflicting existing outputs fail and request a new directory/attempt. Atomic writes retain a `.partial` hard link after committing the canonical filename, avoiding automatic file deletion. Consumers read only canonical filenames and validated coverage records, never partial files. Current commands support a single writer per artifact root.

No delete/cleanup command is part of this workflow. Existing Person 1 folders, original notebooks and releases are preserved. Test logs and ignored runtime outputs are intentionally retained.

The next ready stage in this workspace is `audit` or the synthetic dry run. Real parent scoring/training awaits a matching saved adapter, checkpoint-bound exposure/release evidence and the appropriate GPU environment. Evaluation additionally awaits the frozen Person 3 source/manifests/contract.

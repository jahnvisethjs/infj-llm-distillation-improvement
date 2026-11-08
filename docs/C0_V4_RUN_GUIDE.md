# V4 reference-fit continuation

For the configured class account, stage `.sbatch` files, copy/paste submission commands and the 960 GPU-minute calculation, use the [ASU SOL sbatch guide](C0_SOL_SBATCH_GUIDE.md).

The [V4 implementation plan](C0_V4_IMPLEMENTATION_PLAN.md) defines the experiment. The first configuration selects the Qwen2.5-0.5B R1000 parent trained with seed 17; candidate, selection and continuation seeds are independently set to 42. Seed 123 is a separate parent replication, never an automatic second job.

## Inputs and stages

Use [c0.v4.seed17.json](../configs/c0.v4.seed17.json) or [c0.v4.seed123.json](../configs/c0.v4.seed123.json). Paths are relative to each configuration. The selected parent index, package checksums, V4 checksums and inherited release metadata are pinned. Parent adapter paths and hashes are resolved from exactly the requested model/revision/training seed. Saved label evidence is historical until preparation re-encodes the actual rows.

Four explicit configurations select each requested model and parent seed without an override:

| Model | Parent training seed | Configuration |
| --- | ---: | --- |
| Qwen2.5-0.5B | 17 | [c0.v4.0.5b.seed17.json](../configs/c0.v4.0.5b.seed17.json) |
| Qwen2.5-0.5B | 123 | [c0.v4.0.5b.seed123.json](../configs/c0.v4.0.5b.seed123.json) |
| Qwen2.5-1.5B | 17 | [c0.v4.1.5b.seed17.json](../configs/c0.v4.1.5b.seed17.json) |
| Qwen2.5-1.5B | 123 | [c0.v4.1.5b.seed123.json](../configs/c0.v4.1.5b.seed123.json) |

All four retain 5,000 candidate rows and candidate/selection/continuation seeds of 42. Each model/parent combination gets its own measurements, selections, smoke evidence, checkpoints and handoff. Each requires its own real-model smoke checks. The earlier seed-only filenames remain compatible and default to 0.5B. No command automatically launches the four combinations.

For example, select the 1.5B seed-123 parent explicitly:

```bash
python -m capstone_person4 audit --config configs/c0.v4.1.5b.seed123.json --target inputs
python -m capstone_person4 prepare --config configs/c0.v4.1.5b.seed123.json
```

Preparation for 1.5B retains the inherited guard comparing actual token IDs, labels and attention masks against the pinned 0.5B tokenizer. Equal counts alone are insufficient; a mismatch stops the run. The pinned 0.5B reference tokenizer is read from the checksummed parent package (`qwen2_5_0_5b/R1000_seed<N>/final/tokenizer`), so no Hub download is needed. The local dependency mismatch still blocks actual token verification for both model sizes.

The inherited encoder can live outside V4. All Python files under its package are checked against the original release metadata; the parent run's reference-code hashes must also match. Original `/Users/suraj/...` paths are provenance, not runtime requirements. Original inputs remain immutable.

Protection uses all five frozen question files separately, the actual parent exposure, `loss_dev`, and internal candidate relationships. The supplied enhanced normalization, five-gram retrieval and thresholds are reused as hash-verified definitions without executing source-data rebuilds. EXACT/LIKELY_SAME external matches exclude groups; unresolved similarity and internal near matches quarantine affected rows/groups. Normalized exact duplicates retain a stable eligible row-ID representative. Reviews must bind the exact pair, texts, scope and rule fingerprint; unrelated reviews cannot admit a pair. Matcher recall remains approximate.

Orca's canonical `synthetic` field records upstream source provenance. It remains unchanged. Reports identify generated test fixtures separately from actual V4 input preparation.

## Local CPU preparation

From the repository root, use the existing Windows environment:

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
& 'C:/softwares/pythonAnaconda/python.exe' -B scripts/validate_c0.py
& 'C:/softwares/pythonAnaconda/python.exe' -B -m capstone_person4 audit --config configs/c0.v4.seed17.json --target inputs
& 'C:/softwares/pythonAnaconda/python.exe' -B -m capstone_person4 protect --config configs/c0.v4.seed17.json
& 'C:/softwares/pythonAnaconda/python.exe' -B -m capstone_person4 prepare --config configs/c0.v4.seed17.json
```

`prepare` runs protection if matching evidence is absent. It requires exactly 5,000 candidates with 250 rows in each inherited source/bin stratum. It reports exact deficits and stops if capacity is insufficient. It verifies all candidate and parent token IDs, masks, completion boundaries and EOS with the inherited encoder and the saved parent tokenizer. The default replay anchor must reconcile to 318,845 supervised tokens. No model weights are loaded for preparation.

`audit --target inputs` checks file/preparation readiness. `--target approach` also requires GPU scoring and training dependencies; `--target evaluation` and `--target all` include deferred evaluator readiness. All reports retain visible deferred checks. Missing protected question inputs block preparation; missing answer-checker software does not.

## Artifact locations and reuse

Outputs live under `<output_root>/<profile>/<stage-fingerprint-prefix>/`. Preparation, scoring, selection and training use their own dependency fingerprints, so their directories can differ. Python's `artifact_root(cfg, stage)` resolves each one. The complete configuration snapshot is retained for audit; evaluator/checker settings do not invalidate earlier stages. Changing parent weights or scoring precision invalidates the corresponding measurements. Changing only the continuation seed can reuse prepared candidates, scores and selections.

- `input_bindings/`: release, parent, inherited encoder and configuration snapshots.
- `protection/`: eligible rows, exclusions, quarantines, duplicate clusters, flagged-pair JSONL/CSV, review queue and summary with artifact hashes.
- `prepared/`: canonical `candidates.jsonl`, ordered IDs, per-row token-verification CSV and `preparation.json`.
- `reference/`: scoring smoke, full score index, coverage and runtime observations. Compressed per-token records are shared under `<output_root>/reference_cache/`.
- `selection/B/` and `selection/C0/`: trainer manifests, ordered IDs, audit tables, allocation reports and pairwise token difference.
- `training_smoke/`: disposable attempts of at most four rows. `runs/`: independent full-arm attempts and token reconciliation.
- `handoffs/<arm-pair-fingerprint>/`: `approach_completion.json` and `evaluation_handoff.json`.

Outputs are additive. Conflicting artifacts and old attempts are preserved. No launcher performs cleanup.

## Transfer and SOL configuration

Build the small source bundle locally:

```powershell
& 'C:/softwares/pythonAnaconda/python.exe' -B scripts/package_c0.py --config configs/c0.v4.seed17.json --output outputs/person4/bundles/c0-v4-models-seeds.zip
```

It contains current source, tests, configs, documentation, the original notebook and hash-bound inherited encoder modules. It excludes datasets, weights and local caches. Transfer the V4 release and supplied parent package separately. After extracting the source bundle, verify it before editing or generating local configuration:

```bash
python source/scripts/verify_c0_bundle.py --root /actual/extracted/bundle
```

From the extracted `source/` directory, generate a new SOL configuration with actual accessible paths:

```bash
python -m capstone_person4.v4_config \
  --release /actual/v4/data \
  --inherited-release /actual/extracted/bundle/inherited \
  --parent-package /actual/parent/package \
  --output-root /actual/results/c0-v4 \
  --parent-seed 17 --config-output configs/c0.local.sol.json
export C0_PYTHON=/actual/compatible/environment/bin/python
bash scripts/sol/preflight.sh configs/c0.local.sol.json
bash scripts/sol/prepare.sh configs/c0.local.sol.json
```

To generate another combination, supply `--profile 1.5B --parent-seed 123` (or the other requested model/seed pair) and a distinct `--config-output`. This selects the active profile while retaining the pinned definitions for both model sizes.

These `/actual/...` placeholders must be replaced on SOL. [requirements-person4.txt](../requirements-person4.txt) records the intended isolated model stack; select a driver-compatible CUDA PyTorch. Preparation requires tokenizer dependencies; model stages additionally require PEFT, Accelerate, bitsandbytes and an allocated compatible NVIDIA GPU.

## Explicit GPU stages

Obtain separate execution authorization before real GPU experiments. Within a short allocation, run the eight-row scoring smoke:

```bash
bash scripts/sol/gpu_stage.sh configs/c0.local.sol.json scoring-smoke
```

Inspect finite per-token losses, padded/unbatched agreement (declared `scoring.atol`/`rtol` of 0.05 nats absorbs bfloat16 kernel noise; the report records the largest observed difference), causal counts, unchanged parent state, GPU memory and measured throughput. Then size full-job requests. A full `score-reference` run performs this smoke itself when no proof exists for its exact GPU/runtime. The submit script requires explicit `C0_ACCOUNT`, `C0_PARTITION`, `C0_WALLTIME`, `C0_MEMORY` and `C0_GRES`; optional `C0_QOS` is passed only when supplied. It does not infer another author's account or submit downstream jobs automatically.

For batch execution, `bash scripts/sol/submit.sh CONFIG STAGE ATTEMPT PREREQUISITE_JOB_ID` submits one job. Supported stages are `scoring-smoke`, `score-reference`, `training-smoke`, `train-B` and `train-C0`. All stages after the first require an `afterok` prerequisite ID and validate their own artifacts. A scoring smoke run interactively can be followed by a scoring batch stage invoked through `sbatch` with the same explicit resources; the in-process smoke gate still applies.

Launchers default to the configuration's active profile. An optional profile argument is propagated consistently:

```bash
bash scripts/sol/preflight.sh configs/c0.v4.seed123.json 1.5B
bash scripts/sol/prepare.sh configs/c0.v4.seed123.json 1.5B
bash scripts/sol/gpu_stage.sh configs/c0.v4.seed123.json scoring-smoke "" 1.5B
bash scripts/sol/submit.sh configs/c0.v4.seed123.json train-B pilot01 PREREQUISITE_JOB_ID 1.5B
```

The parent seed comes from the configuration, independently of the profile argument. Invalid profiles fail before model loading or job submission. Slurm job/log names include the configuration name and any profile override, so the four combinations can be distinguished.

After full scoring succeeds, select both arms on CPU:

```bash
"$C0_PYTHON" -B -m capstone_person4 diagnose --config configs/c0.local.sol.json
"$C0_PYTHON" -B -m capstone_person4 select --config configs/c0.local.sol.json --method B
"$C0_PYTHON" -B -m capstone_person4 select --config configs/c0.local.sol.json --method C0
```

Both arms use the common frozen pool, shared allocator, inherited parent proportions and 0.5% token tolerance. Each must total 317,251–320,439 supervised tokens. The pairwise difference is reported; row counts may differ. Allocator infeasibility or exhausted search stops explicitly.

Run a disposable training smoke, then independent full arms:

```bash
bash scripts/sol/gpu_stage.sh configs/c0.local.sol.json training-smoke smoke01
bash scripts/sol/gpu_stage.sh configs/c0.local.sol.json train-B pilot01
bash scripts/sol/gpu_stage.sh configs/c0.local.sol.json train-C0 pilot01
```

Each arm reloads the original trained adapter and starts fresh AdamW with learning rate `1e-4`, ten warmup updates, accumulation eight, one pass and no packing. The final partial accumulation group is retained. A matching successful training smoke is required. Exact consumed tokens, selected identities, changed continued weights and unchanged parent bytes must reconcile before an arm is valid. Resume only the interrupted arm's own checkpoint into a new attempt with `train --resume ...`; another arm always starts fresh.

## Final handoff

After both actual runs complete:

```bash
"$C0_PYTHON" -B -m capstone_person4 finalize --config configs/c0.local.sol.json \
  --b-run /actual/B/pilot01 --c0-run /actual/C0/pilot01
```

Finalization checks the saved summaries, selected exposure, checkpoint hashes, parent identity and full reference coverage. It emits completion and evaluation-handoff records with `evaluation_status: pending`. Preparation or synthetic validation alone cannot satisfy this gate. Benchmark generation, mathematical grading and accuracy comparisons remain a later phase.

# Latest screened B, C0 and C1: 4,955 eligible candidates

This export contains Suraj Shinde's completed pool-matched development matrix:
**12 B + 12 C0 + 12 C1 full continuations**, with GSM and MATH development
evaluation for every checkpoint. B means the new **screened B control** here.

Every current run records bank SHA-256:
`ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f`.

The eligible bank contains **4,955 rows** (3,313 Orca + 1,642 Numina).
Each model trains **1,000 selected examples**, not all 4,955: twenty source/length
strata, fifty rows each; target 318,845 supervised tokens with the existing
0.5% global and 5% stratum tolerances. Screened B uses 317,486 actual tokens.

## Folders

* `B/`: current screened B training/evaluation source, cleaned combined notebook,
  twelve complete training receipts/configs/summaries, selected-ID manifests,
  and twenty-four GSM/MATH result files with compressed predictions.
* `C0/`: C0 source and notebooks, twelve complete run records and selections,
  and twenty-four GSM/MATH result files.
* `C1/`: C1 source/notebooks, twelve run records and selections, twenty-four
  result files, and the four frozen C1 diagnostic reports.
* `Shared/`: original reused B trainer/evaluator/encoder source, C0/C1 selection
  modules, screened pilot and screening protocol, bank/parent/question identity
  metadata, and the publishing helper.
* `Comparison/`: latest screened-only CSVs, 12 figures, exposure/uncertainty
  notes and audit summary. Historical original-B predictions are excluded.
* `EXPORT_MANIFEST.json`: exact exported-file hashes, original sources and
  transformations. It is a publication receipt, not a new experiment.

Each `evaluation/` checkpoint contains `dev_gsm/` and `dev_math/`. In each are
the native receipt, metrics, generation config and `predictions.jsonl.gz`.
Decompress with Python's `gzip.open(...)` or `gzip -dc`; the native receipt's
prediction hash refers to the **uncompressed** JSONL bytes. Compression is
lossless and deterministic. Notebook code is unchanged; outputs/widget state
and execution counts are stripped only from copies under `notebooks/`.
Frozen package files, including a checksum-bound pilot notebook where present,
are copied without modifying their bytes.

## Main results (means over six checkpoints per model size)

| Model | Method | GSM % | MATH % | Equal-weight mean % |
|---|---|---:|---:|---:|
| 0.5B | B screened | 52.90 | 26.90 | 39.90 |
| 0.5B | C0 | 53.60 | 27.83 | 40.72 |
| 0.5B | C1 | 54.30 | 27.87 | 41.08 |
| 1.5B | B screened | 81.43 | 50.87 | 66.15 |
| 1.5B | C0 | 82.40 | 52.27 | 67.33 |
| 1.5B | C1 | 83.10 | 51.63 | 67.37 |

C1 has the highest observed means, but C0 is close. All six aggregate paired
question-bootstrap intervals include zero; this is not a decisive overall
winner claim. Parent baselines and limitations matter: at 0.5B all continuation
means are below the average R1000 parent. See `Comparison/README.md`.

## What remains external

This is a small code/results handoff, not a full model or raw-data backup.
Full candidate/reference text, native training JSONL, adapter/base weights,
optimizer states, GPU caches and virtual environments remain on Sol. The export
is capped at 5 MiB per file and 60 MiB total. Full bank and development question
text are omitted; IDs, source/token metadata and native fingerprints are kept.

Original scientific roots (read-only during export):

* `/scratch/sshind38/capstone_c0_c1_work`
* `/scratch/sshind38/capstone_approach_b`

The notebooks use those existing Sol roots and the working training/solver
environments. Restore the original external inputs and expected directory
layout before attempting training from another machine. Moving frozen source
files or editing hashes is not a substitute for restoring their dependencies.

The source snapshots retain historical hashes and original code paths where
required by the verified wrappers. In particular, the reused runtime has
lineage checks against original B inputs. Those references do **not** mean the
published current continuations used the old eligible bank. Current run configs
record the screened bank; `training/` receipts, selected IDs and evaluation
sidecars provide the direct evidence. Original B model results are not copied.

## Evaluation scope and integrity

Both frozen development sets have 500 questions. Greedy fp32 generation,
batch 16, maximum 1,024 new tokens, merged adapters and p3-parser-v6 are shared.
Parent seeds 17/123 and continuation seeds 42/137/2026 have different roles.
The completed screened B / 0.5B / parent17 / continuation42 pilot is reused.

Reserved final tests are not included or run. Inherited parent/development
exposure remains disclosed. The original 25-exclusion/572-quarantine audit was
unresolved in the reviewed copies; the independent conservative screen is not
a certification that contamination is absent. C1 remains the frozen exploratory
refinement with its original diagnostic gate. No results are recomputed by the
publisher; hashes/coverage and the bank identity are checked before copying.

## Source result locations

Within `/scratch/sshind38/capstone_c0_c1_work`:

* B training: `outputs/B_screened_4955_controls_v1/00d5c71ab844b586`
* B dev: `outputs/B_screened_4955_development_evaluation_v1/3338815c65ed8f68`
* C0/C1 training: `outputs/C0_C1_overnight_training_v1/7376bbd91d80e726`
* C0/C1 dev: `outputs/C0_C1_development_evaluation_v1/931e8fb821c7d70f`

Original absolute paths inside native receipts are preserved as provenance.
The GitHub export folders are a new organization; native receipt hashes and
claims have not been rewritten to pretend those paths are portable.

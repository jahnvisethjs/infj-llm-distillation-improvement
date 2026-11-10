# Method B: fresh stratified random continuation

This folder exports the small Method B source files from Suraj Shinde's Sol
workspace. Code is copied without editing its contents. Exported notebooks have
outputs, execution counts, embedded attachments and widget state removed from
the copy; the original notebooks on Sol stay unchanged.

## Layout

- `method_b_document_v3_code/`: documented Method B selector, execution,
  recovery and unattended-run helpers.
- `b_workbench_500/` or `b_workbench/`: available original training wrappers,
  tests, vendored Python encoder/trainer/evaluator source, and provenance hashes.
- Top-level notebooks and documentation: included when present and small enough.
- `results/`: small canonical development summaries, when available.
- `EXPORT_MANIFEST.json`: exact copied-file hashes, source paths, transformations,
  and skipped files. It is an export receipt, not a new experiment result.

## External inputs and environment

Full training/evaluation needs the original verified parent adapters, manifests,
protected question files, historical parent/reference records, model cache and
runtime configuration. Those inputs remain on Sol at:

`/scratch/sshind38/capstone_approach_b`

Data files, virtual environments, models, optimizer checkpoints, raw prediction
JSONL, machine-local `config.json`/`run_config.json`, and archives are excluded.
Files larger than 5 MiB are skipped. The whole export is capped at 50 MiB.
Notebook inputs are read only up to 64 MiB before stripping output; larger
notebooks are listed as skipped rather than silently claimed as included.

The source-hash/provenance files describe original verified source and external
inputs. Preserve those original files when recreating the runtime; this code
export alone is not a standalone data/model distribution. No checksum is
rewritten to make a missing or modified dependency appear valid.

Use the existing isolated Python 3.12 training environment and separately pinned
OR-Tools solver environment on Sol. Exporting or committing these files needs
no GPU allocation. Do not run a GPU experiment on a login node.

## Scientific scope

Method B selects fresh whole examples using seeded random priorities under
shared source/length quotas and supervised-token constraints. Its continuation
starts from the original R1000 parent with a fresh optimizer and warmup.

The original main 1x study used 6,409 candidates and a 1,000-row selection with
317,506 supervised tokens (target 318,845). Its historical record covers two
model sizes, two parent seeds, and three continuation seeds, with development
evaluation. These historical results are separate from the later 4,955-candidate
screened B/C0 pilot. Do not mix their result populations.

Parent seeds are 17/123; continuation seeds are 42/137/2026. They have different
roles. Development results, inherited parent/development exposure, transfer
experiments and reserved final evaluation must be reported separately. Existing
summaries are included as recorded; the exporter performs no new training,
evaluation or certification of the original matcher audit.

## Entry point

Inspect available arguments without starting training:

```bash
python method_b_document_v3_code/run_method_b.py --help
```

For actual execution, follow the original notebooks and provenance checks with
the external Sol inputs. See `EXPORT_MANIFEST.json` for omitted dependencies.

# Person 1 data preparation

Repository distribution: code, tests and [release records](release_records/person1_v3_20261012_ready/README.md) are tracked here. All datasets and original upstream files are provided as [GitHub Release downloads](docs/DATA_DOWNLOADS.md); extract them to the documented folders before full-data commands.

This package prepares reproducible mathematical-reasoning data for Qwen2.5-0.5B base. **Person 1 deliverables are complete: DATA_READY_FOR_TRAINER.** All 1,096,250 pinned source rows were processed, required Codex reviews are recorded, and 87 tests pass. The immutable [final release handoff](release_records/person1_v3_20261012_ready/PERSON2_HANDOFF.md) passed independent verification of 447 artifacts and 11 manifests.

Read [the folder guide and implementation status](docs/CODEBASE_GUIDE_AND_STATUS.md) and the release's `PERSON2_HANDOFF.md`. Earlier pilots, drafts and snapshot runs remain preserved for provenance. Git work is paused.

## Entry points

The corrected local/Colab notebook is `../Person1_Full_Run_v2.ipynb`. It calls the same named upstream stages as the CLI, with optional Drive mounting and an explicit project-root override. Full processing uses complete pinned upstream fields.

Run from the workspace root in an environment with the release's `code/requirements.txt` installed:

```powershell
& 'C:\softwares\pythonAnaconda\python.exe' -X utf8 -m unittest discover -s capstone_person1/tests -v
& 'C:\softwares\pythonAnaconda\python.exe' -X utf8 -m capstone_person1.scripts.validate_notebook
& 'C:\softwares\pythonAnaconda\python.exe' -X utf8 -m capstone_person1.scripts.validate_release --release capstone_person1/releases/person1_v3_20261012_ready
```

For a new data version: `python -m capstone_person1.release_pipeline --release capstone_person1/releases/YOUR_NEW_VERSION`. A pilot uses `--pilot 10000` and remains diagnostic. Inspect the generated review queues, record actual case decisions, then resume the unchanged draft. Finalized releases are immutable. The local `runtime_deps/` and `vendor/` installs isolate dependencies; this environment requires elevated read access to those existing installs.

## Data and contracts

The release supplies a balanced 40,000-group training bank, disjoint 10,000-group selection bank and 1,000-group held-out loss set, plus `debug_32` and nested 500/1,000/2,000/4,000 prefixes. Final counts, label-derived budgets and hashes are saved in the validated release manifest.

- [Prompt, pinned tokenizer, completion-only labels and EOS](docs/contracts/REFERENCE_ENCODER.md)
- [Canonical/manifest schema and final group mapping](docs/contracts/DATA_SCHEMA.md)
- [Person 2 actual-label and actual-parent checks](docs/contracts/TRAINER_INTEGRATION.md)
- [Person 3 frozen question inputs](docs/contracts/BENCHMARK_INPUT_CONTRACT.md)

Contract v3 preserves verified percentage notation in training answers while keeping canonical comparison values. Combined-text tokenization, prompt masking, one supervised EOS and positional padding masking are shared by training, evaluation and student-fit scoring. No model weights are loaded by data preparation.

Known GSM8K test and MATH500 question texts are protected. Person 3 supplies its own development/fast-development sets and confirms final-test roles; those inputs require a new protected data version before comparative main runs. Person 2 next verifies actual collator labels and executes the 32-example training/save/reload smoke test. Source validity and structural target consistency do not certify all teacher mathematics.

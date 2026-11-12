# B: latest screened 4,955-row study

This folder contains **12 completed B continuations** and **24 development
result files**, two benchmarks per checkpoint. Each run trains on 1,000
selected examples from the common 4,955-row eligible bank.

Read [the study README](../SCREENED_4955_README.md) for scope, exact Sol paths,
external prerequisites, provenance and the interpretation limits.

* `code/`: exact recorded source snapshots; original package checksums preserved.
* `notebooks/`: current notebook entry points, with output removed from copies.
* `training/`: native full-run completion, config, summary and compressed logs.
* `selections/`: selected row IDs/token metadata and native train-JSONL hashes.
* `evaluation/`: receipt, metrics, generation config and compressed predictions.
* `RESULTS.csv`: all 24 benchmark scores for this method.
* `PER_SEED_SCORES.csv`: 12 GSM/MATH/equal-weight checkpoint summaries.

The combined B notebook schedules training followed by development evaluation. The existing screened B pilot is reused. Run these notebooks in the original
Sol workspace with external datasets and parents restored, not as a standalone
GitHub training environment. No GPU work is triggered by opening these files.

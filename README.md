# infj-llm-distillation-improvement

Capstone project investigating improvements in data-efficient mathematical reasoning for small language models.

## Person 4: C0 reference-fit continuation

`capstone_person4/` implements the C0 approach: score 5,000 fresh candidate question–solution pairs by teacher-forced reference loss under an already fine-tuned R1000 parent, select the lowest-loss rows within fixed source/length strata up to the parent's one-pass token budget (318,845 supervised tokens ± 0.5%), continue the parent on them with a fresh optimizer, and hand the checkpoint off for evaluation. It reuses Person 2's notebook trainer through thin interfaces; importing it does not run historical training.

```bash
python -m unittest discover -s tests/person4 -v
python -m capstone_person4 --help
python -m capstone_person4 audit --config configs/c0.v4.0.5b.seed17.json --target inputs
python -m capstone_person4 prepare --config configs/c0.v4.0.5b.seed17.json
```

Stages: `audit → protect → prepare → score-reference → diagnose → select → train → finalize`. Each stage verifies its inputs by SHA256 and refuses stale or synthetic upstream artifacts. Explicit configurations cover Qwen2.5-0.5B and Qwen2.5-1.5B, each with parent seeds 17 and 123: `configs/c0.v4.<0.5b|1.5b>.seed<17|123>.json`. Paths resolve relative to the config file and expect the Person 1 V4 release (`../../data`), the parent package and the inherited encoder as workspace siblings.

## Running on ASU SOL

`scripts/sol/` holds the Slurm stage scripts for account `class_fse57081354fall2026` (`public` partition, `class` QoS, one A100 per GPU job). After generating SOL-local configs with `scripts/sol/configure_sol.sh` and caching the base models with `scripts/sol/prefetch_models.sh`, one command submits the seven-job dependency chain for a configuration:

```bash
bash scripts/sol/submit_pipeline.sh configs/c0.local.sol.0.5b.seed17.json q05-s17
```

## Results

`results/c0/` contains the provenance of the four completed C0 runs (one per base model and parent seed): handoffs, token reconciliation, run configuration and training logs, selection reports and ordered row IDs, reference-loss diagnostics and scoring runtime reports. The adapters themselves are shared separately; `results/c0/README.md` lists their checksums and the evaluation instructions. Accuracy evaluation is pending.

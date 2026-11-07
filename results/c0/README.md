# C0 (reference-fit) continuation runs: evaluation handoff

## What C0 is

Starting from an R1000 parent (Qwen2.5 base + LoRA trained on 1,000 Person 1 rows), C0 scored 5,000 fresh candidate question–solution pairs by teacher-forced mean next-token loss under that parent, selected the lowest-loss rows within each source/length stratum up to the parent's own one-pass budget (318,845 supervised tokens ± 0.5%), and continued training the parent on those rows for one pass with a fresh AdamW optimizer (lr 1e-4, 10 warmup updates, accumulation 8, seed 42). The candidate pool was filtered against the parent's training rows, the reserved loss-dev set and all five frozen evaluation files before selection.

## The four runs

| Run | Base model | Revision | Parent (R1000 training seed) | Rows selected | Supervised tokens | Adapter on Sol |
| --- | --- | --- | ---: | ---: | ---: | --- |
| C0_0.5B_parentseed17 | Qwen/Qwen2.5-0.5B | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 17 | 985 | 318,738 | `/scratch/jseth2/c0/results/0.5B/04ed3ff474c62fc4/runs/C0/pilot01/final/adapter` |
| C0_0.5B_parentseed123 | Qwen/Qwen2.5-0.5B | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 123 | 983 | 318,869 | `/scratch/jseth2/c0/results/0.5B/f2923ed6de792110/runs/C0/pilot01/final/adapter` |
| C0_1.5B_parentseed17 | Qwen/Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | 17 | 981 | 318,816 | `/scratch/jseth2/c0/results/1.5B/f561efe2e3bf2af3/runs/C0/pilot01/final/adapter` |
| C0_1.5B_parentseed123 | Qwen/Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | 123 | 980 | 318,886 | `/scratch/jseth2/c0/results/1.5B/273d12e6c0ac8e57/runs/C0/pilot01/final/adapter` |

The same paths exist inside `c0-results-handoff.tar.gz` under `results/...`. Each adapter folder contains `adapter_config.json` and `adapter_model.safetensors` (PEFT LoRA, r=16, alpha=32, all linear projections); the matching tokenizer is in the sibling `tokenizer/` folder.

### Checksums (`adapter_model.safetensors`)

| Run | Continued adapter SHA256 | Parent adapter SHA256 (must be the starting point) |
| --- | --- | --- |
| C0_0.5B_parentseed17 | `41d619ed2e9eefa112196637ccd92bd711307f5fd719bc886de399e2988333fd` | `11a9de70df8b102e57676fb1c4b8ed61f9f0d1e7f524c6c0f0169368ea17b1cb` |
| C0_0.5B_parentseed123 | `d24e94ff32a373646ad155118b3f13b27860cbbfa4c15eb82ca7fe86f4d4d350` | `aa11429355543bcfe35e47856a3c6ada4e5e0eb7f4647318c1c68aee5bf1c3a2` |
| C0_1.5B_parentseed17 | `faf612e73b2bb8ab42ea3121afdd3f6dafdc402a121b855c3f31ece8e5659896` | `e0e419f2963dae8a6ae8f03637fdf5c255c8634ddf882360164b6f6843c30a75` |
| C0_1.5B_parentseed123 | `2c9e0845bde768488bba81802e591f7287bdb2e87ba061c87eb3412024a0ad4b` | `849ca63fb2ebffc489144c68f00ac8513f1b9cae0c827c9fe4586cf6b38d370d` |

The parent adapters are in Person 2's package `R1000_parents_seeds17_123_20261101T023416637853Z/<model>/R1000_seed<N>/final/adapter`.

## How to evaluate

- Load the base model at the recorded revision and apply the adapter (merged mode, as in the Person 3 contract). Use the same plain-text prompt the training used, with no chat template:
  `Question: {problem}\nSolve the problem step by step. End with: Final answer: <answer>.\nSolution:\n`
  (SHA256 `22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db`, identical to the contract's `prompt_template_sha256`).
- Greedy decoding, EOS id 151643, frozen settings from `eval_contract.json` (fp32, 1,024 new tokens).
- Development sets first: `dev_gsm` (500) and `dev_math` (500). Final test sets only under the team's pre-declared final run.
- **Compare each C0 run with its own parent** (the seed-17 or seed-123 R1000 adapter listed above), not with the seed-42 baselines already in `eval_results/`. Those parents have not been evaluated yet, so eight runs are needed in total: four parents and four C0 adapters.
- Paired comparison by `problem_id` as in the contract (bootstrap, 2,000 resamples, seed 0).

## Provenance files per run (next to each adapter)

- `runs/C0/pilot01/token_reconciliation.json`: consumed tokens, selected row/group identities, parent unchanged, changed LoRA tensors.
- `runs/C0/pilot01/run_config.json` and `training_summary.json`: recipe, environment, optimizer updates (123–124), training seconds.
- `selection/C0/manifest.jsonl` and `allocation_report.json` (in the selection-stage folder of the same model size): the exact rows trained on and the budget accounting.
- `handoffs/<id>/evaluation_handoff.json`: everything above bound by hash, plus the five protected question files.
- `reference/diagnostics/reference_fit.json`: per-stratum loss distribution of the 5,000 candidates under that parent.

## Limitations to carry into the report

- Reference loss measures how predictable the reference solution is; it does not measure independent answer correctness.
- Decontamination used the supplied approximate matcher (exact/likely-same excluded, ambiguous similarity quarantined); it is not proof of complete removal.
- One continuation seed (42) per parent; differences between seed 17 and seed 123 runs reflect parent variation, not continuation-seed variation.

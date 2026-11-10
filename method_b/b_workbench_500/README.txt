APPROACH B — ADD THE APPROXIMATELY-500-EXAMPLE CONTINUATION POINT

This update adds 0.5x to the existing 1x/2x/4x token budgets. Every run still
starts independently from the same completed R1000 parent. This is not an R500
starting parent and not the original 500-example training-from-base sweep.
The selected row count is 501, because the experiment matches supervised tokens.

START ON SOL
1. Upload approach_b_workbench_500.zip into the existing project inputs folder.
2. Run the complete notebook cell in START_ON_SOL.txt. It extracts b_workbench_500/
   and runs run_500_preparation.py using the already-installed Python environment.
3. Wait for: 500-BUDGET PREPARATION COMPLETE.

A CPU session is sufficient. No trained adapter is needed for this batch.
It uses the environment and tokenizer cache that passed your previous SOL run.
It does not reinstall packages or fetch base weights. Tokenizer access defaults
strictly to the local cache; --allow-downloads is available if that cache was lost.
It does not launch any model, training, generation, or benchmark evaluation.

ISOLATION AND RESULTS
The original b_workbench/ code and outputs/B_workbench_v1/ remain intact.
New code lives in b_workbench_500/.
New outputs live in outputs/B_workbench_v2_500/.
The public model cache, environment, and parents/ locations are reused.

Actual stored-token selection results produced in the authoring workspace:
  0.5x:  501 rows,  158717 tokens; exact target 159422.5; difference -0.442535%.
  1x:   1000 rows,  318893 tokens; target 318845; difference +0.015054%.
  2x:   1999 rows,  634906 tokens; target 637690; difference -0.436576%.
  4x:   3999 rows, 1275227 tokens; target 1275380; difference -0.011996%.

All 20 source/length quotas are met. Every original row is preserved, IDs and
problem groups are unique, and there is zero recorded parent group overlap.
The three existing dataset JSONL files are byte-for-byte identical to version 1.
The 0.5x source counts are 250 Orca and 251 Numina.

EXACT HALF-TOKEN TARGET
The R1000 parent contains 318845 supervised tokens, so its half budget is exactly
159422.5. The allocator now accepts an exact Fraction target. It computes the
original +/-0.5% bound without rounding the target, truncating or repeating rows,
or loosening coverage constraints. Allowed integer tokens: [158626, 160219].
Integer-budget selection behavior is preserved. The extended allocator is
versioned shared-priority-allocator-v1-fractional-targets; other methods need the
same budget and coverage policy for a matched comparison.

WHAT THE CPU EXTENSION RUNS
- 48 offline tests, including fractional boundaries, the training/evaluation CLI,
  parent-file gates, safe resume rules, and the isolated CPU stage ordering.
- Reproduce all four selections in the new output directory.
- Run the exact recovered encoder on all 6000 parent/universe records for each
  model; check every budget using actual labels and compare 1.5B IDs/masks to 0.5B.
- Prepare 16 development evaluation plans: 2 model sizes x 2 parent seeds x 4 budgets.
- Print readiness and save progress/logs under B_workbench_v2_500/add_500_logs/.

The new actual tokenizer execution must run on SOL; it was not performed in the
authoring workspace. The prior successful SOL output is included as historical
reference in prior_SOL_preparation.log; it checked the previous three budgets.
There is no fabricated new tokenizer success, GPU smoke, training, or accuracy.

TRAINING/EVALUATION SUPPORT
Both commands now accept --budget 0.5, 1, 2, or 4. For example, after the actual
parent files and model-specific label checks are ready:
  <B_PYTHON> b_workbench_500/workbench.py --root <ROOT> train --model 0.5B --parent-seed 17 --budget 0.5 --smoke
  <B_PYTHON> b_workbench_500/workbench.py --root <ROOT> train --model 0.5B --parent-seed 17 --budget 0.5
  <B_PYTHON> b_workbench_500/workbench.py --root <ROOT> evaluate --model 0.5B --parent-seed 17 --budget 0.5 --execute
The configured matrix still uses R1000 parents, models 0.5B/1.5B and parent seeds
17/123, with continuation seed 42. Parent adapter hashes remain the same.
Get Feni's R1000 adapter_model.safetensors and adapter_config.json; the importer
uses matching supporting metadata already included. It does not load parent
optimizer/pickle files. R500 parents require a distinct manifest/registry setup.

The user reports the team agreed on the existing experiment. The config/template
has not been silently marked frozen: main training uses the actual matching
shared-protocol receipt, including this added budget. No new human approval or
source agreement has been invented. Existing disclosure and scope limits remain:
seven parent-related development rewrites, reviewed-queue screening rather than
exhaustive decontamination, and no mathematical correctness guarantee.

See ORIGINAL_WORKBENCH_README.txt for the original training-loop and evaluation
controls; its b_workbench/ and B_workbench_v1/ paths refer to the older version.

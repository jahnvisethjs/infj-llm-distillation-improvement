# Pool-matched B/C0/C1 development results

Every method here used the screened 4,955-row eligible bank and the same
500 GSM + 500 MATH development questions per checkpoint. There are 12
configurations per method: two sizes x two parents x three continuations.

The CSVs contain **only the 36 screened continuations / 72 result files**.
The original 6,409-row B predictions are intentionally not exported. A parent
baseline figure is included as context for the shared starting checkpoints.

1. `MATCHED_36_CHECKPOINT_SCORES.csv`: all per-run scores.
2. `MATCHED_72_DEVELOPMENT_RESULTS.csv`: benchmark accuracy and output diagnostics.
3. `SEED_SUMMARY.csv`: model/profile means and observed score SD.
4. `MATCHED_PAIRWISE_SUMMARY.csv`: paired differences and conditional question intervals.
5. `MATCHED_TRAINING_36.csv`: token exposure, training time and identity hashes.
6. `SELECTION_OVERLAP.csv`: ID intersections and Jaccard values.
7. `SCREENED_4955_AUDIT_SUMMARY.json`: lineage, verification scope and diagnostics.
8. `figures/`: performance, uncertainty, per-seed results, selected-ID overlap,
   answer changes, output diagnostics, compute and parent baselines.

Equal-weight accuracy is (GSM + MATH)/2. C1 has the highest observed average
at both sizes, but its improvement over C0 is small. All aggregate 95%
question-bootstrap intervals include zero. Question IDs are resampled jointly
across six fixed checkpoint pairs; those intervals exclude training-seed and
parent-sampling uncertainty. They are exploratory, without multiplicity adjustment.

The overall source audit includes historical/context outcomes, so its 55,361
check count is not a count restricted to this filtered GitHub export. The export
itself verifies the exact source archive hashes, 4,955-row bank, all current
checkpoint/manifest links and 72 full 500-question result files before copying.

These are development results with inherited exposure disclosed. Reserved final
tests are separate. The full 19-page comparison PDF was delivered separately;
it is omitted here because it also contains an explicitly historical B appendix.

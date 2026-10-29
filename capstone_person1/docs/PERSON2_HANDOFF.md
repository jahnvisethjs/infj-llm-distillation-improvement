# Person 2: training integration

Person 1's complete local release is DATA_READY_FOR_TRAINER. Read the [release handoff](../release_records/person1_v3_20261012_ready/PERSON2_HANDOFF.md) and [tracked release record](../release_records/person1_v3_20261012_ready/README.md). The verified data archives are packaged locally but have not yet been published; [DATA_DOWNLOADS.md](DATA_DOWNLOADS.md) records the intended distribution layout. After obtaining archive 01 from the data owner, extract it into the repository root and load `manifests/debug_32.jsonl`, then the declared training prefixes.

Use [shared reference encoding](contracts/REFERENCE_ENCODER.md) with contract `person1-plain-text-labels-v3`. Encode combined prompt/completion text without a chat template, mask prompt labels, supervise completion plus one EOS, and mask padding by position. EOS/PAD share 151643. Verified percentage notation is preserved in completions while comparison keys may use ratios.

Follow [actual-label/parent integration checks](contracts/TRAINER_INTEGRATION.md): verify all 32 debug IDs and all 1,000 R1000 IDs against their exact manifest hashes using your actual collator labels. Train/save/reload the debug checkpoint before main experiments. Independent learning-curve runs initialize from the pinned base model. Planned R1000 seen groups are not actual training evidence; bind the real run/checkpoint and seen groups after training.

The selection bank is disjoint from the entire 40K training bank and 1K loss set. Person 4 scores under its frozen actual parent. Main-run readiness also requires Person 3's evaluator/split freeze. No actual training, model checkpoint or evaluation result is claimed by this repository.

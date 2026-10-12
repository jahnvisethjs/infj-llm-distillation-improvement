# Person 3 frozen question input contract

Person 3 chooses, prepares and freezes evaluation datasets. Person 1 imports question text/provenance to check overlap against its data pools. This interface does not create evaluation datasets, use gold answers, implement scoring or run model evaluations.

## Existing inputs and current scope

`inputs_from_team/reserved_evaluation.csv` contains 1,819 known final-test questions: 1,319 GSM8K test and 500 MATH500. Their saved source revisions and file hash are validated. The historical metadata does not contain Person 3 confirmation under the new role contract, so final-test role confirmation is recorded as pending.

No completed frozen development or fast-development manifest is currently supplied. This remains `pending`; Person 1 data preparation continues. The proposed 500+500 full development and 100+100 fast development counts are Person 3 decisions, not enforced quotas or generated datasets.

## Files supplied by Person 3

Use `inputs_from_team/frozen_development.csv` and `inputs_from_team/frozen_development_metadata.json`. The CSV requires:

```csv
set_id,problem_id,problem
```

Full and fast development sets may share a file. IDs are unique within a set. Fast entries must have the same problem IDs and normalized question text as entries in their declared full parent. Additional answer columns are omitted from Person 1's exported protection inputs.

The metadata requires `schema_version=person3-question-manifest-v1`, the actual CSV `manifest_sha256`, actual `set_counts`, `confirmed_by`, timezone-aware `confirmed_utc` and `sets`. Every set has explicit `benchmark_family` (`gsm8k` or `math`), `role` (`development`, `fast_development` or `final_test`), `dataset_id`, immutable `revision`, `split`, optional dataset `configuration`, and `selection_policy`. A sampled selection method requires an integer seed. An immutable revision is a 40-character repository commit or `sha256:<64-character source-file digest>` for a custom source.

Each fast set also declares `parent_set_id`, and its family/source/revision/configuration/split must match that full parent. Set roles come from metadata. Renaming a file does not change its role. Development inputs cannot redefine final tests or reuse official GSM8K-test/MATH500 questions as development.

A metadata example is provided in [inputs_from_team/person3_question_metadata_TEMPLATE.json](../../inputs_from_team/person3_question_metadata_TEMPLATE.json). Its placeholders are intentionally invalid until Person 3 supplies actual information; it is not freeze evidence.

## Import and outputs

From the project workspace, with its Python dependencies installed:

```powershell
python -m capstone_person1.import_person3_manifests --output-root capstone_person1/diagnostics/person3_manifest_import
```

The same helper is embedded in v2's protection stage. It exports:

- `processed/protected_questions.csv`: question texts, explicit roles/source provenance, normalized hashes and input-manifest hashes.
- `reports/evaluation_protection_scope.json`: actual set counts, unique question count, confirmation/coverage status, input hashes and overlap-check scope.

The standalone import reports overlap checks as pending. The notebook updates exact/near check results only after those stages execute, and records that near retrieval is bounded/approximate. Fast subsets do not add new unique protected questions when they are nested inside full development.

Missing both development files allows independent Person 1 work. An incomplete pair, changed hash, missing confirmation, ambiguous role metadata, invalid nesting or development/final intersection fails validation. Supplying a valid one-family development manifest protects its provided questions while leaving complete development coverage pending.

Person 1's later release will distinguish data readiness from main-run and continuation readiness. This interface supplies protection evidence; it does not mark the full data release or experiments ready.

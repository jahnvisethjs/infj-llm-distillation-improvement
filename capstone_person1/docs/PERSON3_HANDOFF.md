# Person 3: evaluation integration

You can prepare evaluation datasets, generation and scoring now. Use the same plain-text prompt as training:

```python
from capstone_person1.person1_data_contract import render_prompt
prompt = render_prompt(problem)
```

Evaluation input ends at `Solution:\n`. Supply only the prompt; the model generates the reasoning, final answer and EOS. Gold answers remain under your evaluation pipeline's control.

Read [the frozen question manifest contract](contracts/BENCHMARK_INPUT_CONTRACT.md). Supply `inputs_from_team/frozen_development.csv` and `frozen_development_metadata.json`, using the CSV template and `person3_question_metadata_TEMPLATE.json`. The older `frozen_development_metadata_TEMPLATE.json` is a legacy example; use the explicit `person3-question-manifest-v1` metadata schema in the current contract.

Freeze set roles, source revisions/splits, selection policy/seeds, question IDs/text, counts and the actual CSV SHA-256. Fast-development sets must be exact subsets of their full-development parents. Populate confirmation identity/time with actual evidence.

```powershell
python -m capstone_person1.import_person3_manifests --output-root capstone_person1/diagnostics/person3_manifest_import
```

The existing reserved question inputs cover 1,319 GSM8K test and 500 MATH500 questions. Your final-test role confirmation and development/fast-development coverage are pending. The import validates frozen inputs and exports protection scope; final overlap checking against Person 1's final banks is still required. No evaluator or evaluation results are supplied by this package.

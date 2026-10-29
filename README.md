# infj-llm-distillation-improvement

Capstone project investigating data-efficient mathematical reasoning for small language models.

Person 1's current data work is complete: **DATA_READY_FOR_TRAINER**. Complete-source processing covers 1,096,250 pinned rows; the validated data release supplies balanced 40K training, 10K fresh selection and 1K held-out loss banks, debug_32 and nested training prefixes. 87 tests pass. Independent read-back verified 447 artifacts and all 11 manifests.

Start with the [codebase guide and status](capstone_person1/docs/CODEBASE_GUIDE_AND_STATUS.md), [Person 2 handoff](capstone_person1/docs/PERSON2_HANDOFF.md), or [Person 3 instructions](capstone_person1/docs/PERSON3_HANDOFF.md). The [tracked release record](capstone_person1/release_records/person1_v3_20261012_ready/README.md) contains pins, reviews, summaries and checksums. The verified dataset archives are packaged locally but have not yet been published; [DATA_DOWNLOADS.md](capstone_person1/docs/DATA_DOWNLOADS.md) records their intended contents, sizes and checksums. Earlier diagnostic fixtures are retained for history and do not replace the final manifests.

Install `requirements-person1.txt` in an isolated environment. The corrected local/Colab notebook `Person1_Full_Run_v2.ipynb` calls the same named stages as the CLI. The original local notebook and preserved snapshots remain available for provenance.

Person 2 next verifies actual labels and executes training/save/reload smoke tests. Person 3 prepares/freezes evaluation sets and validates generation/scoring. Main-run and continuation readiness remain pending those downstream inputs. Person 4 owns scoring, selection and analysis.

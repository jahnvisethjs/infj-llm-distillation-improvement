# Historical maintenance tools

These tools record or apply specific earlier migrations/milestones. They are retained for traceability and are not a sequence to rerun against the current project.

- `initialize_v2.py`: original local preservation and v2 creation.
- `align_v2_notebook.py`, `align_benchmark_stage.py`: earlier notebook migrations.
- `record_milestone.py`, `record_alignment_batch.py`, `record_benchmark_batch.py`: historical evidence recorders. Some have fixed run IDs or reject changes since their recorded milestone.

Their support paths have been adapted to the organized layout. Read a tool before intentionally using it: a historical migration or recorder can update current notebook/evidence files. Routine validation and setup commands are documented in [the current package README](../../README.md).

# Person 1 integration handoff

This package supplies the shared prompt/label encoder, evaluation-question import interface, team input templates, development source code and a 32-row diagnostic fixture. Status: **DIAGNOSTIC_ONLY**. No corrected final training, selection or loss bank is released here.

Start with [PERSON2_HANDOFF.md](docs/PERSON2_HANDOFF.md) and [PERSON3_HANDOFF.md](docs/PERSON3_HANDOFF.md). Current package hashes and verification are recorded in [INTEGRATION_HANDOFF_STATUS.json](evidence/INTEGRATION_HANDOFF_STATUS.json).

Run these commands from the repository root with Python 3.11 or later:

```powershell
python -m pip install -r requirements-person1.txt
python -m capstone_person1.scripts.prepare_tokenizer
python -m unittest discover -s capstone_person1/tests -t . -p "test_*.py"
python -m capstone_person1.scripts.verify_integration_fixture
```

Tokenizer setup downloads only the two tokenizer assets at the pinned Qwen2.5-0.5B revision. The cache is ignored by Git and must be prepared on each teammate's machine. `load_tokenizer()` uses the actual fast Transformers tokenizer; the unit tests and fixture checker also exercise the lightweight offline adapter against the pinned assets. No model weights or training are required for these checks.

The included v2 notebook supports the existing integration tests and preserves the diagnostic pipeline stages. Its embedded controls and contracts are synchronized with this package. Notebook and source preparation code do not establish a completed release; the snapshot rebuild additionally requires historical source files not shipped here. The newer canonicalization, instance-matching and pool-allocation modules are development components, with component tests, awaiting end-to-end release acceptance.

The 32-row fixture and its copied reports originate from `full_run_v2_20261012T181641Z_cfe8ce44`. Historical readiness reports describe that original run and may reference artifacts outside this small package. `REFERENCE_CONTRACT_VERIFICATION.json` is historical verification of the actual tokenizer, not a fresh certification of every copied module. Use the current integration status for package hashes and current test results.

Person 1 still needs to release the reviewed 40K training, 10K selection and 1K held-out loss banks, corrected R500/R1000/R2000/R4000 prefixes, final group mapping, overlap scope, review evidence and checksums. Person 3 owns evaluation preparation/freezing and scoring. Person 2 owns its actual collator verification and training/checkpoint evidence.

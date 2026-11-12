# Independent conservative screening proposal

10 November 2026 UTC. This is a documented alternative to waiting for the missing
teammate matcher records. It changes the eligible population and needs an explicit
study decision before becoming the frozen training protocol. The included CPU
preview creates reviewable alternative selections. It does not adopt the protocol,
create a training-clearance receipt or launch training.

## Concrete proposed population

| Item | Count |
| --- | ---: |
| Original candidate rows | 6,409 |
| Quarantined under the proposed rules | 1,454 |
| Remaining candidate rows | 4,955 |
| Remaining Orca rows | 3,313 |
| Remaining Numina rows | 1,642 |
| Minimum remaining rows in any source/length stratum | 148 |
| Required rows in each of the 20 strata | 50 |

The quarantine includes 195 of the original B selections and 306 of the original
C0 selections. These numbers are screening consequences, not counts of proven
contamination or mathematical invalidity. This is a substantially different
population from the earlier 6,397-row known-repairs preview.

## Fixed screening rules

The screen uses only question text and the already identified defects. It does
not use benchmark accuracy, student loss, candidate-generation outcomes or method
selection membership to choose thresholds or decide quarantine.

References contain 7,319 rows: 2,819 distinct protected benchmark question texts,
1,000 original R1000 parent questions, 1,000 reserved development-reference
questions and 2,500 reserved transfer questions. Protected fast-development
entries are nested and their identities are preserved when deduplicating text.

Fit one TF-IDF vectorizer over all 13,728 candidate/reference texts using
`analyzer=char_wb`, `ngram_range=(3,5)`, `min_df=2`, `float32`. Versions are recorded
in `screening_proposal.json`.

* Quarantine a candidate if its nearest external reference has cosine >=0.65.
* Quarantine both members if a candidate's nearest nonself internal neighbor has
  cosine >=0.95. Candidate-pair directions are reconciled symmetrically.
* Quarantine external equal-question matches under NFKC/whitespace normalization
  and the documented LaTeX/spacing normalization. Quarantine entire internal
  cosmetic-equality clusters, including both previously reviewed representatives.
* Include the 12 previously documented known exclusions. Propagate quarantine to
  all members of a recorded problem group.
* Preserve the original source/length bins, whole references, original-parent
  initialization, training recipe, 50-per-stratum quotas and token tolerances.

Nearest-neighbor cosine search is exact for this fitted vector space. The
similarity rule still has limited semantic recall and false positives. Its purpose
is a conservative operational quarantine, not a test establishing question
identity or mathematical truth. It deliberately removes some valid distinct
problems. A generic shared-topic question may be quarantined under this rule.

All retrieved flag evidence and the complete quarantined ID list are provided.
There are no admitted unresolved hits under these declared thresholds in the
retained population. That statement is limited to these rules and input texts.

## What it does and does not settle

This proposal allows the team to choose a fully specified replacement protocol
without treating the unavailable teammate audit as reproduced. It does not
certify that the original reported 25 exclusions/572 quarantines were all wrong,
all removed, or individually adjudicated. Those original IDs/pair records remain
unavailable and their status must be reported honestly.

Below-threshold semantic repetitions can remain. Unseen loss/reserve manifests
and other unprovided parent populations are not certified. Only nine full
candidate references were inspected during the targeted review; mathematical
correctness of the remaining bank is unknown. The original structural eligibility
checks are preserved. Seven inherited parent/development relationships remain
disclosed; fixed original parent weights do not become newly clean.

The final benchmark question texts are used for exclusion screening only. No
final-model predictions, scores, answers or final-test accuracy informed this
proposal. Reserved final evaluation remains after method/protocol freezing.

Using this population requires **new matched B/C0/C1 runs**. Old B results remain
historical context under the previous population. Do not mix the new C0/C1 models
with old B as if they arose from the same eligible pool. State results as selection
performance under this conservative screen rather than the original matcher.

## Review the alternative's budget feasibility on Sol

Upload this ZIP to `/scratch/sshind38/capstone_c0_c1_work/`. In the existing notebook:

```python
from pathlib import Path
import subprocess

folder = Path('/scratch/sshind38/capstone_c0_c1_work')
name = 'C0_C1_Independent_Screen_Proposal_20261110'
archive = folder / f'{name}.zip'
if not archive.is_file():
    raise FileNotFoundError(f'Upload {archive.name} into {folder} first')
subprocess.run(
    [str(TRAIN_PYTHON), '-m', 'zipfile', '-e', str(archive), str(folder)],
    check=True,
)
solver_python = '/scratch/sshind38/capstone_approach_b/venv_b_solver_v3/bin/python'
subprocess.run(
    [solver_python, '-u', str(folder / name / 'preview_independent.py')],
    cwd=folder, check=True,
)
```

The CPU stage checks the live full-bank identity and the exact projected question
content against the screened input. It verifies the original parent/runtime and
score receipts, checks cached full-row hashes, imports unchanged loss measurements,
recomputes C0 ranks and solves B and C0 under the original budget constraints.
No dependencies need to be installed and no GPU scoring is performed.

Outputs are saved under:

```text
/scratch/sshind38/capstone_c0_c1_work/outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17/<identity>/
```

They include `preview_report.json`, the proposed full `candidate_bank.jsonl`,
imported reference measurements, and both B/C0 manifests with solver receipts.
Share the printed report. A token-feasible preview makes the replacement option
concrete; it does not automatically authorize or implement a new training runner.

Before main training, choose between adopting this disclosed replacement protocol
and retaining the original protocol while obtaining its actual matcher evidence.
If this alternative is adopted, bind a new runner and record that decision together
with the final bank/selection hashes, then smoke/train one complete B/C0 comparison
before expanding to C1 and additional seeds. The installed v1 bridge continues to
refer to the original bank and must not be pointed at these files by weakening its
checks or forging an old-protocol resolved report.

## Reproduce the local screen

`independent_screen.py` uses the extracted previous evidence ZIPs in the relative
locations shown in its source. `screening_proposal.json` records their question-file
hashes, vectorizer versions, reference-content hash, thresholds and complete held
IDs. `screening_flags.json` contains nearest flagged counterpart text/identities
and cosmetic-equality clusters. `known_exclusions.json` binds proposed quarantines
to actual question hashes and the previously inspected reference hashes where
available.

Local validation covers screen execution, counts/capacity, complete quarantine
application and hash/changed-text guards. The Sol constrained solve and training
were not executed locally.

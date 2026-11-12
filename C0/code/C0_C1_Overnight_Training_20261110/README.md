# C0 and C1: unattended training on Sol

Upload the single `C0_C1_Overnight_Training_Only.ipynb` notebook to
`/scratch/sshind38/capstone_c0_c1_work`, open it in Sol Jupyter, and run its
configuration, installation and launch cells. Everything required by this new
runner is embedded in the notebook. The existing B runtime, four R1000 parents,
original screened preview, completed C0 pilot and existing Python environments
must remain available at their current Sol locations.

## Requested runs

| Model | Original parent seed | New C0 training seeds | C1 training seeds |
|---|---|---|---|
| 0.5B | 17 | 137, 2026; seed 42 is reused | 42, 137, 2026 |
| 0.5B | 123 | 42, 137, 2026 | 42, 137, 2026 |
| 1.5B | 17 | 42, 137, 2026 | 42, 137, 2026 |
| 1.5B | 123 | 42, 137, 2026 | 42, 137, 2026 |

That is **11 new C0 + 12 new C1 = 23 new training runs**. The final training
inventory includes the existing C0 pilot, giving 12 C0 and 12 C1 adapters.
Parent seeds are **17 and 123**; continuation seeds are **42, 137 and 2026**.

No B training and no GSM/MATH development or final-test evaluation are scheduled.
C1 necessarily includes original-parent generation and grading on the 4,955
training candidates to identify confirmed mistakes. This selection preparation
uses the existing mathematical grader; it is separate from benchmark evaluation.
Training smoke/update/reload checks also run before each new full training.

## Disconnection and allocation limits

The launch cell checks all four parents, offline model caches, the completed
pilot, source hashes and the existing environments before submission. It then
submits a serial Slurm chain and prints the job IDs and log paths.
**After those IDs are printed, the notebook and laptop can disconnect.**

Each job requests one A100, four CPUs, 32 GB RAM and a four-hour allocation.
Six allocations are requested by default: **24 hours of total allocation caps**,
not an expected runtime. Jobs run one at a time. Each worker hands remaining
work to the next pre-submitted job. Unneeded later jobs are cancelled when
training finishes or when there is no runnable work left. A shorter chain can
be accepted if Sol rejects further submissions; the printed ledger records its
actual length. Queue time is additional, and completing everything in one night
is not guaranteed.

The controller keeps enough time to stop before the allocation ends. Native
candidate generations resume from compatible per-question records in bounded
512-row chunks. Reference scores cache per row. Completed training stages are
hash checked and reused. A partially trained model restarts from its matching
original parent in a new attempt, preserving the partial attempt; it never
guesses optimizer state. Attempts are bounded to three. Restarting the launch
cell does not submit a second active chain.

## Scientific controls

The separately disclosed conservative screen retains 4,955 of 6,409 candidates.
All selections retain 1,000 rows, 50 per source/length stratum, global tokens
within 0.5% of 318,845 and stratum tokens within 5%. Each model/parent receives
its own reference scoring and C1 candidate outcomes. Selections are reused
across that parent's continuation seeds.

C1 uses the existing declared rule `round(1000000 * fit) * (1 + confirmed_mistake)`
with lambda fixed to 1. Only parsed, confirmed incorrect answers get the bonus.
Missing, unparseable, ambiguous and truncated-without-answer outputs do not
become confirmed mistakes. Checker execution errors block selection until retried.

The existing diagnostic gate is preserved: a confirmed-mistake share below 5%
or above 95%, or C0/C1 Jaccard overlap at least 95%, blocks the affected C1
training. Other valid parent profiles can continue. No gate, token bound,
parent identity or generation setting is automatically relaxed to fill a table.

The missing old matcher audit and inherited parent/development exposure remain
disclosed. This training-only package does not certify contamination-free
generalization or establish an accuracy improvement. Matching screened B
controls beyond the completed pilot and all benchmark comparisons remain
separate work.

## Outputs

The launch cell prints a content-addressed directory under:

`/scratch/sshind38/capstone_c0_c1_work/outputs/C0_C1_overnight_training_v1/`

| Artifact | Meaning |
|---|---|
| `STATUS.json` | Actual counts and stage status, including failures |
| `TRAINING_RUNS.csv` | All 24 requested model/method/seed combinations and adapter paths |
| `TRAINING_COMPLETE.json` | Written only when all 23 new runs and their required preparation pass |
| `C0_C1_training_receipts.zip` | Small evidence package; excludes model weights and candidate raw outputs |
| `profiles/<model>_seed<parent>/<identity>/runs/<C0 or C1>/seed<seed>_attemptNN/final/adapter/` | New trained adapter |
| `stage_logs/` and `slurm_logs/` | Separate stage and job logs |

The original C0/0.5B/parent17/continuation42 adapter remains in its original pilot
directory and is referenced in the CSV. It is not copied or retrained.

Run the notebook status cell after reconnecting. If the status is
`incomplete_allocation_chain_exhausted`, rerun the launch cell to submit another
bounded chain and reuse valid completed stages. For `incomplete_blocked`, inspect
the reported stage logs; changing the scientific protocol requires a new
version rather than erasing failed checks. A Slurm `COMPLETED` status alone does
not establish that all requested training finished.

## Validation and limitations

Local CPU tests cover the 23-run matrix, candidate coverage, confirmed-mistake
classification, exact source/length/token controls, original-parent optimizer
isolation receipts, generic 1.5B/parent123/seed137 verification, preservation of
partial attempts, held-job recovery, duplicate submission protection and the
full controller across an interrupted allocation. GPU work in the controller
tests is simulated; no Qwen training, real Sol scheduling or GPU throughput was
executed in this environment. On Sol, each real training runs its own adapter
change and reload checks and produces hash-bound consumption evidence.

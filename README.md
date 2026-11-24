# infj-llm-distillation-improvement

Capstone project investigating improvements in data-efficient mathematical reasoning for small language models.

All Person 4 packages are additive: they reuse the R1000 parents, Person 2's trainer and the frozen Person 3 evaluator/parser without modifying historical training, native runtimes, adapters or original inputs. Each run directory is create-only and sealed with SHA256 receipts; interrupted work is preserved and repaired under a new attempt name.

| Approach | Package | Guide |
|---|---|---|
| C0 reference-fit continuation | `capstone_person4/` | [C0 run guide](docs/C0_RUN_GUIDE.md) |
| C1 continuation (Method B bank) | `capstone_person4_approachC1/` | [C1 SOL guide](docs/C1_SOL_SBATCH_GUIDE.md) |
| Approach E: mistake-only continuation | `capstone_person4_approachE/` | [E run guide](docs/E_RUN_GUIDE.md) |
| Replay + C1 50/50, Qwen2.5-0.5B | `capstone_person4_approachReplayC1/` | [Replay+C1 SOL guide](docs/REPLAY_C1_SOL_GUIDE.md), [v5 parent123 release](releases/replay_c1_v5/README.md) |
| Replay + C1 50/50, Qwen2.5-1.5B | `capstone_person4_replay_c1_15b/` | [1.5B SOL guide](docs/REPLAY_C1_15B_SOL_GUIDE.md) |
| 1.5B parent17 evaluation recovery | `capstone_replay15b_recovery/` | [Recovery guide](docs/REPLAY15B_EVALUATION_RECOVERY.md) |
| Question-only adapter routing | `capstone_routing/` | [Routing run guide](docs/ROUTING_RUN_GUIDE.md), [pilot guide](docs/ROUTING_PILOT_GUIDE.md) |

## Person 4: C0 reference-fit continuation

`capstone_person4/` implements the C0 approach: score 5,000 fresh candidate question–solution pairs by teacher-forced reference loss under an already fine-tuned R1000 parent, select the lowest-loss rows within fixed source/length strata up to the parent's one-pass token budget (318,845 supervised tokens ± 0.5%), continue the parent on them with a fresh optimizer, and hand the checkpoint off for evaluation. It reuses Person 2's notebook trainer through thin interfaces; importing it does not run historical training.

```bash
python -m unittest discover -s tests/person4 -v
python -m capstone_person4 --help
python -m capstone_person4 audit --config configs/c0.v4.0.5b.seed17.json --target inputs
python -m capstone_person4 prepare --config configs/c0.v4.0.5b.seed17.json
```

Stages: `audit → protect → prepare → score-reference → diagnose → select → train → finalize`. Each stage verifies its inputs by SHA256 and refuses stale or synthetic upstream artifacts. Explicit configurations cover Qwen2.5-0.5B and Qwen2.5-1.5B, each with parent seeds 17 and 123: `configs/c0.v4.<0.5b|1.5b>.seed<17|123>.json`. Paths resolve relative to the config file and expect the Person 1 V4 release (`../../data`), the parent package and the inherited encoder as workspace siblings.

## C1 continuation

C1 batch scripts for the same Method B candidate bank are documented in
[the C1 SOL guide](docs/C1_SOL_SBATCH_GUIDE.md). They require matching completed
C0 runs and Person 3 candidate predictions, then submit gated selection,
smoke checks, three C1 continuation seeds per parent, and paired handoffs.

## Approach E: mistake-only continuation

`capstone_person4_approachE/` reuses the completed C1 parent grading, common bank,
Method B allocator and Person 2 trainer through an isolated E entry point. Start
with [the E run guide](docs/E_RUN_GUIDE.md), including the required existing SOL
outputs, preflight, twelve-run submission procedure and additive Megha handoff.
Run `python -m capstone_person4_approachE --help` for commands. No new candidate
grading or reference scoring is required.

## Replay + C1: 50/50 supervised-token hybrid

The `REPLAY_C1_50_50` arm continues each original R1000 parent on a fixed mix of
half replayed parent examples and half fresh C1-selected examples from the native
screened 4,955-row bank. The target is the same 318,845 supervised tokens; each
whole-example half holds 500 examples (25 per frozen source/length stratum) and
158,626–160,219 tokens, solved with CP-SAT. Replay uses seeded hash priorities over
parent examples; fresh rows use the native C1 rule
`round(1000000 * fit) * (1 + confirmed_mistake)` from cached reference scores and
generation outcomes only. The selection is frozen across continuation seeds 42, 137
and 2026, and every run trains from the original parent with a fresh optimizer after
a 16-example smoke. There is no ratio sweep and no new generation or scoring.

- **0.5B** — `capstone_person4_approachReplayC1/` ([guide](docs/REPLAY_C1_SOL_GUIDE.md)).
  Parent17 is the original pilot; [release v5](releases/replay_c1_v5/README.md)
  extends the same launcher to parent123. Hybrid and A adapters are re-evaluated
  under C1's common runtime on the same 500 GSM + 500 MATH development questions,
  with paired bootstrap differences against A and C1.
- **1.5B** — `capstone_person4_replay_c1_15b/` ([guide](docs/REPLAY_C1_15B_SOL_GUIDE.md))
  runs parents 17 and 123 in a separate deployment: six full runs and two smokes.
- **Evaluation recovery** — `capstone_replay15b_recovery/`
  ([guide](docs/REPLAY15B_EVALUATION_RECOVERY.md)) re-runs only the evaluation of the
  completed 1.5B parent17 runs, whose original job stopped when the old validator
  rejected the legitimate `empty_marker` parser status. It reuses all three trained
  adapters as attempt `eval02_recovery` and refuses parent123.

```bash
python -m capstone_person4_approachReplayC1 configure --config-output <config.json> --output-root <dir>
python -m capstone_person4_approachReplayC1 preflight --config <config.json>
python -B -m unittest discover -s capstone_person4_approachReplayC1/tests -t . -v
python -B -m unittest discover -s capstone_person4_replay_c1_15b/tests -t . -v
python -B -m unittest discover -s capstone_replay15b_recovery/tests -t . -v
```

Configure and preflight are read-only and submit no jobs. SOL launchers are
`scripts/sol/replay_c1_*.sh`, `scripts/sol/replay15b_*.sh` and
`scripts/sol/recover_replay15b_worker.sh`, with examples in `configs/replay_c1*.json`
and `configs/replay15b.*.json`. `scripts/share_replay_c1.py` validates the completion
receipts of the twelve trained replay+C1 adapters and copies them into a new,
checksummed shared folder for evaluation handoff.

## Question-only adapter routing

`capstone_routing/` provides matched-block correctness/oracle analysis, a held-out
router experiment, A-only seed-diversity controls, and verified live answering.
Start with the [routing run guide](docs/ROUTING_RUN_GUIDE.md); expensive stages are
explicit, and the first deliverable uses saved development predictions only.

The first deliverable is correctness matrices, rescue/risk counts and oracle accuracy
for the existing 0.5B A/B/C1 matched blocks. Later stages — preparing an independent
question pool, labeling, embedding, fitting and freezing a router, a single frozen
test evaluation, and live `route`/`answer` — each run only when invoked. Before full
labeling, the [pilot guide](docs/ROUTING_PILOT_GUIDE.md) describes a small screen
(parent17, seed42, pool `{A42, B42, C1_42}` vs. A-only `{A42, A137, A2026}`, about
2,250 answers) that stops after a validation report.

```bash
python -m capstone_routing --help   # inspect, opportunity, patterns, prepare-pool, label, embed, train, freeze, evaluate-test, report, route, answer
python -m unittest discover -s tests/routing -v
python -B scripts/package_routing.py --output <routing.zip>
```

Embedding and fitting need `requirements-routing.txt` in a separate CPU environment;
`label` runs in the existing native GPU evaluator environment. SOL scripts are
`scripts/sol/routing_pilot_*.sh`, and `configs/routing.sol.example.json` is the
configuration template.

## Running on ASU SOL

`scripts/sol/` holds the Slurm stage scripts for account `class_fse57081354fall2026` (`public` partition, `class` QoS, one A100 per GPU job). After generating SOL-local configs with `scripts/sol/configure_sol.sh` and caching the base models with `scripts/sol/prefetch_models.sh`, one command submits the seven-job dependency chain for a C0 configuration:

```bash
bash scripts/sol/submit_pipeline.sh configs/c0.local.sol.0.5b.seed17.json q05-s17
```

Replay+C1, recovery and routing packages are deployed as separate ZIP bundles extracted into new directories on SOL; see each guide for its transfer, preflight and submission steps.

## Results

`results/c0/` contains the provenance of the four completed C0 runs (one per base model and parent seed): handoffs, token reconciliation, run configuration and training logs, selection reports and ordered row IDs, reference-loss diagnostics and scoring runtime reports. The adapters themselves are shared separately; `results/c0/README.md` lists their checksums and the evaluation instructions. Accuracy evaluation is pending.

Replay+C1 results are written on SOL under `reports/<evaluation-attempt>/` (`accuracy.csv`, `paired_differences.csv`, `run_index.csv`, `summary.json`, `provenance.json`) for each model/parent identity; the guides list the exact paths. They are not yet committed to this repository.

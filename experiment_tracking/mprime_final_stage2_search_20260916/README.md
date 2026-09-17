# MPrime final validation-led Stage-2 search

The final Phase-B-A search campaign contains exactly forty identities:

- ten seeds x two value-head modes x fixed 20/70 MCTS;
- the same ten seeds x two value-head modes x PW70;
- the Phase-B-A-selected checkpoint for every lineage;
- the exact selected policy job/log, source training job, checkpoint path and
  checkpoint SHA-256 for every identity.

The obsolete pre-rescore array `21388436` was cancelled and dropped from all
result tables because it used the saturated training validator rather than the
final Phase-B-A selector. Its raw scheduler logs remain immutable operational
provenance only; they are not an experimental branch or “quarantined result.”

The corrected controller completed all 420 Phase-B-A checkpoint evaluations,
froze twenty endpoints, reused twelve exact-hash policy evaluations, and ran
only eight genuinely missing VH-on policy endpoints. All twenty selected policy
scores are now complete. Array `21429177_[0-39]` evaluates the exact selected
checkpoints under fixed 20/70 and PW70; all forty tasks were running at the
17 September 2026 11:51 IDT snapshot.
Each task requests six CPUs, 120 GiB and a 72-hour allocation, with three
rolling workers, a six-hour per-instance limit and a 10,000-action limit.
The fixed and PW arms share width 20, 70 simulations, PUCT 0.1 and estimator
mixture 0.5. PW additionally uses Kmin 3, c 0.6 and alpha 0.5.

`manifest.csv` is the scientific identity table. `submissions.tsv` maps every
identity to its Slurm array task. `source_ready_manifest_20260916_1024.csv` and
`selected_policy_scores_20260916_1024.txt` freeze the upstream policy evidence.
The local runner verifies the declared code revision and checkpoint hash before
evaluation and writes one durable completion ledger per identity.

At 11:51 IDT, 234/800 classifications were durable:

| Method | VH | Classified | 30m / 2h / 6h successes | Mean lower bound |
|---|---|---:|---:|---:|
| Fixed | off | 53/200 | 52 / 53 / 53 | >=5.2 / >=5.3 / >=5.3 |
| Fixed | on | 51/200 | 50 / 51 / 51 | >=5.0 / >=5.1 / >=5.1 |
| PW70 | off | 69/200 | 67 / 69 / 69 | >=6.7 / >=6.9 / >=6.9 |
| PW70 | on | 61/200 | 60 / 61 / 61 | >=6.0 / >=6.1 / >=6.1 |

These are live lower bounds, not final paired results; no confidence interval or
significance claim is made until every matched seed is terminal. Exact identity,
checkpoint, policy-job and log provenance is in
`phase_b_a_manifest_20260917.csv`; live counts and completion globs are in
`live_progress_20260917_1151.csv`.

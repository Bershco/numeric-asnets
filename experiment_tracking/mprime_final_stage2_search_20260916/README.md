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

These Stage-2 networks were trained with the historical **dropout-current KL**
semantics. Their exact training commands contain the constant anchor
coefficient but not `--policy-anchor-kl-deterministic-current`. Accordingly,
the MPrime RQ1/RQ3 extension is evidence for the historical implemented
Stage-2 pipeline, not evidence for a corrected deterministic-current pipeline.

`manifest.csv` is the scientific identity table. `submissions.tsv` maps every
identity to its Slurm array task. `source_ready_manifest_20260916_1024.csv` and
`selected_policy_scores_20260916_1024.txt` freeze the upstream policy evidence.
The local runner verifies the declared code revision and checkpoint hash before
evaluation and writes one durable completion ledger per identity.

At 15:33 IDT, 422/800 classifications were durable:

| Method | VH | Classified | 30m / 2h / 6h successes | Mean lower bound |
|---|---|---:|---:|---:|
| Fixed | off | 83/200 | 71 / 78 / 83 | >=7.1 / >=7.8 / >=8.3 |
| Fixed | on | 67/200 | 60 / 64 / 67 | >=6.0 / >=6.4 / >=6.7 |
| PW70 | off | 148/200 | 130 / 146 / 148 | >=13.0 / >=14.6 / >=14.8 |
| PW70 | on | 124/200 | 109 / 122 / 124 | >=10.9 / >=12.2 / >=12.4 |

These are live lower bounds, not final paired results; no confidence interval or
significance claim is made until every matched seed is terminal. Exact identity,
checkpoint, policy-job and log provenance is in
`phase_b_a_manifest_20260917.csv`; live counts and completion globs are in
`live_progress_20260917_1533.csv`.

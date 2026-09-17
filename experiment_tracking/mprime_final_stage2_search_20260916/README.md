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

At 13:08 IDT, 332/800 classifications were durable:

| Method | VH | Classified | 30m / 2h / 6h successes | Mean lower bound |
|---|---|---:|---:|---:|
| Fixed | off | 64/200 | 58 / 64 / 64 | >=5.8 / >=6.4 / >=6.4 |
| Fixed | on | 56/200 | 53 / 56 / 56 | >=5.3 / >=5.6 / >=5.6 |
| PW70 | off | 121/200 | 108 / 121 / 121 | >=10.8 / >=12.1 / >=12.1 |
| PW70 | on | 91/200 | 81 / 91 / 91 | >=8.1 / >=9.1 / >=9.1 |

These are live lower bounds, not final paired results; no confidence interval or
significance claim is made until every matched seed is terminal. Exact identity,
checkpoint, policy-job and log provenance is in
`phase_b_a_manifest_20260917.csv`; live counts and completion globs are in
`live_progress_20260917_1308.csv`.

# MPrime final validation-led Stage-2 search

This directory records a premature search campaign built before the final
Phase-B-A Stage-2 endpoint rescore. It contains exactly forty identities:

- ten seeds x two value-head modes x fixed 20/70 MCTS;
- the same ten seeds x two value-head modes x PW70;
- epoch 0 for every lineage, selected by the ordinary saturated training
  validator—not by Phase-B-A;
- the exact selected policy job/log, source training job, checkpoint path and
  checkpoint SHA-256 for every identity.

All forty tasks were submitted as Slurm array `21388436` on 16 September 2026.
The provenance audit then found that `checkpoint_selection=phase_b_replicate_a`
and `selected_validation_score=30` had been assigned without a Phase-B-A
rescore. The array was cancelled at 11:48 IDT after 1:11:16. Its provisional
aggregate table has been deleted and none of its reported means enters
RQ1-RQ4. Raw logs remain only as immutable execution provenance. After the
independent selector finishes, an individual raw result may be reused only if
checkpoint hash and every search parameter exactly match a final identity.
Each task requests six CPUs, 120 GiB and a 72-hour allocation, with three
rolling workers, a six-hour per-instance limit and a 10,000-action limit.
The fixed and PW arms share width 20, 70 simulations, PUCT 0.1 and estimator
mixture 0.5. PW additionally uses Kmin 3, c 0.6 and alpha 0.5.

`manifest.csv` is the scientific identity table. `submissions.tsv` maps every
identity to its Slurm array task. `source_ready_manifest_20260916_1024.csv` and
`selected_policy_scores_20260916_1024.txt` freeze the upstream policy evidence.
The local runner verifies the declared code revision and checkpoint hash before
evaluation and writes one durable completion ledger per identity.

At the 11:07 IDT snapshot all forty tasks were running and 222/800
classifications were durable. The array is now cancelled. After the independent
Phase-B-A selector is frozen, a row is reusable only when its checkpoint hash
and full fixed/PW configuration match the true selected endpoint exactly.
Nonmatching rows are dropped. No provisional score table is retained.

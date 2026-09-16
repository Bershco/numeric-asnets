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
rescore. The array is scientifically quarantined and explicit approval has
been requested to cancel it. Its outputs must not enter RQ1-RQ4 merely because
they complete.
Each task requests six CPUs, 120 GiB and a 72-hour allocation, with three
rolling workers, a six-hour per-instance limit and a 10,000-action limit.
The fixed and PW arms share width 20, 70 simulations, PUCT 0.1 and estimator
mixture 0.5. PW additionally uses Kmin 3, c 0.6 and alpha 0.5.

`manifest.csv` is the scientific identity table. `submissions.tsv` maps every
identity to its Slurm array task. `source_ready_manifest_20260916_1024.csv` and
`selected_policy_scores_20260916_1024.txt` freeze the upstream policy evidence.
The local runner verifies the declared code revision and checkpoint hash before
evaluation and writes one durable completion ledger per identity.

At the 11:07 IDT snapshot all forty tasks were still running and 222/800
classifications were durable. Evidence is preserved. After the independent
Phase-B-A selector is frozen, a row is reusable only when its checkpoint hash
and full fixed/PW configuration match the true selected endpoint exactly.
Nonmatching rows remain invalid for the final Stage-2 RQs.
The corresponding provisional policy-pair table is retained explicitly as
`quarantined_old_validator_policy_pairs.csv`; it is not a canonical RQ input.

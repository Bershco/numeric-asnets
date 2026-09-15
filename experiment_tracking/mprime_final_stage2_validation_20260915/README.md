# MPrime final validation-led Stage 2

This is the clean final MPrime Stage-2 campaign after the Phase-C validator decision and the complete Phase-B-A anchor rescore.

- Checkpoint selection: Phase-B replicate A.
- Branch: validation-led only.
- VH-off anchor: 30.
- VH-on anchor: 10.
- Seeds: the same ten replication seeds per VH mode.
- Training: 100 Stage-2 epochs, learning rate 0.0003, three target-generation workers, 20 retained children, PUCT 0.1, estimator mixture 0.5.

All twenty lineages are retrained in one uniform current-code campaign. Two historical VH-on lineages matched the selected checkpoint and coefficient superficially, but mixing their older build/configuration provenance with eighteen fresh lineages would make the final campaign heterogeneous. The two additional jobs remove that ambiguity.

`manifest.csv` freezes every source checkpoint and its selector provenance. `submissions.tsv` is copied back from the cluster after submission and provides the Slurm job route for every lineage.

## Live overlap and policy-curve materialization

At the 15 September 2026 18:57 IDT audit, one of training jobs
`21303717`-`21303736` was terminal at epoch 99 and the other nineteen were
running with latest immutable checkpoints spanning epochs 42-98. Epoch 99 is
the final checkpoint after 100 zero-indexed Stage-2 epochs. A safe live
scanner/controller was deployed so policy-only curve
evaluation overlaps the remaining training instead of waiting for all twenty
lineages:

- controller `21345695`, one task, two CPUs and 2 GiB;
- at most 24 policy jobs active simultaneously, each 10 CPUs, 20 GiB and a
  four-hour hard limit;
- 300 immutable checkpoints discovered, 46 submitted, 24 complete and 22
  active at the audit;
- every live row is `learning_curve` only; validation-selected and final roles
  are withheld until the corresponding training lineage is scientifically
  complete through epoch 99;
- every manifest row carries the source training job, epoch, exact checkpoint
  path and SHA-256 hash, and the shared submission ledger prevents duplicates.

Current counts and direct remote/local provenance routes are frozen in
`live_policy_progress_20260915_1857.csv`. The files carrying `1810` in their
names are immutable earlier manifest/ledger snapshots, not the latest counts.

The controller exits only after all twenty lineages are complete and every
materialized policy row has been submitted. Final MCTS and PW manifests must be
built from the resulting validation-selected endpoints; no historical MCTS
identity may be reused unless its checkpoint hash and full configuration match.

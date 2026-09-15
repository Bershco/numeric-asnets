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

At the 16 September 2026 01:37 IDT audit, nineteen of the twenty training jobs
`21303717`-`21303736` were terminal at epoch 99 and only `21303720` was running,
at snapshot 78. Epoch 99 is the final checkpoint
after 100 zero-indexed Stage-2 epochs. A safe live scanner/controller overlaps
policy-only curve evaluation with the remaining training:

- primary controller `21362560`, one task, two CPUs and 2 GiB;
- failure-aware retry controller `21362500`, one task, two CPUs and 2 GiB;
- at most 24 primary and 36 retry policy jobs active simultaneously; each uses
  10 CPUs, 20 GiB and a four-hour hard limit;
- 414 immutable checkpoint rows discovered, 269 primary submissions and 40
  retry attempts in the dated 01:34 local archive;
- the subsequent scheduler count reconciles those 309 attempts exactly: 237
  complete, 30 running and 42 failed; attempts are not unique scientific identities;
- every live row is `learning_curve` only; validation-selected and final roles
  are withheld until the corresponding training lineage is scientifically
  complete through epoch 99;
- every manifest row carries the source training job, epoch, exact checkpoint
  path and SHA-256 hash, and the shared submission ledger prevents duplicates.

The failure-aware path was added after policy evaluations landed on
`ise-cpu128-03` and failed before inference because the node could not create
POSIX semaphores (`ENOSPC`). Both submission paths now exclude only that node.
The two post-repair failures inspected at this audit were jobs already placed
on the same node before the exclusion was effective; their replacements use
the private excluded-node wrapper. No partial policy score is reused after a
pre-inference failure.

Current direct local provenance is frozen in
`live_policy_ready_20260916_0134.csv`,
`live_policy_submissions_20260916_0134.tsv`, and
`live_policy_retry_submissions_20260916_0134.tsv`. Earlier dated files remain
immutable historical snapshots.

The controller exits only after all twenty lineages are complete and every
materialized policy row has been submitted. Final MCTS and PW manifests must be
built from the resulting validation-selected endpoints; no historical MCTS
identity may be reused unless its checkpoint hash and full configuration match.

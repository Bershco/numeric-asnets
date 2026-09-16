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

## Completed training and policy-curve materialization

At the 16 September 2026 10:48 IDT audit, all twenty training jobs
`21303717`-`21303736` were terminal through epoch 99. Epoch 99 is the final
checkpoint after 100 zero-indexed Stage-2 epochs. The final immutable ready
manifest contains 420 unique policy checkpoints (21 checkpoints x 20
lineages). Of those, 403 had valid policy logs and seventeen non-selected curve
points had repeatedly failed before inference with native runtime errors.

The training log's ordinary internal validator saturated and selected epoch 0
for every lineage. Those selections are not the adopted Phase-B-A validator
and are therefore not primary endpoints. A corrected, independent Phase-B-A
rescore now evaluates all 420 saved checkpoints as twenty lineage tasks. Its
preflight is job `21390403`, main array `21390404_[0-19]`, and finalizer
`21390429`. MPrime Stage-2 policy values remain withheld from RQ1/RQ3 until
that finalizer freezes the true endpoints.

Of the 420 checkpoints, 403 already have valid test-policy logs. Seventeen
non-selected-under-the-old-validator curve points repeatedly failed before
inference and are being recovered exactly with one worker, two CPUs, 20 GiB
and an eight-hour hard limit under jobs `21389264`-`21389280`. After Phase-B-A
selection, every selected endpoint reuses an existing valid policy log when
available; only a selected identity still missing valid policy evidence may be
recovered.

Every manifest row carries the source training job, epoch, exact checkpoint
path and SHA-256 hash. Primary/retry ledgers and the dedicated recovery ledger
map every attempt back to the same immutable scientific identity.

The failure-aware path was added after policy evaluations landed on
`ise-cpu128-03` and failed before inference because the node could not create
POSIX semaphores (`ENOSPC`). Both submission paths now exclude only that node.
The two post-repair failures inspected at this audit were jobs already placed
on the same node before the exclusion was effective; their replacements use
the private excluded-node wrapper. No partial policy score is reused after a
pre-inference failure.

The 01:34 files remain immutable historical snapshots. The premature search
manifest under `../mprime_final_stage2_search_20260916/` used the saturated
internal-validator epoch-0 roles and is quarantined. Its outputs may be reused
only if the independently frozen Phase-B-A endpoint has the exact same
checkpoint hash and full search configuration.

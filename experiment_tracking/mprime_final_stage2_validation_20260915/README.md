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

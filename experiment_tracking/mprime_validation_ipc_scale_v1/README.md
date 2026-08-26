# MPrime corrected-validation experiment

The original generated validation set selected epoch 1 for all twenty MPrime
Stage-1 lineages and was structurally much smaller than the IPC test set.  The
replacement protocol is fixed before checkpoint scoring and does not use test
coverage for generation or selection.

## Frozen design

- 30 deterministic instances: ten easy, ten medium, ten hard.
- Food and pain ranges overlap the IPC test scale.
- Fixed generator seeds and SHA-256 checksums are recorded in the generated
  set's `manifest.csv` and `metadata.json`.
- Goals must be absent initially and every generated instance must pass a
  planner sanity check with a strictly positive plan length.
- `lineage_manifest.tsv` points to the twenty existing Stage-1 lineages whose
  checkpoints will be rescored after the sanity gate passes.

## Current state

The first replacement set exposed a generator bug: all goals were already true
and all 30 planner plans had length zero.  That set was rejected before any
selection decision.  The generator was corrected, the 30-instance set was
regenerated, and every file passed the planner sanity gate with a non-zero plan.

The corrected checkpoint rescore then exposed two deployment-only issues:

1. the production validator did not yet support `--summary-csv`; and
2. a temporary validator copied outside the repository inferred the wrong
   domain-module root from its own location.

The validator was placed under the production repository's `asnets/tools`
directory and the corrected MPrime module was changed to an absolute package
import.  The fixed rescore campaign is resumable: it reuses any complete policy
evaluation log and performs only the missing VAL summary.

Partial results already establish the design decision without consulting test
coverage: across the 18 lineages with summaries at the recorded refresh, epoch
1 was never the corrected-validation optimum.  Later checkpoints improved from
epoch-1 scores as low as 1--22/30 to current best scores as high as 19--30/30.
This directly refutes the old validation set's universal epoch-1 selection.
Two lineages (seed 1073581256, VH off/on) still required release through the
fixed rescore wrapper at the last refresh.

The next primary campaign is therefore a fresh 20-lineage Stage-1 run (ten
seeds, VH off/on) using this exact frozen validation module.  Submission is
gated only by byte-for-byte checksum verification on the cluster and recording
the submitted job IDs in a dedicated manifest ledger.

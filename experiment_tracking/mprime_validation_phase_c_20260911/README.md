# MPrime validation adequacy Phase C

Phase C is a candidate-limited, structurally redesigned validation audit. It
exists because Phase B eliminated saturation but still showed weak Stage-2
validation/test rank agreement.

## Frozen design

- 120 generated candidate problems: 40 per easy/medium/hard tier.
- Planner/VAL gate selects the first ten certified problems per tier under the
  predeclared order and minimum plan-length thresholds in `protocol.json`.
- Forty primary lineages: Stage 1 and validation-led Stage 2, two value-head
  modes, ten seeds.
- At most twelve checkpoints per lineage: the deduplicated union of Phase-B
  replicate-A top five, replicate-B top five, original selected checkpoint and
  final checkpoint.
- Actual candidate inventory: 347 checkpoint evaluations; maximum 480. The
  removed 133 entries were repeated references to the same saved checkpoint
  within a lineage: replicate-A and replicate-B top-five lists overlap, and
  the original selected/final checkpoint can already appear in either list.
  No distinct checkpoint or lineage was removed.
- The deployed checkpoint manifest contains no test-score column. Aggregate
  fixed-test structural support did inform the generator ranges, so this is a
  transductive validation-design audit rather than an independent new test.

## Slurm provenance

| Role | Job | Resources | Purpose |
|---|---:|---:|---|
| Planner/VAL screen | `21183533[0-2]` | 2 CPU, 8 GiB, 4h each | Screen easy/medium/hard candidates independently |
| Gate/finalizer | `21183534` | 1 CPU, 2 GiB, 20m | Freeze manifest/checksums and submit compute work |
| Preflight | `21183541` | 3 requested CPU, 20 GiB, 2h | Completed successfully in 1:21:45 |
| Rescore | `21183542[0-39]` | 40 tasks; 3 requested CPU, 20 GiB, 24h each | Evaluate the 347 candidate checkpoints |

All three screen tasks certified 10/10, finalizer `21183534` completed, and
preflight `21183541` completed successfully. At the targeted 11 September
18:48 IDT check, every task in `21183542[0-39]` was running and had accumulated
about 3:24--3:29 of its 24-hour allocation. Slurm allocates four CPUs per task
despite the three-CPU request, reflecting node allocation granularity; the
observed live allocation is therefore 160 CPUs and 800 GiB.

The source checkpoint inventory with original jobs/logs is
`checkpoint_candidates.csv`. Candidate structure and hashes are in
`candidates.csv`; the final thirty-instance manifest is written only after the
planner/VAL gate succeeds.

Initial screen `21183472[0-2]` failed before planning because a Windows newline
translation made the pre-write text hashes differ from the durable candidate
bytes. No inference or planning was lost. The corrected generator hashes files
after writing. Screen `21183496[0-2]` raced the slow replacement transfer and
repeated the same safe pre-planning failure. Remote checksums were then verified
before final submission `21183533[0-2]`.

## Decision rule

Within each frozen Phase-C validation set, checkpoint selection uses validation
scores only. After Phase C, compare saturation, replicate stability,
validation/test rank agreement and test-selection regret against Phase B, then
adopt the best available frozen validator even if imperfect. Because existing
test results participate in choosing among validation designs, disclose that
design choice as test-informed; it is not untouched confirmatory evidence.
Only then revisit MPrime anchor ranking and decide whether existing checkpoints
merely need re-evaluation or new Stage-2 training is justified.

# TPP first-update Phase B: frozen-replay cross-over

## Scientific question

Phase A exactly reproduced the validation-led TPP VH-off contrast:

- catastrophic checkpoint + its own replay schedule: `20 -> 10/20`;
- stable checkpoint + its own replay schedule: `20 -> 20/20`.

Both runs omitted worker slot 2 and ingested slots `1,0`, so missing-worker
count, missing-worker identity and ingestion order do not explain the
contrast. The outlier nevertheless moved farther: mean per-step policy KL
`0.1421` versus `0.0993`, mean per-step p99 KL `0.8220` versus `0.5241`, and
cumulative parameter movement `2.0949` versus `1.7239`.

Phase B asks whether the collapse follows the starting checkpoint, the replay
content, or their interaction. It reuses the two diagonal Phase-A outcomes and
runs only the missing off-diagonal cells:

| Starting checkpoint | Frozen 60-batch schedule | Meaning |
|---|---|---|
| Catastrophic seed | Stable seed | Does stable replay rescue the susceptible checkpoint? |
| Stable seed | Catastrophic seed | Does catastrophic replay damage a stable checkpoint? |

No new MCTS exploration occurs. Each run uses the exact source Stage-1
checkpoint, its original optimizer state, coefficient `3`, learning rate
`0.0003`, MSE coefficient `0.5`, and exactly 60 frozen optimizer batches.
It deliberately preserves the legacy asymmetric KL forwards (current policy
in training mode; frozen anchor in inference mode). Changing that behavior in
Phase B would confound replay/checkpoint attribution with a new treatment.

## Interpretation frozen before outcomes

| Off-diagonal result | Primary interpretation |
|---|---|
| Bad+stable remains low; stable+bad remains high | Starting-checkpoint susceptibility dominates. |
| Bad+stable recovers; stable+bad collapses | Replay content dominates. |
| Both collapse | Each replay schedule is damaging outside its native checkpoint, or the checkpoints interact differently with foreign targets; inspect per-step diagnostics. |
| Both remain high | Collapse requires the original bad-checkpoint/bad-replay interaction. |
| Intermediate scores | Mixed checkpoint/replay interaction; do not reduce it to one cause. |

These are mechanism diagnostics on one outlier/control pair, not an estimate of
population incidence.

## Integrity gates

1. `freeze_tpp_phase_a_batch_checksums.py --freeze` records path, byte count
   and SHA-256 for all `120` NPZ files and an aggregate hash for each schedule.
   It refuses to overwrite an existing ledger.
2. The real-path compute smoke verifies the checksum ledger and applies the
   complete stable diagonal schedule through the frozen-replay code path.
3. Scientific training is dependency-pending on a successful smoke.
4. Endpoint/probe evaluation is dependency-pending on both scientific arms.
5. Runtime records the source file and SHA-256 for every applied step. All 60
   post-update weight sets are retained so finer probe analysis can be done
   without retraining.

The Stage-1 successful-trajectory probe bank is observation-only. It cannot
choose the treatment, threshold, checkpoint or outcome definition.

## Jobs and resources

- Integrity smoke: one job, `6 CPU / 48 GiB`, two-hour hard allocation.
- Cross-over training: two one-epoch tasks, each `6 CPU / 48 GiB`, two-hour
  hard allocation; maximum concurrent footprint `12 CPU / 96 GiB`.
- Endpoint/probe evaluation: two tasks, each `5 CPU / 20 GiB`, two-hour hard
  allocation; expected runtime is minutes.

The scientific tasks deliberately retain Phase-A training parameters even
though exploration is skipped. The resource request is conservative and small
relative to the original MCTS target-generation jobs.

## Phase C prevention gate

Hard trust-region rollback is not part of Phase B. If the cross-over implicates
checkpoint sensitivity or a damaging interaction, Phase C may test prevention
with two additional one-epoch treatment runs.

The first prevention candidate, Phase C1, is a deterministic-current anchor:
evaluate only the current-policy side of the KL term with `training=False`,
while retaining `training=True` and dropout for the replay-policy loss. This
directly tests the Phase-A finding that the legacy step-0 anchor gradient was
initially aligned with the replay gradient. C1 remains held until Phase B is
interpreted and separately implemented/smoke-tested.

The predeclared stable-only guard uses the empirical 99th percentiles across
the stable Phase-A run's 60 updates:

- per-step mean KL limit: `0.2123709661`;
- per-step p99 KL limit: `1.5138580917`.

A step is excessive when either limit is exceeded. The empirical p99 was
chosen to admit 99% of observed stable updates while rejecting material tail
excursions; it changes one of the 60 stable diagnostic steps, so the stable
control is mandatory. These numbers use no catastrophic-seed test outcome or
probe statistic. Phase C must restore model weights, Adam slots and optimizer
iteration, double anchor protection, and retry the exact batch up to four
times; a still-excessive step is rejected. Phase C is not submitted until the
Phase-B mechanism is reviewed.

## Provenance

- Frozen design: `manifest.csv`.
- Phase-A evidence: `../tpp_first_update_phase_a_20260914/`.
- Training: `scripts/tpp_first_update_phase_b_crossover_train.sbatch`.
- Endpoint/probe: `scripts/tpp_first_update_phase_b_crossover_endpoint.sbatch`.
- Real-path smoke: `scripts/tpp_first_update_phase_b_crossover_smoke.sbatch`.
- Safe submitter: `scripts/submit_tpp_first_update_phase_b_crossover.py`.
- Checksum freezer/verifier: `scripts/freeze_tpp_phase_a_batch_checksums.py`.

No Phase-B or Phase-C job has been submitted by this implementation task.

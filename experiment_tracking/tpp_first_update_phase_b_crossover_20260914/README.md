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

This asymmetry is not automatically a software bug: it can be interpreted as
regularizing a dropout-noisy student toward a deterministic teacher. It does,
however, mean that the measured KL penalizes dropout-induced disagreement as
well as parameter drift. Simply enabling independent dropout in the frozen
anchor is not the preferred control because it would add a second independent
noise source. If a treatment is justified after the crossover, the clean
primary comparison is deterministic current and frozen forwards for the KL
term while retaining dropout for the replay/task loss; a shared-mask paired
dropout KL is a secondary alternative.

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
- Frozen 120-file schedule ledger: `frozen_schedule_checksums.csv`.
- Submitted dependency chain: `submissions.tsv`.

## Current execution

Implementation commit `9732f3a5` was deployed as the detached cluster
worktree `/home/hersco/bershco-nu-asnets/tpp-first-update-b-9732f3a5` so no
shared checkout used by other live jobs was modified. The two 60-batch
schedules were frozen and verified as 120 files with aggregate SHA-256 values
`a834cfeaaba3c481dd97e25650b53b5eaec229e7d67dcf35e020dc12ab3a708c`
and `e659e4161fbb866f9eddf858c643c7387b29aaddf25062ffc03385a5dd090025`.

The first smoke (`21260859`) failed because the isolated checkout did not expose
the compiled TensorFlow operator at the package path used at runtime. The
operator link was repaired and its checksum matched the production build. The
second smoke (`21264717`) passed all five frozen-replay loader tests and
verified all 120 frozen batches, but its native training child exited `-4` on
`ise-cpu-intl-10` before completing the scientific step. Both dependency-gated
chains cancelled cleanly, so neither produced scientific evidence.

The `21265386` chain was cancelled before starting when the parallel
Block Grouping smoke exposed the same native `-4` failure on another already
documented incompatible node. Smoke `21266581` used the complete exact
node blacklist already maintained by the MPrime controller
(`ise-cpu-intl-01,09,10,11,13,27`) without excluding an entire node family,
but exposed a distinct instrumentation defect rather than a node failure:
`run_worker_opt_profiled` tried to attach profiling fields to the deliberately
frozen `ProblemInitData` returned during initialization. It failed before
scientific training; dependents `21266582` and `21266583` cancelled without
work.

Commit `a6c773f2` restricts those mutable fields to `WorkerOutput`, preserving
frozen initialization records unchanged. Smoke `21267359` nevertheless failed
before training because it targeted the shared safe-context checkout rather
than the documented detached cross-over checkout, so the frozen-replay module
was absent. The first repair, smoke `21280249`, selected the same wrong
checkout and inverted the compile path; it verified all 120 batches and then
failed before training on the missing `frozen_replay.py`. Both dependency
chains cancelled without scientific work.

The execution scripts now default to the actual isolated checkout
`/home/hersco/bershco-nu-asnets/tpp-first-update-b-9732f3a5`, restore the
correct nested package paths, and carry the reviewed `a6c773f2`
`spawn_train_worker.py` fix inside that checkout. Replacement smoke
`21280647` then verified the frozen data but exposed one final static-path
error: the nested `asnets/scripts` compile paths were written as top-level
`scripts`. It also failed before training. All six compiled paths now match
files verified in the detached checkout. Replacement smoke `21281126` gates
the two one-epoch cross-over tasks `21281127[0-1]`, which gate endpoint tasks
`21281128[0-1]`. The complete attempt history is retained in
`submissions.tsv`. Phase C remains held and unsubmitted.

Smoke `21281126` passed and released both scientific crossover tasks. Both
one-epoch training arms completed. At the 15 September 00:09 IDT verification,
the stable-checkpoint x bad-replay endpoint had completed at 20/20, while the
bad-checkpoint x stable-replay endpoint was still running. The completed arm
shows that the catastrophic replay schedule is not by itself sufficient to
damage the stable checkpoint. Final attribution still requires the reciprocal
endpoint.

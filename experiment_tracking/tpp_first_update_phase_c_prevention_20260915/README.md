# TPP first-update Phase C: catastrophic-seed prevention

## Gate and scientific scope

Phase B is complete:

| Starting checkpoint | Frozen replay | Endpoint |
|---|---|---:|
| Catastrophic | Catastrophic | 10/20 |
| Stable | Stable | 20/20 |
| Catastrophic | Stable | 3/20 |
| Stable | Catastrophic | 20/20 |

The collapse therefore follows starting-checkpoint susceptibility rather than
the replay schedule alone.  This satisfies the predeclared Phase-C prevention
gate.  It remains a mechanism screen on one outlier/control pair, not an
estimate of population incidence.

Phase C adds exactly two one-epoch treatment runs: the catastrophic checkpoint
with its own frozen 60-batch schedule and the stable checkpoint with its own
frozen 60-batch schedule.  It does not generate new MCTS targets, rerun Stage
1, or continue for 100 epochs.  The successful Stage-1 trajectory probes are
observation-only and cannot choose a threshold or treatment.

## Treatment

The replay/task loss retains the existing `training=True` forward and dropout.
Only the current-policy side of `KL(pi_stage1 || pi_current)` changes to
`training=False`; the frozen Stage-1 anchor is also deterministic.  This
removes the Phase-A mismatch in which the anchor term penalized sampled
dropout disagreement and initially aligned with the replay gradient.

The hard guard measures deterministic pre-update to post-update policy KL on
the exact replay examples proposed for each optimizer step. The compute smoke
first replays the stable control's frozen 60 batches without rollback and
freezes the empirical 99th percentiles of the resulting 60 deterministic
mean-KL and p99-KL step statistics. The two treatment arms then read that
immutable `calibrated_thresholds.json`. This same-statistic calibration avoids
reusing the Phase-A thresholds, whose pre-update side included dropout and was
therefore not on the scale guarded by Phase C.

A proposal is excessive when either limit is exceeded. The corrected trainer
restores all model weights, every Adam slot and the optimizer iteration, keeps
the anchor coefficient fixed at `3`, and retries the same frozen batch at half
the preceding learning rate. It permits two retries (`0.0003`, `0.00015`, then
`0.000075`); a still-excessive third proposal is restored and rejected. The
base learning rate is restored before the next frozen batch.

The replay examples and targets are identical on a retry, but the
`training=True` replay forward resamples its dropout mask because TensorFlow's
RNG state is not rolled back. The guard therefore tests a pragmatic
same-data-batch learning-rate backtrack, not an exact deterministic rescaling
of one frozen stochastic gradient. Endpoint preservation can show that this
guard prevents the observed failure; it cannot by itself attribute the repair
only to the learning-rate factor.

This backtracking schedule is calibrated to the treatment evidence without
using an endpoint. The stable deterministic limits are mean `0.0029394862` and
p99 `0.0107414210`. Across the failed catastrophic run's finite proposals, the
largest base-step p99 was `0.0547682`, or `5.10x` the stable limit. Under the
local quadratic KL-versus-step-size approximation, one halving can remain at
`1.275x` the limit, while two halvings reduce that bound to `0.319x`. Thus a
factor of `0.5` with two retries is the smallest standard halving schedule
that covers the observed scale with margin. The guard still decides from the
real recomputed KL rather than this approximation.

Every attempt records coefficient, requested/effective learning rate, KL
statistics, optimizer iterations, accepted/rejected outcome, source batch path
and SHA-256. All 60 resulting weight states are retained.

These thresholds deliberately remain the predeclared stable-only values even
though the treatment makes the current KL forward deterministic.  They are
not recalibrated from either catastrophic-seed outcome or test probe.

## Integrity and execution chain

1. The existing 120-file Phase-A checksum ledger is verified before smoke and
   before each treatment task.
2. The smoke compiles the changed runtime, runs 13 focused tests, verifies the
   new CLI, executes the complete stable 60-batch deterministic-KL calibration
   path, and freezes `calibrated_thresholds.json`.
3. The two scientific treatment tasks are dependency-pending on smoke success.
4. The two endpoint/probe tasks are dependency-pending on both treatment arms.
5. The idempotent submitter refuses an existing current-attempt ledger and
   requires an explicit deployed checkout. Failed attempts are moved to named
   immutable ledgers before a reviewed replacement is submitted.

## Resources

- Smoke: one job, 6 CPUs, 48 GiB, two-hour hard allocation.
- Treatment: two tasks, each 6 CPUs and 48 GiB, two-hour hard allocation;
  maximum concurrent training footprint 12 CPUs / 96 GiB.
- Endpoint/probe: two tasks, each 5 CPUs and 20 GiB, two-hour hard allocation.
- All jobs use `main` and exclude the six previously observed incompatible
  nodes `ise-cpu-intl-01,09,10,11,13,27`.

## Provenance

- Frozen design: `protocol.json` and `manifest.csv`.
- Backtracking calibration: `backtracking_calibration.json`.
- Phase-A inputs and diagonal results:
  `../tpp_first_update_phase_a_20260914/`.
- Phase-B gate evidence:
  `../tpp_first_update_phase_b_crossover_20260914/`.
- Runtime treatment: `asnets/asnets/policy_anchor_trust_region.py` and
  `asnets/asnets/supervised.py`.
- Smoke: `scripts/tpp_first_update_phase_c_prevention_smoke.sbatch`.
- Training: `scripts/tpp_first_update_phase_c_prevention_train.sbatch`.
- Endpoint/probe: `scripts/tpp_first_update_phase_c_prevention_endpoint.sbatch`.
- Safe submitter: `scripts/submit_tpp_first_update_phase_c_prevention.py`.
- Same-statistic threshold calibrator:
  `scripts/calibrate_tpp_phase_c_thresholds.py`.

## Deployment and submission status

The treatment is deployed in the isolated checkout
`/home/hersco/bershco-nu-asnets/numeric-asnets-tpp-phase-c-20260915`.
`submissions_attempt_history.csv` and the per-attempt TSV files retain every
pre-scientific failure and dependency cancellation.

- `21304220` failed before training because the fresh checkout omitted the
  frozen-replay module.
- `21304435` failed before training because the native TensorFlow operator was
  linked to the package root rather than `asnets/ops/`.
- `21304740` failed before training because the fresh checkout lacked the
  matching `action_history_digest` runtime helper.
- Replacement smoke `21305112` completed the real deterministic calibration
  path, then failed in the final threshold-file helper because it was invoked
  under host Python without NumPy. Treatment `21305113` and endpoints
  `21305114` were dependency-cancelled, so this attempt produced no treatment.
- The helper now implements the same linear quantile using the Python standard
  library. Smoke `21307445` passed in 6m15s and released both one-epoch
  treatment tasks in `21307446[0-1]`. The stable arm completed and retained
  perfect validation, but it used one coefficient-doubling retry. The
  catastrophic arm rejected every finite proposal, carried the doubled
  coefficient forward until it became non-finite, and failed after 21 logged
  batches. Dependent endpoints `21307448[0-1]` were cancelled. This is a real
  treatment-implementation failure, not evidence that trust-region prevention
  fails.

The stable result would be reusable under the corrected implementation only if
its audit contained zero retries, because the no-retry paths are unchanged.
It actually retried step 1 at coefficient `6`, so its `20/20` result is useful
non-regression evidence but is not the matched constant-3 backtracking control;
both arms must be rerun.

No failed smoke produced a scientific treatment result.

Attempt 6 leaves all attempt-5 artifacts untouched. It writes to the fresh
`tpp_first_update_phase_c_prevention_lrbt` directory and uses the corrected
constant-coefficient learning-rate-backtracking treatment. Smoke `21318360`
completed in 4m54s. Both one-epoch treatments `21318362[0-1]` completed: the
catastrophic seed required 11 backtracking retries and the stable control 4;
all 60 optimizer updates were eventually accepted in each arm. Endpoint jobs
`21318364[0-1]` then completed at 20/20 for both arms: catastrophic seed
`1972442430` and stable control seed `2082152039`. The exact stable seed is
confirmed by the deployed manifest and the `21318362_1`/`21318364_1` logs.

This is a successful prevention result for the deliberately selected outlier:
the catastrophic seed improved from its historical first-Stage-2 10/20 to
20/20, while the stable control remained 20/20. It shows that a measured
rollback/backtracking guard can prevent this observed first-update collapse.
It does not estimate how often such collapses occur, and the dropout-resampling
caveat above prevents attributing the repair solely to an exact learning-rate
rescaling. `results_final_20260915.csv` records the endpoint, retry count,
parameter displacement and exact remote audit/log routes. The old
coefficient-doubling attempt remains in place and is explicitly labelled a
failed treatment implementation.

The result must not be described as gradient clipping. Gradient clipping
rescales a gradient before applying it. This treatment first proposes a whole
Adam update, measures its deterministic policy displacement, restores both
network and optimizer state when the displacement is excessive, lowers the
learning rate, and recomputes the same data batch. It also changed the KL
measurement from a dropout-contaminated current-policy forward to a
deterministic current-policy forward. The two changes jointly define the
successful treatment; this selected pair does not identify either one as the
sole repair.

## Remaining causal closure

No additional 100-epoch training is needed to support the narrow conclusion
that the selected collapse was preventable. Two follow-ups would close the
remaining causal questions:

1. Repeat the catastrophic and stable one-epoch treatments while restoring the
   dropout RNG state on every rejected retry, followed by two endpoints. This
   separates rollback/learning-rate scaling from lucky dropout resampling.
2. After that mechanism check, run one guarded epoch plus one endpoint on the
   other eight TPP/off seeds (or all ten under one uniform same-build protocol)
   to estimate non-regression and how often the guard activates. This is a
   population screen, not required to explain the already selected pair.

The first option is four small jobs. The second is sixteen jobs when reusing
the completed selected pair as historical evidence, or twenty jobs for a
strict uniform rerun. These remain documented next steps and are not submitted.

## Reproduction

Deploy the reviewed commit to a new isolated cluster checkout and copy the five
execution scripts plus `scripts/freeze_tpp_phase_a_batch_checksums.py` into
`/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt`.
Then validate without submitting:

```text
python3 /home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt/submit_tpp_first_update_phase_c_prevention.py --checkout /absolute/reviewed/phase-c-checkout
```

After explicit review, the same command with `--submit` creates one smoke job,
one two-task training array and one two-task endpoint array with `afterok`
dependencies.  The submitter records their IDs, dependencies, scripts and
checkout in `submissions.tsv`.

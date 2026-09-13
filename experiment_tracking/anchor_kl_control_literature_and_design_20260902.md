# ANCHOR-KL-CONTROL: literature-grounded nonconstant Stage-2 anchoring

Status: **two adaptive arms completed; controller did not activate**.

## Immediate TPP/off causal screen

The minimal screen is two seeds by two modes, but only **two new training
jobs** are required:

- bad seed `1972442430`: reuse completed constant-anchor job `20684881`; run
  one new adaptive-target-KL arm;
- stable control seed `1963100312`: reuse its completed constant-anchor
  tuning lineage; run one new adaptive-target-KL arm.

The constant arms are historical baselines with complete checkpoints and logs;
rerunning them would spend compute without adding a new treatment. The
adaptive arms must start from the exact matching Stage-1 sources. Block
Grouping and linear-decay arms remain possible follow-ups, not part of this
first causal screen.

The implementation measures post-optimizer KL on the just-used replay batch,
updates the coefficient after each of the 60 replay optimizer steps, logs the
pre/post coefficient and realized KL, and persists the complete controller
state in every checkpoint. This within-epoch feedback is necessary because the
observed TPP collapse is already present at the first saved Stage-2 epoch.
The coefficient is never allowed below the validated constant baseline of 3;
the adaptive arm may strengthen protection and relax back to 3, but cannot
silently become a weaker-anchor treatment.

The target is frozen at **0.1143**, the stable control's first Stage-2 epoch
mean anchor KL (`0.11429792096217474`) from constant job `20553944`. This uses
training diagnostics from the predeclared stable control, not the catastrophic
seed's test outcome. With the PPO tolerance of 1.5, the upper adjustment
threshold is 0.17145. The bad constant run's first-epoch KL was 0.17864, so it
would have crossed that predeclared threshold; the new run tests whether
within-epoch feedback prevents the coverage collapse.

This replaces the informal `ANCHOR-SCHEDULE` idea. The experiment is not
premised on an invented "strong first, then decay to the selected constant"
schedule. It compares two mechanisms that have direct precedents with the
existing constant-anchor baseline.

## Why this experiment exists

The terminal-led epoch-0 audit shows that one full Stage-2 update can already
damage the policy. The clearest individual case is TPP/off seed 1972442430:
Stage-1 is 20/20, Stage-2 epoch 0 is 10/20, and the raw anchor KL is 0.1059
versus a 0.0516 seven-peer mean. Block Grouping/on also loses 4.7 plans on
average at epoch 0. A constant coefficient selected by validation AUC does not
control the *realized* KL of any particular update.

## Related-work basis

1. **PPO adaptive KL penalty.** Schulman et al. describe adapting the KL
   coefficient to a target divergence: divide the coefficient by two when
   measured KL is below `target/1.5`, and multiply it by two when KL is above
   `1.5*target`. This is the primary arm because it reacts to an abnormally
   large realized update instead of assuming a fixed epoch schedule.
   Source: https://arxiv.org/abs/1707.06347 (Section 4).
2. **Kickstarting / teacher-policy distillation.** Schmitt et al. compare
   constant teacher-loss weights, linear decay to zero, and population-based
   adaptation. Their constant schedules plateau, while correctly timed linear
   decay and PBT perform better. This supplies a defensible nonconstant
   distillation-weight arm, but does not prove that decay will prevent our
   first-update collapse. Source: https://arxiv.org/abs/1803.03835.
3. **TRPO trust region.** TRPO constrains mean KL between consecutive policies
   rather than merely adding a fixed penalty. It is the principled reference,
   but a full constrained/natural-gradient implementation is outside this
   small diagnostic. Source: https://proceedings.mlr.press/v37/schulman15.pdf.
4. **KL-regularized fine-tuning.** Stiennon et al. retain a KL penalty to the
   supervised initialization and show that excessive optimization can degrade
   the true objective. This supports retaining an immutable Stage-1 reference
   and explicitly measuring behavior drift. Source:
   https://arxiv.org/abs/2009.01325.

## Frozen screening protocol

The screening unit is a matched Stage-1 source, not a test-selected checkpoint.

Cells:

- TPP/off: catastrophic seed 1972442430 plus one predeclared stable control.
- Block Grouping/on: two predeclared tuning seeds because the mean epoch-0
  collapse is broad rather than confined to one seed.
- Both use the already selected constant coefficient as the baseline.

Arms:

1. **Constant baseline:** the current frozen coefficient.
2. **Adaptive target-KL:** PPO-style multiplicative coefficient controller.
3. **Linear teacher-weight decay:** Kickstarting-style decay from the frozen
   coefficient to zero over a predeclared fraction of Stage-2 updates.

The adaptive target must be calibrated from the two designated tuning seeds'
training/replay states only. Held-out test coverage must not set the target,
the decay horizon, or the winner. Coefficients are clipped to a predeclared
finite range, persisted in checkpoints, and restored exactly on continuation.

Required logs per epoch:

- coefficient before and after the update;
- target KL and measured raw anchor KL;
- policy, value, anchor and total losses;
- unclipped/clipped gradient norm and weight delta;
- validation score and selected checkpoint;
- Stage-1 source and exact training/evaluation log paths.

Selection uses validation stability/AUC plus an explicit epoch-0 non-collapse
gate. Only a winning mechanism is expanded to held-out seeds. Test scores are
reported after selection and never used to tune the controller.

## Implementation gate

Before submission, the generic trainer must support and test:

- `constant`, `adaptive_target`, and `linear_decay` modes;
- coefficient-state checkpoint/restore;
- deterministic coefficient updates from logged KL;
- zero-coefficient behavior identical to the existing implementation;
- an Apptainer smoke test covering fresh training and resume.

The adaptive-target controller and exact checkpoint restoration pass five local
regression tests. Compute-node smoke job `21144340` additionally imports the
compiled TensorFlow operator, verifies the two new CLI options and passes all
five tests inside the production container.

## Submission state — 9 September 2026

The existing constant-anchor evidence is reused, so the live screen contains
only the two adaptive jobs:

- `21144388`: catastrophic outlier seed `1972442430`;
- `21144389`: stable control seed `1963100312`.

Both use six CPUs, 48 GiB, coefficient floor/start `3`, and target KL `0.1143`.
Their exact source checkpoints and stdout paths are recorded in
`anchor_kl_control_retry_submissions_20260909.tsv`.

Initial jobs `21144210` and `21144211` failed before training because a clean
Git worktree omitted the ignored compiled `_asnet_ops_impl.so`. The isolated
checkout now links the checksum-verified production build. This was a deployment
packaging failure, not an optimization result, and cost about one minute per
job. The retry ledger's `retry_of` field preserves that lineage explicitly.

## Final training result — 11 September 2026

Both adaptive jobs completed 100 updates:

| Role | Job | Final cumulative epoch | Final validation | Coefficient path | Controller adjustments |
|---|---:|---:|---:|---:|---:|
| Catastrophic seed 1972442430 | 21144388 | 114 | 21/30 | 3 → 3 | 0 |
| Stable control 1963100312 | 21144389 | 102 | 30/30 | 3 → 3 | 0 |

The realized post-update KL never crossed the controller's upper threshold, so
the adaptive arm was behaviorally identical to a constant-coefficient arm for
all 200 logged updates. This does **not** show that adaptive KL fails; it shows
that this target/floor calibration never applied the treatment. A test-policy
evaluation can quantify the new trajectories, but it cannot establish the
causal benefit of adaptation because no coefficient change occurred. A further
training rerun would require a recalibrated, predeclared target or a true
trust-region rule; it should not be launched merely to rescue this screen.

## Defensible causal follow-up

The failed activation exposed a metric mismatch. The target `0.1143` was
calibrated from an epoch-aggregate raw anchor-KL statistic, while the live
controller monitored post-update KL on each just-used replay batch. Those are
not interchangeable scales: the catastrophic adaptive run's maximum monitored
value was only `0.079579`, and the stable control's was `0.022100`. A lower
threshold must therefore first be calibrated on the *same per-step post-update
statistic* used by the controller.

Three concrete, testable causes remain plausible:

1. The catastrophic seed's first replay/target batch produced an unusually
   large or misdirected parameter update. Freeze and compare epoch-0 replay
   identities, target distributions, gradient norms and parameter displacement
   against the stable seed.
2. Its six worker-timeout warnings, versus two in every peer, may have changed
   which MCTS targets entered the replay batch. Join worker outcome/duration to
   replay membership and rerun only the first update from a complete, frozen
   target batch.
3. Mean batch KL hid large localized changes on a small set of test-critical
   states. Measure per-state KL quantiles and maxima on the Stage-1-success
   trajectories rather than relying only on a replay-batch mean.

A recalibrated lower-target controller could strengthen later optimizer steps
within epoch 0 and test whether subsequent anchoring or recovery improves. It
cannot undo the first offending optimizer step. A hard trust-region or
rollback-and-retry rule is the treatment that directly tests *prevention*: reject
an update whose same-metric KL exceeds a frozen threshold, increase protection,
and recompute it. The smallest next screen remains the catastrophic seed plus
one stable control, with test-policy evaluation after a validation-only choice.

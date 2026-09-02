# ANCHOR-KL-CONTROL: literature-grounded nonconstant Stage-2 anchoring

Status: **held-ready design; no Slurm jobs submitted**.

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

Until those tests pass, this experiment remains held and must not be submitted.

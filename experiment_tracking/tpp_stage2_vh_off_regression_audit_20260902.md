# TPP Stage-2 VH-off regression audit

This is an urgent, static, log-linked audit of the severe TPP held-out-seed
regression. It must not be summarized as ordinary seed noise.

## Exact lineage

- VH: off.
- Seed: `1972442430`.
- Stage-1 validation-selected endpoint: epoch 15, `20/20`, job `20544566`.
- Stage-2 training: job `20684881`, anchor coefficient 3, initialized from the
  Stage-1 epoch-15 snapshot.
- Stage-2 validation-selected endpoint: epoch 12, `9/20`, evaluation job
  `20720723`.
- All nine printed plans are VAL-valid.
- The other eleven instances did not hit a scheduler or per-instance time
  timeout. Each executed 10,000 actions and remained unsolved.

The Stage-2 test learning curve is also persistently poor: epoch 0 scores 10,
epoch 5 scores 13, epoch 10 scores 9, selected epoch 12 scores 9, and the later
every-five scores remain between 3 and 11 through epoch 99. This is therefore
not a one-checkpoint accident that a later test checkpoint repairs. It is both
a genuine Stage-2 policy collapse on this held-out seed and evidence that the
validation-selected epoch does not track the test-set optimum for this lineage.

The exact per-lineage provenance is stored in
`tpp_stage2_vh_off_regression_audit_20260902.csv`. The immediate follow-up is a
validation-versus-test curve comparison for this seed, including per-instance
action traces for the eleven 10,000-action failures. Do not use the test set to
retroactively select a replacement checkpoint.

## Training-trajectory comparison against the other held-out seeds

All eight held-out VH-off jobs ran the same 100 Stage-2 epochs, so the outlier
did **not** receive more optimization updates. It took 30.17 hours versus 16.06
hours on average for the other seven jobs. The extra wall time coincides with
six worker-timeout warnings versus two in every peer, so it reflects more
expensive/failed data generation, not a later epoch limit.

The outlier is also distinct throughout training, rather than only at checkpoint
selection:

- mean validation coverage: 0.754 versus 0.957 across the seven peers;
- mean total training loss: 1.891 versus 1.612;
- mean policy loss: 1.273 versus 1.203;
- mean anchor KL loss: 0.106 versus 0.052.

Epoch 12 was not unusually late—the peer-selected epochs range from 0 to 92.
It was chosen because its validation score was 29/30, above epoch 0's 27/30.
The best observed every-five test checkpoint was epoch 5 at 13/20, four plans
better than the selected epoch but still seven below the untouched Stage-1
policy. Validation misranking therefore worsened the endpoint, but cannot
explain the underlying collapse: the first saved Stage-2 checkpoint was already
only 10/20.

The complete peer comparison, including original training-log paths, is in
`four_domain_preservation/tpp_stage2_vh_off_training_comparison_20260902.csv`.

## Why one seed can collapse while the other nine do not

The training log rules out a saturated validation curve: its 30-problem
validation coverage fluctuates roughly from 0.57 to 0.87, and epoch 12 was
selected at about 0.87. The three leading explanations are therefore:

1. **Seed-specific catastrophic forgetting.** Stage-2 MCTS targets and replay
   sampling create a nonlinear optimization trajectory. This seed left the
   stable Stage-1 basin while the other seeds did not.
2. **Validation/test misranking.** Thirty validation problems are enough to
   choose epoch 12 within that lineage, but not enough to expose the severe IPC
   test failures. The low test scores persist throughout the later curve, so
   selecting a different existing epoch does not solve the collapse.
3. **Off-support anchor weakness.** The KL anchor constrains outputs on states
   actually revisited during Stage-2 training/replay. It does not guarantee
   preservation on test-relevant states absent from that distribution. A
   coefficient of 3 can therefore preserve nine seeds while failing badly on
   a rare held-out trajectory.

This is a tail-risk result, not evidence that the test evaluator failed. A
defensible follow-up compares replay-state coverage, KL drift, and
validation/test per-instance membership for this seed against one stable
matched seed; it does not discard the seed or choose a checkpoint using test
coverage.

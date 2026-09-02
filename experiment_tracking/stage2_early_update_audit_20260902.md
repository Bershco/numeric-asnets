# Stage-2 early-update stability audit

This audit distinguishes two effects that must not be conflated:

1. **Early refinement instability:** the Stage-2 epoch-0 snapshot is saved only
   after the first complete explore/replay/optimizer update. A difference between
   the Stage-1 source and Stage-2 epoch 0 is therefore the effect of one Stage-2
   training epoch, not an initialization mismatch.
2. **Checkpoint-selection misranking:** validation may later choose a checkpoint
   whose test score is below the best test score observed in the frozen
   every-five curve. This can amplify an existing collapse but cannot explain a
   collapse already visible at epoch 0.

The complete terminal-led mainstream epoch-0 audit contains ten seeds in every
domain/VH cell. Its seed-level rows, exact epochs, training job IDs, evaluation
job IDs, and absolute original log paths are in
`stage2_epoch0_seed_audit_20260902.csv`; aggregates are in
`stage2_epoch0_summary_20260902.csv`. The available validation-led epoch-0 logs
are retained too, but that older every-five release is incomplete and is not
used to make ten-seed claims.

The complete terminal-led audit shows broad early damage in Block Grouping
VH-on (mean change -4.7 plans; seven of ten seeds lose at least five) and
Counters VH-on (mean change -6.9; five of ten lose at least five). The severe
TPP/off held-out seed is therefore a tail case of a broader early-update
stability problem, not an entirely isolated phenomenon.

## TPP/off seed 1972442430

The Stage-1 selected source scored 20/20. Stage-2 epoch 0 scored 10/20, so the
first refinement update immediately lost ten plans. The best observed
every-five test checkpoint was epoch 5 at 13/20; validation selected epoch 12,
which scored 9/20. Validation misranking cost about four plans relative to the
observed test-best checkpoint, but the first update had already caused the
larger failure.

Its raw anchor KL mean was 0.1059, versus 0.0516 for the seven other held-out
VH-off peers: 2.05 times the peer mean, 2.58 peer standard deviations above it,
and the largest value in the group. With coefficient 3, that raw term contributes
about 0.318 to the mean objective versus about 0.155 for peers. This is strong
evidence that the lineage diverged unusually far from its Stage-1 anchor on
replay states despite the same coefficient. It is a symptom and risk marker,
not proof that KL itself caused the collapse: the replay-state distribution and
MCTS targets may have driven both the large KL and the bad policy.

## Held continuation: ANCHOR-SCHEDULE

A stronger early anchor that decays to the already selected coefficient is a
defensible hypothesis, but not yet an established remedy. A fixed anchor only
penalizes average disagreement on sampled replay states; it neither caps a
single update nor protects unseen test states. The staged experiment should:

- compare constant anchor, strong-then-decaying anchor, and an update-size guard;
- use the TPP outlier plus a stable matched TPP seed, and Block Grouping VH-on
  where the mean first-update loss is broad rather than seed-specific;
- report epoch-0 policy loss, raw/weighted KL, gradient norm, replay-state
  membership, validation/test coverage, and recovery over later epochs;
- remain held until its exact schedule and confirmatory seed count are frozen.

Do not use test coverage to choose the decay schedule. A small diagnostic can
test whether the first-update drop is prevented; a later held-out confirmation
is required before changing the mainstream protocol.

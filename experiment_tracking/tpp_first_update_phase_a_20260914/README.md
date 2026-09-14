# TPP VH-off first-update causal audit

## Why this exists

The validation-led TPP VH-off seed `1972442430` drops from a 20/20 Stage-1
policy to 10/20 after the first saved Stage-2 epoch and never recovers beyond
13/20. The matched held-out seed `2082152039` remains 20/20 throughout Stage
2. Both original jobs lost one exploration worker at epoch 0, so warning count
alone cannot explain the collapse.

The three leading, falsifiable explanations are:

1. **Replay/gradient basin:** the outlier's first MCTS targets oppose the
   Stage-1 policy or create an unusually large/conflicting update.
2. **Omitted-target identity/content:** both cells lost one worker, but losing
   a particular PDDL/target shard may bias the ingested replay. The relevant
   variable is the omitted content, not the count of warnings.
3. **Localized destruction hidden by mean KL:** average anchor KL can remain
   moderate while decisions on a small set of test-critical Stage-1 states
   flip.

## Phase A: submitted diagnostic treatment

Phase A contains exactly two one-epoch Stage-2 jobs: the outlier and matched
stable control. Each starts from its exact validation-selected Stage-1
checkpoint, uses VH-off, anchor coefficient 3, `hadd-astar`, three workers,
and performs one exploration/replay epoch with exactly 60 optimizer updates.
It does not rerun Stage 1 and does not continue to 100 epochs.

The opt-in instrumentation records without resampling:

- expected/completed/missing worker slots, deterministic worker seeds, PDDL
  checksums, and actual replay-ingestion order;
- all 60 sampled batch arrays in numbered NPZ files plus ordered row hashes,
  target entropy/source and target-versus-network disagreement;
- separate policy and weighted-anchor gradient norms and cosine;
- raw/clipped total gradient, parameter displacement, Stage-1-anchor KL and
  pre-update-to-post-update policy KL as mean/p90/p99/max.

Two small Stage-1 policy jobs freeze the already-successful trajectories as an
observation-only probe bank. Two dependent epoch-0 policy jobs report test
coverage and measure KL/action flips on the frozen Stage-1 states. The probe
bank is diagnostic only: its test states must never set a training threshold,
select a treatment or select a checkpoint.

## Phase B: frozen-replay cross-over causal isolation

The earlier adaptive-KL screen did not change treatment: coefficient 3 never
changed because the monitored statistic never crossed 0.1143, and the
epoch-aggregate calibration did not match the per-step controller statistic.

Because Phase A reproduced the endpoints exactly (10/20 outlier and 20/20
control), Phase B crosses the two frozen 60-batch schedules with the two
Stage-1 checkpoints. It adds only the two off-diagonal one-epoch runs plus
their endpoint/probe evaluations. No new MCTS targets are generated. This
separates replay-content effects, checkpoint susceptibility and their
interaction. The 120 source NPZ files must be frozen by filename and SHA-256
before the jobs become eligible.

## Phase C: held hard trust-region prevention

Hard rollback is prevention, not causal isolation, and is therefore Phase C.
It remains gated on the Phase-B mechanism result. The proposed rule restores
both model and Adam state after an excessive step, increases anchor
protection, and recomputes the exact same batch. Thresholds are calibrated
only from stable Phase-A training-state statistics; test-probe states remain
observation-only.

## Leading anchor-failure mechanism found in Phase A

The Stage-1 anchor did not begin at zero KL even though the trainable and
anchor weights were identical. The current policy used by the replay loss and
the KL term was evaluated with `training=True` and dropout `0.1`, while the
frozen Stage-1 anchor used `training=False` and dropout `0`. The resulting
step-0 anchor KL was `0.311` for the catastrophic seed and `0.216` for the
stable control. More importantly, the weighted anchor gradient initially
aligned with the replay-policy gradient rather than opposing it (cosine
`+0.831` catastrophic, `+0.539` stable). The first parameter displacement was
also larger for the catastrophic seed (`0.0617` versus `0.0425`).

Consequently, coefficient `3` acted partly as a stochastic dropout-consistency
penalty and did not guarantee protection from the sampled first update. This
is a concrete, testable mechanism, not yet a causal conclusion. Phase B keeps
this legacy behavior unchanged so that its checkpoint/replay cross-over
explains the historical collapse. A later Phase-C1 treatment may change only
the current-policy side of the KL calculation to `training=False`, while
leaving the replay policy loss and its dropout untouched.

## Resources

- Phase-A training: 2 tasks × 6 CPU × 48 GiB, four-hour hard allocation.
- Stage-1 probe capture: 2 tasks × 5 CPU × 20 GiB, two-hour hard allocation.
- Dependent endpoint/probe evaluation: 2 tasks × 5 CPU × 20 GiB, two-hour
  hard allocation.
- Maximum training footprint: 12 CPU / 96 GiB.

Compute smoke `21252717` completed in 5m39s and verified all five controller
tests, one real instrumented optimizer step, a frozen replay-batch artifact,
and the Stage-1-to-endpoint trajectory-probe round trip. Scientific training
array `21253011[0-1]` and Stage-1 probe array `21253014[0-1]` are submitted.
Endpoint/probe array `21253018[0-1]` was submitted dependency-pending on both arrays.
Exact job IDs, resources and output paths are recorded in `submissions.tsv`.

## 14 September final Phase-A result

Both one-epoch training tasks, Stage-1 probe captures and endpoint evaluations
completed. The stable-control endpoint scored **20/20** and the catastrophic
endpoint scored **10/20**. Phase A therefore reproduced the historical first-
epoch contrast exactly.

The one-epoch audit already narrows the causal story:

- both seeds lost worker slot 2 and ingested the surviving slots in order
  `1,0`; the simple omitted-slot/count and completion-order explanation is
  therefore not supported;
- the catastrophic seed has mean per-step policy KL `0.1421` versus `0.0993`
  for the control, maximum step KL `3.7681` versus `2.1516`, and cumulative
  parameter displacement `2.0949` versus `1.7239`;
- the catastrophic seed's target/Stage-1 argmax disagreement rate is lower,
  not higher (`41.0%` versus `48.5%`), so a simple excess-disagreement-count
  explanation is not supported;
- its replay is more diverse and more entropic in this run: 36 unique
  observations and 41 unique targets versus 27 and 36, with mean target
  entropy `0.1870` versus `0.1362`.

These are matched descriptive diagnostics, not a statistical population
comparison. The exact endpoint reproduction makes the frozen-replay Phase-B
cross-over eligible. Hard rollback remains Phase C and is gated on the
cross-over mechanism rather than launched as an unexplained treatment.

Exact metrics, job identities and source paths are in
`phase_a_interim_results_20260914.csv`.

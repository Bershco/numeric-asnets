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

## Phase B: held behind a causal gate

The earlier adaptive-KL screen did not change treatment: coefficient 3 never
changed because the monitored statistic never crossed 0.1143, and the
epoch-aggregate calibration did not match the per-step controller statistic.

Phase B is therefore a distinct hard trust-region/rollback test, not part of
Phase A. It is eligible only if Phase A reproduces the outlier and identifies
a stable-control-calibrated per-step mean/q99 separation. It would add only
two one-epoch jobs plus two endpoint evaluations. It must replay the frozen
Phase-A schedules and restore both model and Adam state when rejecting a step.
Thresholds must be frozen from the stable control only; the outlier's test
score and probe bank cannot tune them.

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
Endpoint/probe array `21253018[0-1]` is dependency-pending on both arrays.
Exact job IDs, resources and output paths are recorded in `submissions.tsv`.

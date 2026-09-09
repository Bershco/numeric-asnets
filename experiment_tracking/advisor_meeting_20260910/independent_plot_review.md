# Context-free advisor-plot review — 2026-09-09

The reviewer received only a short description of Numeric ASNets, Stage 1,
Stage 2, VH modes, fixed/PW MCTS, and the five `before_review` PNGs. It did not
inspect the repository or receive the experiment history.

## What the reviewer understood

- The best observed coverage gains are in Drone, FO Counters and Counters;
  Delivery, TPP and Zenotravel remain near ceiling.
- Stage 2 is strongly domain/configuration dependent rather than uniformly
  beneficial.
- Fixed MCTS most consistently helps Drone/VH-on and FO Counters, while Block
  Grouping is often negative and Counters is high variance.
- PRESERVE-3 is stable for most seeds but has rare, serious TPP regressions.
- MPrime's current validation scores cannot rank Stage-2 checkpoints.

## Problems identified in the first version

1. The scorecard clipped method labels and the conclusion; it did not state the
   axis unit, denominator, seed aggregation, or that the green bar was a
   heterogeneous best-of-method result.
2. The learning-dynamics plot lacked real epoch ticks and an explicit y-axis;
   the joined axis made Stage 1 and Stage 2 look like one continuous run.
3. The MCTS forest was too dense, used internal configuration names, omitted
   the effect unit, and did not clearly define the Holm family.
4. PRESERVE-3 did not define its scope or axis, overlapped zero-valued seeds,
   and incorrectly called observations “cells.” The conclusion also hid the
   separate TPP/on 20-to-15 regression.
5. The MPrime plot did not state the score denominator, aggregation unit, or
   denominator behind “100% tied.”

## Changes made

- All axes, units, denominators, aggregation rules and seed counts are stated.
- The scorecard is wider, unclipped, and explicitly labelled exploratory.
- Both full learning curves are shown on adjacent stage-local epoch axes with a
  dotted boundary and a stated validation-led initialization rule.
- MCTS results are split into Stage-1 and Stage-2 figures, use readable method
  labels, and define the paired unit, CI, effect unit and Holm family.
- PRESERVE-3 is validation-led only, repeated zeros are stacked, loss/neutral/
  gain have a legend, and both large TPP regressions are visible.
- MPrime labels both endpoints and defines selection regret as the hindsight
  test-best score minus the validation-selected checkpoint's test score.

The immutable first versions remain in `before_review/`; the revisions are in
`after_review/`.

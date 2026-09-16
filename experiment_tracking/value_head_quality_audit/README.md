# Value-head quality side experiment

## Goal

RQ3 tests whether enabling the value head improves Stage-2 policy refinement;
RQ4 tests whether it changes the benefit of MCTS. Neither directly asks
whether the learned scalar value is calibrated or ranks successor states well.
MCTS currently blends learned value and an ENHSP estimate, normally at 0.5, so
MCTS coverage cannot isolate learned-value quality.

## Phase V1: offline calibration and sibling ranking

Use VH-on Stage-1 and validation-led Stage-2 checkpoints on the same frozen
validation states. Enumerate applicable successors in a batch and record raw
learned values. Keep three label sources separate:

1. held-out MCTS/replay target — target consistency, but partly circular;
2. deterministic continuation outcome and remaining length — practical utility;
3. bounded ENHSP result/estimate — independent external planner evidence, not
   guaranteed optimal value.

Metrics: MAE/MSE and calibration bins; Spearman/Kendall sibling ranking;
pairwise ordering; top-value child regret and agreement with externally best
successor; dead-end/unsolved discrimination where meaningful; and matched
Stage-1 to Stage-2 change. Report per seed/domain rather than pretending every
state is independent.

Recommended pilot: Rover, Drone and FO Counters; two predeclared VH-on seeds
per domain; Stage-1 and validation-led Stage-2 checkpoints. This is 12
checkpoint tasks, about 4 CPU and 40–120 GiB each, with an 8–12 hour bound
depending on labels. Avoid Block Grouping initially because successor
generation is expensive and Counters because extremely long trajectories can
dominate. Add MPrime after endpoints freeze.

## Phase V2: raw-value-greedy inference

Only proceed if V1 shows useful sibling ranking. At each external step:

1. enumerate applicable actions and batch-generate successors;
2. evaluate raw learned value only;
3. mask known terminal non-goals and prioritize an immediate goal;
4. choose maximum value, using stable action ID only for a remaining exact tie.

Compare the same checkpoint under policy argmax, raw-value greedy, optionally
ENHSP-only greedy, and optionally the existing 0.5 blend. Report coverage,
action count, successors generated, runtime and failure mode. This diagnostic
is feasible because batched successor generation and network inference already
exist; implementation plus tests is estimated at one to two development days.
The validation screen would add 12 evaluation tasks at roughly 2 CPU / 120 GiB
each.

## Interpretation limits

Planner estimates are not optimal ground truth; label sources must not be
pooled. Global correlations can be driven by instance difficulty, so sibling
ranking is more informative. Policy-only states have survivorship bias; sample
off-policy/planner-reachable states too. A failed value-greedy trajectory can
reflect compounding distribution shift or successor-generation cost as well as
poor calibration.

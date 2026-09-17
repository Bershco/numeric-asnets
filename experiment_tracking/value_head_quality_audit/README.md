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

### V1 execution plan and resources

1. Freeze two seeds, exact Stage-1/Stage-2 checkpoint hashes and a state-source
   mixture per domain before inspecting value scores.  Include policy-reached,
   planner-reached and limited off-policy states so the audit is not restricted
   to trajectories that the policy already survives.
2. Materialize a checksumed state manifest and cache successor identities once.
   Cache each label family separately; never turn a timeout from one labeler
   into a numeric target from another.
3. Run the twelve checkpoint tasks, then aggregate at the lineage/domain level.
   State-level observations within a trajectory are not treated as independent
   seeds.
4. Accept the V1 gate only if sibling ordering, top-child regret and dead-end
   discrimination are reproducible across both seeds in at least two domains.
   Global correlation alone is insufficient.

Maximum simultaneous request: 48 CPU and 480 GiB–1.44 TiB, depending on the
domain-specific memory preflight.  With full concurrency the cluster hard
bound is 8–12 hours after manifests and label caches are ready.  Preparation,
implementation and smoke testing are estimated at one development day.  Every
aggregate CSV must include the checkpoint, state-manifest and raw label-log
paths.

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

### V2 execution plan and resources

Implement the new inference mode behind an explicit flag, leaving policy and
MCTS behavior unchanged.  Unit-test batching, goal/terminal precedence,
numeric minimization/maximization sign, successor/action identity and stable
ties.  Smoke-test one short instance before releasing the twelve validation
tasks.

Development plus tests is estimated at one to two working days.  The first
screen requests at most 24 CPU / 1.44 TiB concurrently and uses a 6–12 hour
task bound after a short runtime preflight.  It is followed by policy-argmax
and value-greedy comparisons on the same checkpoints and instances; no MCTS
run is needed to answer the narrow diagnostic.

V2 is normally gated by V1.  The gate may be deliberately overridden only by
freezing that decision before viewing V2 test outcomes and labelling the run
as an exploratory inference diagnostic.  Skipping a failed V1 gate cannot be
used to claim that the value head is calibrated; at most it can reveal a
surprising trajectory-level behavior worth investigating.

## Interpretation limits

Planner estimates are not optimal ground truth; label sources must not be
pooled. Global correlations can be driven by instance difficulty, so sibling
ranking is more informative. Policy-only states have survivorship bias; sample
off-policy/planner-reachable states too. A failed value-greedy trajectory can
reflect compounding distribution shift or successor-generation cost as well as
poor calibration.

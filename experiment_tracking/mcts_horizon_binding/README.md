# Remaining-horizon experiments

`MCTS-HORIZON` is the completed Drone 750-action aware/unaware non-result.
`MCTS-HORIZON-COUNTERS` is the live 10,000-action Counters efficacy pilot.
They are separate experiments.

## Cutoff-counter semantics and verification

At external action `t`, aware search receives `remaining_horizon=max_len-t`.
Selection records a cutoff whenever it reaches a non-terminal node at search
depth greater than or equal to that remaining horizon. The counter is
cumulative across every root search performed by a worker and is printed in
the final compact MCTS depth summary.

The exact boundary behavior is covered by two regression tests:

- remaining horizon 1 forbids expansion beyond the root's children and records
  depth-1 cutoffs;
- remaining horizon 2 records visits reaching a grandchild at depth 2 and
  records no cutoff below that boundary.

The new depth-2 regression passed locally on 2 September 2026. Therefore, if
an aware run genuinely executes action 9,998 of a 10,000-action trajectory and
its twenty/seventy-simulation searches reach depth 2, its final summary must
contain a nonzero cutoff. A selected root child's visit count alone does not
mathematically guarantee that every visit reaches a grandchild: a simulation
stops at its first expandable frontier and visits can spread among children.
Nevertheless, repeated seventy-simulation searches this late should normally
reach depth 2, so a completed 10,000-action aware trajectory with zero cutoffs
would be treated as an instrumentation/propagation defect and audited, not
accepted casually.

The four aware Counters jobs had each recorded only 22 terminal instances at
the 13:35 IDT audit. Their most recently completed successful trajectories were
218--1,105 actions; none of that terminal evidence approached 10,000 actions.
Their zero cutoff counts are therefore expected so far. Long current instances
have not emitted terminal completion records yet and cannot be classified as
10,000-action outcomes before they finish.

Original evidence is linked by `counters_submissions.tsv`,
`counters_manifest.csv`, each job's completion JSONL under
`/home/hersco/training_new_domains/2026-08-31/mcts_horizon_counters/completion/`,
and the evaluation logs in the parent directory.

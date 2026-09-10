# Remaining-horizon experiments

**Lifecycle note (10 September 2026): both campaigns are complete negative
experiments.** References below to “live” or “running” preserve the historical
execution timeline; they are not current scheduler status. `MCTS-HORIZON` is
the Drone 750-action aware/unaware non-result and `MCTS-HORIZON-COUNTERS` is the
10,000-action Counters efficacy pilot. They are separate experiments.

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

At the 16:22 IDT audit, 138 completion-ledger records existed and all 138 were
successful trajectories. The maximum completed plan length was 1,105 actions;
there was not yet a completed 10,000-action failure, OOM-terminal record, or
other terminal long-trajectory record in the ledger. The aware logs do show
local search depths as high as 3,059 edges, but every completed success still
finished far from the remaining-action boundary. Their zero cutoff counts are
therefore expected so far. Long current instances have not emitted terminal
completion records yet and cannot be classified before they finish.

The action count and search depth are deliberately different quantities. The
evaluation worker increments `step` once per externally executed action and
passes `remaining_horizon=max_len-step` into a fresh root search. The MCTS depth
counter is `len(path)-1` inside that one root search. Consequently, a 10,000-step
external trajectory can be assembled from 10,000 searches whose individual
depths are much smaller. Near step 9,998, however, any aware simulation that
reaches depth two must increment the cutoff counter; the two-step regression
test verifies exactly that boundary.

The 138 row-level records and their absolute completion-ledger/log pointers are
in `horizon_completion_records_20260902.csv`.

Original evidence is linked by `counters_submissions.tsv`,
`counters_manifest.csv`, each job's completion JSONL under
`/home/hersco/training_new_domains/2026-08-31/mcts_horizon_counters/completion/`,
and the evaluation logs in the parent directory.

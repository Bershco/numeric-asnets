# ACT-HISTORY-ABLATION — held design

## Question

Does the cumulative grounded-action count feature help direct policy inference
while harming MCTS through history-dependent network outputs combined with
physical-state transpositions?

## Pilot domains

- **Drone:** cyclic/dead-end search case where action-history aliasing may alter
  safety and route selection.
- **Counters:** strongly reversible case with the largest observed contextual
  node multiplier.
- **TPP:** mostly-monotone control.  `bought` is monotone nondecreasing and
  `on-sale` is monotone nonincreasing; only `drive` can revisit locations.
  TPP is not claimed to be strictly acyclic, but it has substantially more
  irreversible goal progress than Block Grouping, Counters, Delivery or
  Zenotravel.

## Required experiment

Disabling `USE_ACT_HISTORY_FEATURES` changes the network input dimension and
therefore requires fresh Stage-1 training.  Existing checkpoints cannot be
reused.  Compare matched seeds for:

1. policy inference;
2. fixed MCTS;
3. Kmin=3 progressive widening;
4. training and inference runtime;
5. retained nodes, timeouts and OOMs.

Every MCTS comparison reports 30-minute, 2-hour and 6-hour conservative
coverage.  The experiment remains explicitly held.

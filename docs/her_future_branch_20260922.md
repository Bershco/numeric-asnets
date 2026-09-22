# HER future branch

This branch contains an explicit-opt-in implementation of future achieved-goal
relabeling.  Ordinary runs remain on `--mcts-her-strategy off --her-k 0` and
retain their historical input bytes.  HER requires
`--mcts-her-strategy future --her-k K`; it refuses a simultaneous Tree-Sampling
K so the first screen contains no K x HER interaction.

## Frozen initial contract

- Representable achieved goals are nonempty positive dynamic proposition masks
  that become true after the relabelled state. Numeric goal expressions are
  rejected and counted; they are never silently approximated.
- Every input in an HER-enabled replay bucket has a proposition-goal tail.
  Ordinary trajectory rows receive the original static goal mask, while HER
  rows receive the sampled achieved-goal mask. The network consumes the tail at
  runtime and the versioned replay compatibility signature includes the tail
  width, strategy, and contract name.
- The HER policy target is the action actually taken on the recorded episode
  segment. The value target is the explicitly declared Monte Carlo success
  return for that demonstrated segment. The current screen is undiscounted
  (`gamma=1.0`), matching the campaign's success-value scale.
- Replay provenance is retained as `TRAJECTORY`, `ENHSP_PLAN`, `HER_FUTURE`, or
  `TREE_SAMPLE` through insertion and eviction, and emission/rejection counts
  are logged.

## Release gate

No scientific HER arm may be released until the exact domain instances pass a
container smoke recording `(goal_props, goal_flnts)`, emit at least one
`HER_FUTURE` row, emit zero `TREE_SAMPLE` rows, and demonstrate that a traced
`PropNetwork` consumes distinct goal tails. A zero-emission smoke blocks the
domain rather than producing a mislabeled no-op arm.

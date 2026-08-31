# MCTS SAFE-CONTEXT experiment

SAFE-CONTEXT keeps physical-state identity for cycle detection but keys MCTS
nodes, cached network predictions, priors, values, visits, and children by
`(physical_state, action_history_digest)`. This removes action-history aliasing
without storing a growing action sequence.

All twenty arms and all ten matched Drone pairs are terminal.

| VH | Matched n | Physical mean | Contextual mean | Paired change | 95% paired t CI | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| Off | 5 | 8.80/20 | 8.40/20 | -0.40 | [-1.08, 0.28] | .5000 |
| On | 5 | 10.80/20 | 7.40/20 | -3.40 | [-6.64, -0.16] | .0625 |
| Combined (exploratory pooled) | 10 | 9.80/20 | 7.90/20 | -1.90 | [-3.60, -0.20] | .015625 |

Holm correction across the two VH tests gives adjusted p=.500 for VH-off and
p=.125 for VH-on; neither individual VH result is confirmatory at .05. The
pooled row is descriptive because it mixes VH modes. Context-specific
statistics remove a real representational alias but reduce transposition
sharing and harm coverage, especially with VH-on. The behavioral arm is not
promoted. Exact job, score, runtime, and log pointers are in
`live_reconciliation_20260831.csv`.

The 30-minute and two-hour post-hoc cutoffs do not rescue contextual nodes:

| VH | Arm | Full/6h mean | 30m mean | 2h mean |
|---|---|---:|---:|---:|
| Off | Physical | 8.80 | 8.80 | 8.80 |
| Off | Contextual | 8.40 | 8.40 | 8.40 |
| On | Physical | 10.80 | 10.20 | 10.80 |
| On | Contextual | 7.40 | 7.20 | 7.40 |

For VH-off, contextual nodes create nine additional per-instance timeouts. For
VH-on, they create fewer timeouts but nineteen additional ordinary-unsolved
trajectories. Therefore the negative result is not merely scheduler censoring.

Operational decision: preserve the implementation and evidence, but do not
enable contextual nodes in production or future primary experiments.  A domain
could theoretically benefit when action-history prediction disagreement is
large and physical-state sharing is mostly harmful, but this campaign supplies
no positive matched cell.  Any future revisit must be a small gated diagnostic,
not an unannounced algorithm replacement.

# MCTS SAFE-CONTEXT experiment

SAFE-CONTEXT keeps physical-state identity for cycle detection but keys MCTS
nodes, cached network predictions, priors, values, visits, and children by
`(physical_state, action_history_digest)`. This removes action-history aliasing
without storing a growing action sequence.

At the 31 August 2026 03:50 IDT snapshot, nineteen of twenty arms and nine
matched Drone pairs are terminal. The remaining job is the contextual VH-off
arm for seed 1073581256.

| VH | Matched n | Physical mean | Contextual mean | Paired change | 95% paired t CI | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| Off | 4 | 9.25/20 | 9.00/20 | -0.25 | [-1.05, 0.55] | 1.0000 |
| On | 5 | 10.80/20 | 7.40/20 | -3.40 | [-6.64, -0.16] | .0625 |
| Combined | 9 | 10.11/20 | 8.11/20 | -2.00 | [-3.92, -0.08] | .03125 |

This is a provisional negative result concentrated in VH-on. Context-specific
statistics remove a real representational alias but reduce transposition
sharing and currently harm coverage. Do not promote the contextual arm before
the tenth pair is terminal and resource/node-multiplication evidence is
reviewed. Exact job, score, runtime, and log pointers are in
`live_reconciliation_20260831.csv`.

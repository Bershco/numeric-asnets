# Block Grouping policy-prior tie-break causal screen

This is a two-job, one-seed diagnostic extension of the Counters tie-break
finding. It is not a new RQ score and is not a domain-wide confirmation.

The exact Stage-1 validation-selected VH-off checkpoint for seed `1963100312`
solved four instances that the historical narrow fixed-MCTS run did not:

- test positions 19 and 20: `instance_100_20_5_1.pddl` and
  `instance_100_25_6_2.pddl`;
- test position 6: `instance_11_40_10_1.pddl`;
- test position 1: `instance_5_25_6_1.pddl`.

Both new jobs use the same current build and differ only in final root-visit
tie resolution: historical stable action ID versus the candidate policy prior.
They retain narrow search (five children, 20 simulations), PUCT 0.1, estimator
mixture 0.5, one worker, six-hour per-instance timeout and 10,000 external
actions. Action-level tracing records whether any rescue is actually associated
with an equal maximum-visit tie. Each task requests 2 CPUs, 120 GiB and 30 hours.

The four targets and their policy/MCTS provenance are frozen in `manifest.csv`.
A full Block Grouping confirmation is allowed only if this selected-failure
screen shows a credible rescue mechanism; PW70 is not implicated by this test.

Across the complete ten-seed Stage-1 VH-off evidence, policy solves 163/200
seed-instance pairs and historical narrow MCTS solves 154/200. Of the 46 MCTS
failures, 16 are policy-success/MCTS-failure cases that this mechanism could
plausibly recover; 30 are shared policy/MCTS failures that a preservation
tie-break is not expected to solve. MCTS also gains seven cases that policy
misses. The four selected targets are therefore genuine recovery opportunities:
they are 4/16 of all VH-off policy-success/MCTS failures, and all four historical
runs ended by exhausting 10,000 external actions.

The analogous VH-on totals are 159/200 policy successes and 162/200 MCTS
successes. Only 4/38 MCTS failures are policy-success/MCTS failures, while 34
are shared failures and MCTS gains seven policy failures. That makes VH-off the
correct first transfer test.

## Status

The strengthened compute smoke passed all 23 tie-break tests and exercised the
real evaluator. At the 13 September 2026 23:45 IDT snapshot, array `21237401`
was running both same-build arms. Each rule had durably classified the same
first three targets as ordinary unsolved outcomes (0/3); the fourth target was
still active. This is an interim matched result, not yet the final 0/4 outcome.
The exact node `ise-cpu-intl-13` is excluded after a native evaluator exit `-4`;
the live tasks are on `ise-cpu-intl-23` and `cs-cpu-11`.

PW70 remains a separate question. In the two-seed PW70 Block Grouping screen,
17/27 six-hour PW failures are policy-success/PW-failure cases, so its coverage
loss is not merely inherited policy failure. All 17 are recorded timeouts, and
the old PW traces do not establish tied root visits as the cause. The defensible
sequence is to finish this fixed-search screen, inspect its root traces, and
only then run a small traced PW70 transfer if the same mechanism is present.

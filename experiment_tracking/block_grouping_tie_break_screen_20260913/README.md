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

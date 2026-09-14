# Current experiment status

Updated: 2026-09-14T19:05:33+03:00

This is the only canonical changing Markdown status page. Dated status files are
historical snapshots. Full RQ tables, methods and conclusions are in
`experiment_tracking/advisor_followup_20260910/README.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
| Block Grouping PW70 mechanism trace | running + pending | 4 | 8 | 480.0 GiB |
| Counters Stage1 strict tie-break confirmation | running | 14 | 84 | 1680.0 GiB |
| MPrime Phase-B-A Stage1 PW70 confirmation | running | 15 | 90 | 1800.0 GiB |
| MPrime Phase-B-A anchor rescore successor | running | 7 | 28 | 140.0 GiB |
| MPrime Phase-B-A rescore controller | pending | 1 | 1 | 2.0 GiB |
| TPP frozen-replay causal crossover | pending | 2 | 12 | 96.0 GiB |
| TPP frozen-replay crossover smoke | pending | 1 | 6 | 48.0 GiB |
| TPP frozen-replay endpoint probes | pending | 2 | 10 | 40.0 GiB |

## Current scientific endpoints

- MPrime Phase B: complete at 2,260/2,260 checkpoint-replicates and 60/60
  lineages. Harder validation removed saturation but Stage-2 rank agreement with
  test remains weak.
- Adaptive KL: both arms completed 100 updates. Neither changed coefficient 3,
  so the adaptive treatment never activated.
- Counters visit audit: both arms are complete. The first VH-off divergences
  occur after 881–1,105 actions under tied visit maxima and equal Q values, not
  at the first action. The VH-on behavior arm is not a positive control because
  its policy solved none of the three targets.
- FO Counters validation-led Stage-2 MCTS is complete. The exact VH-off mean is
  6.1/20 at every cutoff versus policy 2.9/20; the final recovery instance used
  its full six-hour allowance and did not add a success.
- MPrime final-validator Stage-1 fixed MCTS is complete at 400/400 classified
  instances. VH-off is 13.0/14.7/15.7 and VH-on is 13.3/15.1/16.0 at
  30m/2h/6h. At six hours neither differs reliably from its own policy; the
  VH-off 30-minute loss remains significant in the provisional six-domain
  extension.
- The Block Grouping PW70 mechanism smoke passed. One trace runs and three are
  resource-pending. The TPP crossover replacement is dependency-gated behind
  repaired smoke 21267359.

## Canonical sources

- Scheduler rows: `experiment_tracking/cluster_workload_latest.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ statistics: `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv`
- Provenance audit: `experiment_tracking/result_csv_provenance_index_latest.csv`

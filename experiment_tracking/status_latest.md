# Current experiment status

Updated: 2026-09-13T19:01:37+03:00

This is the only canonical changing Markdown status page. Dated status files are
historical snapshots. Full RQ tables, methods and conclusions are in
`experiment_tracking/advisor_followup_20260910/README.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
| Block Grouping selected-failure tie-break screen | running | 1 | 4 | 240.0 GiB |
| Counters Stage1 strict tie-break confirmation | running | 1 | 108 | 2160.0 GiB |
| MPrime Phase-B-A Stage1 fixed MCTS | running | 1 | 120 | 2400.0 GiB |
| MPrime Phase-B-A anchor rescore original tasks | running | 1 | 56 | 280.0 GiB |
| MPrime Phase-B-A anchor rescore recovery tasks | running | 1 | 24 | 120.0 GiB |
| MPrime Phase-B-A rescore controller | pending | 1 | 1 | 2.0 GiB |

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
- FO Counters validation-led Stage-2 MCTS: three minimal jobs are running only
  the 42 instances left unclassified by three historical partial allocations.

## Canonical sources

- Scheduler rows: `experiment_tracking/cluster_workload_latest.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ statistics: `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv`
- Provenance audit: `experiment_tracking/result_csv_provenance_index_latest.csv`

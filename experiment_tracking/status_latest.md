# Current experiment status

Updated: 2026-09-10T21:53:42+03:00

This is the only canonical changing Markdown status page. Dated status files are
historical snapshots. Full RQ tables, methods and conclusions are in
`experiment_tracking/advisor_followup_20260910/README.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
| Adaptive KL-control TPP/off screen | running | 1 | 6 | 48.0 GiB |
| Counters root-visit distribution audit | running | 2 | 4 | 240.0 GiB |
| FO Counters validation-led Stage-2 exact recovery | running | 3 | 18 | 360.0 GiB |
| MPrime validation adequacy Phase B full rescore | running | 2 | 8 | 40.0 GiB |

## Current scientific endpoints

- MPrime Phase B: 2,243/2,260 checkpoint-replicates and 58/60 complete
  lineages; repaired exact two-lineage tail `21178598[15,42]` is running.
- Adaptive KL: stable control complete; outlier has 98/100 logged updates and
  has not changed its coefficient from 3.
- Counters visit audit: two jobs are running on the exact three-instance failure
  set and matched VH-on control.
- FO Counters validation-led Stage-2 MCTS: three minimal jobs are running only
  the 42 instances left unclassified by three historical partial allocations.

## Canonical sources

- Scheduler rows: `experiment_tracking/cluster_workload_latest.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ statistics: `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv`
- Provenance audit: `experiment_tracking/result_csv_provenance_index_latest.csv`

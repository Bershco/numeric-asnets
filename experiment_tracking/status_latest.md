# Current experiment status

Updated: 2026-09-15T00:09:00+03:00

This is the only canonical changing Markdown status page. Dated status files are
historical snapshots. Full RQ tables, methods and conclusions are in
`experiment_tracking/advisor_followup_20260910/README.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
| Block Grouping PW70 mechanism trace | 2 running + 2 array-limit pending | 2+2 | 4 | 240.0 GiB |
| Counters Stage1 strict tie-break confirmation | running | 13 | 78 | 1560.0 GiB |
| MPrime Phase-B-A Stage1 PW70 confirmation | running | 2 | 12 | 240.0 GiB |
| TPP frozen-replay endpoint probe | running | 1 | 6 | 20.0 GiB |

The running total is 18 tasks, 100 CPUs and 2,060 GiB. Dependency- and
array-limit-pending rows consume no allocation.

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
- MPrime anchor reranking is complete at 588/588. The frozen coefficients are
  30 for VH-off and 10 for VH-on after manual curve review. Eighteen old
  Stage-2 lineages are definitely non-reusable; two VH-on identities still
  require an exact hash/configuration reuse audit.
- MPrime PW70 has fourteen of sixteen confirmation-extension tasks terminal
  and two live. Conservative ten-seed means are at least 15.7/17.6/17.7
  VH-off and 14.5/16.8/17.4 VH-on at 30m/2h/6h. Final inference waits for
  complete reconciliation.
- The Block Grouping PW70 mechanism smoke passed. Two traces run and two are
  array-limit pending; no terminal mechanism result exists yet.
- TPP crossover smoke 21281126 passed after static path corrections. Both
  one-epoch cross-over training arms completed. The stable-checkpoint x
  bad-replay endpoint is 20/20; the bad-checkpoint x stable-replay endpoint is
  still running. The completed arm already rules out catastrophic replay alone
  as sufficient to damage the stable checkpoint.

## Canonical sources

- Scheduler rows: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260915_0009.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ statistics: `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv`
- Provenance audit: `experiment_tracking/result_csv_provenance_index_latest.csv`

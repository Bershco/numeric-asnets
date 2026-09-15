# Current experiment status

Updated: 2026-09-15T14:39:55+03:00

This is the canonical changing status page. Dated files are historical
snapshots. Primary RQ tables and figures are in
`experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
| MPrime final validation-led Stage 2 | running | 20 | 120 | 960 GiB |
| Counters strict policy-prior confirmation | running | 8 | 48 | 960 GiB |
| Counters exact OOM recovery | running | 3 | 18 | 600 GiB |
| Counters full-root trace controller | dependency-pending | 1 | 0 allocated | 0 allocated |

The running total is 31 tasks, 186 scheduler-allocated CPUs and 2,520 GiB.
The controller is pending on dependencies and consumes no allocation.

## Current scientific endpoints

- MPrime Phase B and Phase C are complete. Phase-B replicate A is the adopted
  validator. Anchor rescoring is complete at 588/588; the frozen coefficients
  are 30 for VH-off and 10 for VH-on.
- MPrime Stage-1 fixed MCTS and PW70 are complete and already appear in RQ2 and
  RQ4. PW70 is 15.7/17.6/17.7 VH-off and 15.5/17.9/18.5 VH-on at
  30m/2h/6h.
- All twenty final MPrime Stage-2 lineages are running from the adopted
  Stage-1 checkpoints. At this snapshot every lineage has saved checkpoints;
  the range is epoch 22-60/100. Policy curves/endpoints and matched fixed/PW
  evaluations remain downstream.
- The targeted Counters causal pilot remains action-ID 0/3, Q 0/3 and
  policy-prior 3/3 for three VH-off policy-success instances. The matched VH-on
  arm was not a positive control because its policy solved 0/3.
- The strict Counters confirmation has nine original tasks complete, eight
  running and three OOM sources. Exact recovery array `21308619[4,6,14]` runs
  only the 49 unclassified identities with 200 GiB each. Controller `21319149`
  will then trace only the exact policy-success/action-ID-failure union with
  full root vectors under both action-ID and policy-prior.
- Seed 1510771779 is fully joined: policy solved 10, action-ID 18 and
  policy-prior 17. Both MCTS rules retained all ten policy successes; the sole
  action-ID-only success (`fz_instance_19.pddl`) was a policy failure, so this
  is not a policy-preservation regression under policy-prior.
- The four Block Grouping PW70 mechanism traces all timed out. Three first
  divergences had unique visit winners and the fourth maximum tie excluded the
  policy action. The cached Q/U decomposition shows the representative unique
  winners were Q/exploitation-dominated while U favored the policy child; this
  does not support a broad exploration-coefficient sweep.
- The TPP crossover is complete: bad checkpoint x bad replay 10/20, bad x
  stable 3/20, stable x bad 20/20, stable x stable 20/20. The obsolete
  coefficient-doubling treatment failed operationally. Corrected compute smoke
  `21318360` and both fixed-anchor/LR-backtracking treatments `21318362[0-1]`
  completed. Endpoint jobs `21318364[0-1]` also completed at 20/20 for both
  arms. The selected catastrophic seed therefore improved from its historical
  first-Stage-2 10/20 to 20/20, while the stable control remained 20/20.
  This is a positive selected-pair prevention screen, not a population estimate;
  retry steps resampled dropout and therefore do not isolate exact learning-rate
  scaling as the sole causal mechanism.

## Canonical sources

- Scheduler rows: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260915_1439.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- PUCT feasibility audit: `experiment_tracking/mcts_puct_balance_feasibility_20260915.md`
- Provenance index: `experiment_tracking/result_csv_provenance_index_latest.csv`

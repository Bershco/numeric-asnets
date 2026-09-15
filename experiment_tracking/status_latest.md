# Current experiment status

Updated: 2026-09-15T18:57:00+03:00

This is the canonical changing status page. Dated files are historical
snapshots. Primary RQ tables and figures are in
`experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
| MPrime final validation-led Stage 2 | 19 running, 1 terminal | 19 allocated | 114 | 912 GiB |
| MPrime live policy curves | 22 running, 24 terminal | 22 allocated | 220 | 440 GiB |
| Counters strict policy-prior confirmation | 5 original tasks running | 5 | 30 | 600 GiB |
| Counters exact unclassified-instance recovery | 2 running tasks covering 3 identities | 2 | 4 | 240 GiB |
| Counters full-root trace controller | dependency-pending | 1 | 0 allocated | 0 allocated |

The running total is 49 allocated tasks, 370 scheduler-allocated CPUs and
2,194 GiB. The trace controller is pending on dependencies and consumes no
allocation.

## Current scientific endpoints

- MPrime Phase B and Phase C are complete. Phase-B replicate A is the adopted
  validator. Anchor rescoring is complete at 588/588; the frozen coefficients
  are 30 for VH-off and 10 for VH-on.
- MPrime Stage-1 fixed MCTS and PW70 are complete and already appear in RQ2 and
  RQ4. PW70 is 15.7/17.6/17.7 VH-off and 15.5/17.9/18.5 VH-on at
  30m/2h/6h.
- All twenty final MPrime Stage-2 lineages started from the adopted Stage-1
  checkpoints. At this snapshot one is terminal at epoch 99 and nineteen are
  running, with latest immutable saved epochs spanning 42-98. Their original
  72-hour allocations have approximately 63.5-64.4 hours of scheduler time
  remaining, but observed rates indicate roughly less than one to twelve hours
  of scientific work. The walltime was deliberately conservative and is not a
  completion estimate.
- Live policy-curve controller `21345695` overlaps policy evaluation with
  training. It found 300 immutable checkpoints and had submitted 46 unique
  jobs: 24 terminal plus 22 active at the snapshot. Live checkpoints
  are labelled learning-curve-only; selected/final roles and all downstream
  fixed/PW MCTS remain gated on terminal lineage evidence.
- The targeted Counters causal pilot remains action-ID 0/3, Q 0/3 and
  policy-prior 3/3 for three VH-off policy-success instances. The matched VH-on
  arm was not a positive control because its policy solved 0/3.
- The strict Counters confirmation has five original tasks still running. A
  ledger audit proved that the old recovery array was repeating explicit
  six-hour timeouts because those outcomes were present in stdout but absent
  from JSONL; it was cancelled without deleting evidence. Reconciliation left
  only three genuinely unclassified identities. Exact replacement
  `21347260[7,19]` is running those identities only: one in task 7 and two in
  task 19, one worker per task, six hours per instance and a 14-hour allocation
  for the two-instance arm. Corrected controller `21347597` waits for the
  original array and exact recovery, then traces only the policy-success and
  same-build action-ID-failure union with complete root vectors.
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
  This is a positive selected-pair prevention screen, not a population estimate.
  The corrected treatment changed both the KL measurement (deterministic current
  policy, excluding dropout disagreement) and the optimizer behavior (full
  weights-and-Adam rollback plus learning-rate backtracking). Retry steps still
  resampled dropout, so the result does not isolate backtracking or exact
  learning-rate scaling as the sole causal mechanism.

## Canonical sources

- Scheduler rows: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260915_1857.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- PUCT feasibility audit: `experiment_tracking/mcts_puct_balance_feasibility_20260915.md`
- Provenance index: `experiment_tracking/result_csv_provenance_index_latest.csv`

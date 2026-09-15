# Current experiment status

Updated: 2026-09-16T01:16:00+03:00

This is the canonical changing status page. Dated manifests and ledgers are
immutable snapshots. Primary validation-led RQ tables and figures are in
`experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`.
Terminal-led campaigns remain provenance-only and are excluded from the
primary RQ report.

## Live workload

| Experiment | State | Allocated jobs | CPU | RAM | Expected / hard remainder |
|---|---|---:|---:|---:|---|
| MPrime final validation-led Stage 2 | 17/20 terminal; 3 running | 3 | 18 | 144 GiB | observed completion likely within hours / about 58 h scheduler bound |
| MPrime policy curves | 225 completed attempts, 30 running, 38 failed attempts | 30 | 300 | 600 GiB | each job <=4 h; failure-aware retries active |
| MPrime policy controllers | primary and failure-aware retry controllers running | 2 | 4 | 4 GiB | exit after complete submission/retry reconciliation |
| Counters strict tie-break confirmation | 16/20 terminal; 4 running | 4 | 24 | 480 GiB | <=12.6 h hard |
| Counters exact identity recovery | tasks 7 and 19 running; task 9 repaired and running | 3 | 6 | 360 GiB | 0-14 h for tasks 7/19; <=20 h task 9 |
| Counters full-root trace controller | dependency-pending, job 21362904 | 0 | 0 | 0 GiB | starts only after exact classification |

The allocated snapshot is 42 jobs, 352 CPUs and 1,588 GiB. The Counters trace
controller is pending on dependencies and consumes no allocation. MPrime
policy retry jobs and primary jobs share the same scientific identity ledger;
attempt counts must not be mistaken for unique checkpoints.

## Current scientific endpoints

- MPrime Phase B and Phase C are complete. Phase-B replicate A is the adopted
  validator. Anchor rescoring is complete at 588/588; the frozen coefficients
  are 30 for VH-off and 10 for VH-on.
- MPrime Stage-1 fixed MCTS and PW70 are complete and already appear in RQ2 and
  RQ4. PW70 is 15.7/17.6/17.7 VH-off and 15.5/17.9/18.5 VH-on at
  30m/2h/6h.
- Final MPrime Stage-2 training has 17 complete lineages and three live
  lineages (jobs 21303720, 21303721 and 21303731). Observed lineage rates imply
  completion within hours, while the conservative scheduler allocations retain
  about 58 hours.
- MPrime live policy materialization remains active. The 01:12 local archive
  holds 413 data rows, 259 primary submissions and 34 retry attempts. The primary
  controller continues discovering and submitting immutable checkpoints. A
  separate failure-aware controller retries only failed identities. The current
  aggregate is 225 completed, 30 running and 38 failed attempts; attempts are
  not unique scientific identities. At least eighteen audited failures occurred
  before inference with POSIX semaphore ENOSPC on `ise-cpu128-03`. Other failed
  attempts include native pre-inference failures on additional nodes, so the
  full failure count is not attributed to that one node. Both current submission
  paths exclude the specifically demonstrated bad node and retry failed
  identities without duplicating successful evidence.
- Policy-curve rows remain learning-curve-only until their lineage terminates.
  Endpoint selection and the approved 20 fixed-MCTS plus 20 PW70 Stage-2 jobs
  remain gated on terminal checkpoint selection.
- The targeted Counters pilot remains valid: on three VH-off instances that
  the same policies solve, action-ID and Q tie-breaking solved 0/3 while
  policy-prior tie-breaking solved 3/3. The unrelated VH-on behavior arm is not
  used as evidence because its policies did not solve those instances.
- A full strict Counters audit found explicit 21,600-second timeouts printed in
  stdout but absent from JSONL. Reconciliation now treats those as terminal and
  avoids repeating them. Tasks 7 and 19 contain three genuinely interrupted
  identities. A later OOM in task 9 exposed three more genuinely unclassified
  identities (45, 48 and 52), so an exact task-9 recovery was added. At the
  01:10 snapshot none of the six genuine recovery identities had yet produced a
  new terminal record; tasks 7 and 19 were near their first six-hour scientific
  bound and task 9 had just started. Its first
  setup attempts are retained as failed provenance; corrected job `21362903`
  runs only those three identities and excludes `ise-cpu-intl-25`, where the
  native evaluator reproduced exit -4 twice.
- `21362904` is the only live Counters full-root controller. It waits for the
  original strict array plus both exact recovery groups, requires 59 terminal
  identities for every action-ID seed, and then traces only the final
  policy-success/action-ID-failure union under both rules. The standardized
  visit-margin/prominence branch remains held until that exact-tie evidence is
  complete.
- The TPP frozen-replay crossover is complete: bad checkpoint x bad replay
  10/20, bad x stable replay 3/20, stable x bad replay 20/20, and stable x
  stable replay 20/20. A selected-pair one-epoch rollback/LR-backtracking guard
  then restored the catastrophic seed to 20/20 while preserving the stable
  control at 20/20. This proves preventability for the selected pair, not a
  population-level replacement for the primary 9/20 Stage-2 result. Exact-RNG
  causal closure is locally implemented, tested and independently reviewed,
  but cluster deployment awaits explicit authorization to push reviewed commit
  `fccaa89e` to the GitHub branch used by the isolated checkout. No 100-epoch
  guard campaign has been adopted.

## Canonical sources

- Scheduler snapshot: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260916_0116.csv`
- MPrime ready rows: `experiment_tracking/mprime_final_stage2_validation_20260915/live_policy_ready_20260916_0112.csv`
- MPrime primary ledger: `experiment_tracking/mprime_final_stage2_validation_20260915/live_policy_submissions_20260916_0112.tsv`
- MPrime retry ledger: `experiment_tracking/mprime_final_stage2_validation_20260915/live_policy_retry_submissions_20260916_0112.tsv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- Provenance index: `experiment_tracking/result_csv_provenance_index_latest.csv`

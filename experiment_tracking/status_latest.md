# Current experiment status

Updated: 2026-09-16T01:37:30+03:00

This is the canonical changing status page. Dated manifests and ledgers are
immutable snapshots. Primary validation-led RQ tables and figures are in
`experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`.
Terminal-led campaigns remain provenance-only and are excluded from the
primary RQ report.

## Live workload

| Experiment | State | Allocated jobs | CPU | RAM | Expected / hard remainder |
|---|---|---:|---:|---:|---|
| MPrime final validation-led Stage 2 | 19/20 terminal; 1 running at snapshot 78/99 | 1 | 6 | 48 GiB | about 4 h at its observed rate / 57 h 36 m scheduler bound |
| MPrime policy curves | 237 completed attempts, 30 running, 42 failed attempts | 30 | 300 | 600 GiB | each job <=4 h; failure-aware retries active |
| MPrime policy controllers | primary and failure-aware retry controllers running | 2 | 4 | 4 GiB | exit after complete submission/retry reconciliation |
| Counters strict tie-break confirmation | 16/20 terminal; 4 running | 4 | 24 | 480 GiB | <=12.6 h hard |
| Counters exact identity recovery | task 7 terminal; tasks 19 and 9 running | 2 | 4 | 240 GiB | <=7 h 7 m task 19; <=19 h 6 m task 9 |
| Counters full-root trace controller | dependency-pending, job 21362904 | 0 | 0 | 0 GiB | starts only after exact classification |

The allocated snapshot is 39 jobs, 338 CPUs and 1,372 GiB. The Counters trace
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
- Final MPrime Stage-2 training has 19 complete lineages and one live lineage,
  job 21303720. It had saved snapshot 78/99 at the refresh. Its observed rate
  implies roughly four hours to finish; the deliberately conservative scheduler
  allocation still retains about 57 hours 36 minutes.
- MPrime live policy materialization remains active. The 01:34 local archive
  holds 414 data rows, 269 primary submissions and 40 retry attempts. The primary
  controller continues discovering and submitting immutable checkpoints. A
  separate failure-aware controller retries only failed identities. The current
  aggregate is 237 completed, 30 running and 42 failed attempts; attempts are
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
  01:37 snapshot task 7's sole identity had reached and printed the exact
  21,600-second timeout. Its ledger remains at 58/59 only because reconciliation
  occurs in the dependent controller after every recovery is terminal; it will
  then become a durable timeout without rerunning the instance. Task 19 printed
  the same exact timeout for its first identity and is evaluating its second.
  Task 9 is evaluating the first of three exact identities. Its first
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
  stable replay 20/20. A fresh selected-pair one-epoch rollback/LR-backtracking
  rerun then yielded 20/20 for the bad seed and 20/20 for the stable control,
  apparently preventing the analogous collapse. Because rejected retries
  resampled dropout, this is encouraging selected-pair prevention evidence—not
  yet causal isolation of the guard or a population-level replacement for the
  primary 9/20 Stage-2 result. Exact-RNG
  causal closure is locally implemented, tested and independently reviewed,
  but cluster deployment awaits explicit authorization to push reviewed commit
  `fccaa89e` to the GitHub branch used by the isolated checkout. No 100-epoch
  guard campaign has been adopted.

## Canonical sources

- Scheduler snapshot: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260916_0137.csv`
- MPrime ready rows: `experiment_tracking/mprime_final_stage2_validation_20260915/live_policy_ready_20260916_0134.csv`
- MPrime primary ledger: `experiment_tracking/mprime_final_stage2_validation_20260915/live_policy_submissions_20260916_0134.tsv`
- MPrime retry ledger: `experiment_tracking/mprime_final_stage2_validation_20260915/live_policy_retry_submissions_20260916_0134.tsv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- Provenance index: `experiment_tracking/result_csv_provenance_index_latest.csv`

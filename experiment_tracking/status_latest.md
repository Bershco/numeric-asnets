# Current experiment status

Updated: 2026-09-16T17:44:17+03:00 (live scheduler and evidence refresh)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State | Running / pending | CPU / RAM | Expected / hard bound |
|---|---|---:|---:|---|
| MPrime Phase-B-A checkpoint evaluations | 18 complete; 227 original tasks plus exact task-100 retry `21412365_100` running | 228 / 0 | 684 requested CPU, 912 allocated / 4,560 GiB | first completions took about 16–22 m; 8 h per-task hard limit |
| MPrime finalizer / downstream | `21412366` waits for original array termination and successful retry; `21412367` waits for finalizer | 0 / 2 | 1 CPU / 2 GiB each | 30 m each hard limit |
| Counters strict policy-prior confirmation | original task `21233925_1` running | 1 / 0 | 6 CPU / 120 GiB | 39 h 15 m hard remainder; scientific completion also needs reconciliation |
| Counters task-9 exact recovery | `21362903_9` running | 1 / 0 | 2 CPU / 120 GiB | 2 h 59 m hard remainder |
| Counters full-root trace controller | `21362904` dependency-pending | 0 / 1 | 1 CPU / 2 GiB when active | starts only after all action-ID ledgers reconcile to 59/59 |

## Scientific status

- **MPrime acceleration:** the old lineage-level arrays `21390404` and
  `21392547`, old finalizer/downstream `21392548`/`21392549`, and obsolete
  policy watcher `21362500` were cancelled without deleting their evidence.
  Exactly 174/420 identity-valid summary+receipt pairs were preserved. The 246
  missing lineage/epoch identities were frozen into a checksum-pinned manifest
  and submitted one per task. The 246-task cap requests 4,920 GiB, leaving
  about 964 GiB under the 6-TiB envelope at the submission snapshot.
- The new MPrime finalizer refuses anything other than 21 exact epoch results
  per lineage, exactly 30 binary VAL rows per result, and a matching completion
  identity containing checkpoint, validator and evaluator hashes. It depends
  only on the replacement fine array; the downstream controller depends only
  on that finalizer. The downstream path reuses exact existing policy/search
  evidence. If a newly selected endpoint lacks policy evidence, it submits only
  that exact endpoint once into a date-frozen recovery root and then retries;
  it fails closed if evidence remains absent, before any search release.
- Original array task `21411413_100` suffered an early native code -4 exit on
  `ise-cpu-intl-01`. Exact retry `21412365_100` reruns only that frozen identity
  on `ise-cpu128-06`; no completed checkpoint is repeated. A transient
  user-environment retrieval hold was released without changing its manifest.
  Replacement
  finalizer `21412366` waits for every original task to terminate and the retry
  to succeed. Old pending finalizer/controller `21411414`/`21411415` were
  cancelled before execution and replaced by `21412366`/`21412367`.
- **Counters recovery:** “three identities” means evaluator IDs 45, 48 and 52
  for the action-ID arm of seed 2082152039, not three seeds or three full jobs.
  They were the only task-9 cases without durable terminal classification after
  stdout/ledger reconciliation. The job reruns only those cases with the
  original six-hour per-instance budget. The separate policy-prior task-19
  recovery already ran and awaits final ledger reconciliation.
- The Counters targeted causal pilot remains: on three VH-off policy-success
  instances, action-ID/Q solved 0/3 and policy-prior solved 3/3. The ten-seed
  strict campaign remains incomplete, so primary Counters RQ values are frozen.
- **TPP Phase D is complete:** under the same frozen replay and RNG-314159
  schedule, legacy dropout-current KL scores 11/20 on catastrophic seed
  1972442430 while deterministic-current KL scores 20/20; stable seed
  2082152039 scores 20/20 under both semantics. This is selected-pair causal
  evidence that the legacy KL semantics create the loss under this stream, not
  a population estimate and not a replacement for the historical 9/20 primary
  endpoint.
- **Historical anchor-gradient availability:** all 100 validation-led Stage-2
  logs for Block Grouping, Drone, FO Counters, Rover and Counters exist, but
  zero contains `policy_gradient_l2`, `weighted_anchor_gradient_l2`, or
  `policy_anchor_gradient_cosine`. This means severity is not retrospectively
  measurable—not that zero lineages were affected. Aggregate KL/loss/total
  gradient is not a valid substitute, so no broad retraining decision follows.

## Canonical sources

- Live scheduler snapshot: `experiment_tracking/live_experiment_status_latest.csv`
- MPrime rescore: `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/README.md`
- Anchor-gradient audit: `experiment_tracking/advisor_followup_20260910/stage2_anchor_gradient_availability_20260916.csv`
- Primary RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- Experiment registry: `experiment_tracking/experiment_registry.csv`

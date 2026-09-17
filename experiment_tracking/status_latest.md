# Current experiment status

Updated: 2026-09-17T23:26:51+03:00.

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State and scientific progress | Allocated resources |
|---|---|---:|
| MPrime final Stage-2 fixed/PW70 | 22 scheduler-complete, 10 OOM and 8 running. Success-ledger lower bounds are fixed/off >=14.2/15.8/16.6, fixed/on >=12.5/14.2/15.4, PW70/off >=16.0/17.7/17.9 and PW70/on >=15.2/17.0/17.2 at 30m/2h/6h. | 48 CPU / 960 GiB |
| PRIMARY-50 KL crossover repair | 38/48 legacy captures complete; ten exact capture repairs running. Fourteen deterministic treatments are retained and 34 exact replacements are dependency-pending, followed by verifier, 96 endpoints and finalizer. | 60 CPU / 1,200 GiB |
| Counters strict tie-break | Final Action-ID arm has 57/59 exact terminal identities: 28 successes and 29 six-hour timeouts. Instances 44 and 59 remain unresolved. | 6 CPU / 120 GiB |

**Allocated total:** 114 CPU / 2,280 GiB. Dependency-pending work allocates
nothing until released.

## Scientific interpretation

- MPrime policy endpoints remain complete at 16.5/20 VH-off and 16.7/20
  VH-on. The live search lower bounds are encouraging for PW70, but timeout,
  action-limit and OOM reconciliation is still required before paired CIs,
  p-values or an RQ2/RQ4 update.
- The old six-checkpoint KL mechanism screen is complete and explicitly not a
  population estimate. The current PRIMARY-50 experiment uses ten distinct
  primary Stage-1 VH-off checkpoints per domain. Only two Counters pairs are
  reused because only their exact checkpoint hashes match the primary manifest.
- The Counters primary RQ values remain frozen until the last Action-ID arm and
  exact full-root trace follow-up complete.
- The local MCTS-divergence Stage-0 join and the value-head V1 extractor/schema
  preparation are complete. Neither is yet a new scientific result.

## RQ impact

No primary RQ value changes at this snapshot. MPrime fixed/PW70 values remain
explicit lower bounds and the KL experiments are method diagnostics rather
than replacements for the historical primary pipeline.

## Canonical sources

- `experiment_tracking/cluster_workload_latest.csv`
- `experiment_tracking/live_experiment_status_latest.csv`
- `experiment_tracking/mprime_final_stage2_search_20260916/`
- `experiment_tracking/imperfect_kl_primary_lineage_crossover_20260917/`
- `experiment_tracking/imperfect_kl_frozen_replay_crossover_20260917/`
- `experiment_tracking/mcts_policy_divergence_cause_audit/`
- `experiment_tracking/value_head_quality_audit/`

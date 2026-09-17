# Current experiment status

Updated: 2026-09-17T10:53:00+03:00 (live scheduler/accounting snapshot)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State | Running / pending / complete | Allocated CPU / RAM | Expected / hard bound |
|---|---|---:|---:|---|
| MPrime selected policy endpoints | seven exact VH-on endpoints running; twelve selected endpoints reused and one recovery complete | 7 / 0 / 13 | 70 CPU / 140 GiB | policy jobs have 4 h hard limits; downstream normally releases after all eight classify |
| MPrime fixed/PW handoff | controller `21428612` dependency-pending | 0 / 1 / 0 | 0 allocated now; controller requests 1 CPU / 2 GiB when released | validates policy evidence, then submits 20 fixed 20/70 and 20 PW70 identities |
| TPP KL multi-RNG | final legacy catastrophic-seed endpoint running; fifteen endpoints complete | 1 / 0 / 15 | 6 CPU / 20 GiB | 2 h hard limit; matched deterministic endpoint completed 20/20 in 23m06 |
| Counters strict tie-break (unchanged) | one original task still running | 1 / 1 controller / 19 scheduler-terminal tasks | 6 CPU / 120 GiB | 72 h task hard limit; no new scientific result reported here |

**Total currently allocated:** 82 CPU / 280 GiB. Dependency-pending controllers
allocate no resources. Excluding the unchanged Counters task, newly relevant
active work uses 76 CPU / 160 GiB.

## Newly completed work

- **MPrime Phase-B-A rescore:** 420/420 checkpoint evaluations complete.
  Corrected finalizer `21428590` froze twenty endpoints. Controller `21428591`
  found twelve exact matching policy results and submitted only eight missing
  VH-on endpoints (`21428604`--`21428611`). Job `21428610` is already complete
  at 20/20; seven remain live. Fixed and PW are not running yet;
  controller `21428612` is correctly waiting on those eight.
- **Imperfect-domain KL screen:** 20/20 one-epoch training jobs and 20/20
  endpoint jobs complete. Deterministic-current KL removes the artificial
  step-zero anchor gradient throughout the screen. Endpoint comparisons are
  descriptive, not causal, because prospective treatment arms did not share
  an identical frozen replay schedule. A frozen-replay crossover is the
  required next diagnostic before any 100-epoch retraining.
- **Historical Counters epoch-0 comparator:** complete at 4/59.

## RQ impact

No primary RQ row changes at this snapshot. MPrime can enter RQ1/RQ3 only once
the eight selected policy endpoints finish, and RQ2/RQ4 only after the matched
fixed 20/70 and PW70 campaigns finish. TPP and the imperfect-domain KL screen
are implementation/mechanism diagnostics rather than replacements for the
historical primary rows.

## Canonical sources

- Live scheduler snapshot: `experiment_tracking/live_experiment_status_latest.csv`
- MPrime rescore: `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/README.md`
- TPP multi-RNG: `experiment_tracking/tpp_kl_multirng_susceptibility_20260916/`
- Imperfect-domain KL screen: `experiment_tracking/imperfect_domains_kl_semantics_screen_20260916/`
- Methods inventory: `docs/thesis_methods_configuration_inventory_20260916.md`
- Divergence audit: `experiment_tracking/mcts_policy_divergence_cause_audit/README.md`
- Value-head audit: `experiment_tracking/value_head_quality_audit/README.md`

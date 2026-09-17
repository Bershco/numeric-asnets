# Current experiment status

Updated: 2026-09-17T15:33:00+03:00 (live scheduler and scientific-ledger audit)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State and scientific progress | Allocated / requested resources | Expected completion |
|---|---|---:|---|
| MPrime final Stage-2 fixed/PW | 38 tasks running and two scheduler-complete; 422/800 instances durably classified | 228 CPU / 4,560 GiB | late six-hour failures dominate the remaining work; no final paired estimate yet |
| Imperfect-domain frozen-replay KL crossover | training and 10-pair verifier complete; 19/20 test endpoints complete; Block Grouping seed-42 legacy endpoint held before inference after environment retrieval failure | 0 allocated; held endpoint requests 5 CPU / 20 GiB | requires explicit release or exact retry before the tenth pair closes |
| Counters strict tie-break | requeued component `21234831`, Stage-1 VH-off seed 923500475, Action-ID narrow 5/20; 28 durable ledger records and 30 raw timeout text matches | 6 CPU / 120 GiB | still genuinely active; controller will reconcile unique terminal identities when the final arm exits |

**Allocated total:** 234 CPU / 4,680 GiB. Pending/held work requests no
resources until its gate opens.

## Newly completed results

- **MPrime policy endpoints:** all twenty Phase-B-A-selected Stage-2 policy
  endpoints are complete. Twelve exact-hash results were reused and only eight
  missing VH-on endpoints were run; those eight took 7–41 minutes. The means
  are VH-off 16.5/20 and VH-on 16.7/20.
- **MPrime live search lower bounds:** fixed off >=7.1/7.8/8.3, fixed on
  >=6.0/6.4/6.7, PW70 off >=13.0/14.6/14.8 and PW70 on >=10.9/12.2/12.4 at
  30m/2h/6h. These are 422 partial classifications, not final paired estimates.
  The source Stage-2 networks use legacy dropout-current KL semantics.
- **TPP KL multi-RNG:** complete. The catastrophic seed scores 11, 10, 20 and
  19/20 under legacy dropout-current KL, versus 20/20 in all four
  deterministic-current arms. The stable control is 20/20 under both semantics
  throughout. The seed is stochastically susceptible to the legacy KL
  implementation; deterministic-current KL protects it across the tested RNGs.
- **Frozen-replay crossover:** all twenty treatments and the integrity verifier
  completed. The verifier certified identical ordered replay hashes, step-0
  policy gradients and target bits for all ten pairs. Nineteen policy endpoints
  completed; one Block Grouping legacy endpoint is held before inference. The
  nine complete pairs show heterogeneous changes: one BG +2; Drone 0/-2; FO
  Counters -3/-3; Rover 0/0; Counters +29/-21. Deterministic-current removes
  the artificial step-0 anchor gradient but is not uniformly better in coverage.

## RQ impact

MPrime now adds completed policy-only evidence:

- RQ1/VH-off: 16.3 -> 16.5, +0.2 plans, 95% CI [-1.90, 2.30], raw exact
  p=.9297, Holm p=1.0. No reliable Stage-2 policy improvement.
- RQ3/VH-on direct: 15.7 -> 16.7, +1.0 [-.58, 2.58], raw p=.2539,
  Holm p=1.0.
- RQ3 interaction against VH-off: +0.8 [-1.63, 3.23], raw p=.5137,
  Holm p=1.0. The positive mean does not establish a value-head training
  benefit.

RQ2/RQ4 do not change yet. The MPrime fixed/PW completion records are live
lower bounds and must not receive CIs or significance tests before all matched
cells finish.

## Canonical sources

- `experiment_tracking/cluster_workload_latest.csv`
- `experiment_tracking/live_experiment_status_latest.csv`
- `experiment_tracking/mprime_final_stage2_search_20260916/`
- `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/mprime_policy_rq_extension_20260917.csv`
- `experiment_tracking/tpp_kl_multirng_susceptibility_20260916/`
- `experiment_tracking/imperfect_kl_frozen_replay_crossover_20260917/`
- `docs/thesis_methods_configuration_inventory_20260916.md`

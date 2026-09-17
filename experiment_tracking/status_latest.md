# Current experiment status

Updated: 2026-09-17T13:17:14+03:00 (live scheduler; scientific ledger audited at 13:08)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State and scientific progress | Allocated / requested resources | Expected completion |
|---|---|---:|---|
| MPrime final Stage-2 fixed/PW | 39 tasks running and one scheduler-complete; 332/800 instances durably classified at the 13:08 ledger audit | 234 CPU / 4,680 GiB | the recent empirical rate suggests roughly 6 h if it stayed constant, but late six-hour failures make 8–24 h more defensible |
| Imperfect-domain frozen-replay KL crossover | 17/20 one-epoch treatments complete; three Rover treatments running; verifier and endpoints dependency-pending | 18 CPU / 360 GiB allocated now | completed tasks took 3 min–1h20; the last three should normally finish shortly, then the integrity verifier and twenty short endpoints run |
| Counters strict tie-break | requeued component `21234831`, Stage-1 VH-off seed 923500475, Action-ID narrow 5/20; 28 durable ledger records and 27 printed timeout markers | 6 CPU / 120 GiB | still genuinely active; controller will reconcile printed terminal outcomes when the final arm exits |

**Allocated total:** 258 CPU / 5,160 GiB. Dependency-pending work requests no
resources until its gate opens.

## Newly completed results

- **MPrime policy endpoints:** all twenty Phase-B-A-selected Stage-2 policy
  endpoints are complete. Twelve exact-hash results were reused and only eight
  missing VH-on endpoints were run; those eight took 7–41 minutes. The means
  are VH-off 16.5/20 and VH-on 16.7/20.
- **MPrime live search lower bounds:** fixed off >=5.8/6.4/6.4, fixed on
  >=5.3/5.6/5.6, PW70 off >=10.8/12.1/12.1 and PW70 on >=8.1/9.1/9.1 at
  30m/2h/6h. These are 332 partial classifications, not final paired estimates.
  The source Stage-2 networks use legacy dropout-current KL semantics.
- **TPP KL multi-RNG:** complete. The catastrophic seed scores 11, 10, 20 and
  19/20 under legacy dropout-current KL, versus 20/20 in all four
  deterministic-current arms. The stable control is 20/20 under both semantics
  throughout. The seed is stochastically susceptible to the legacy KL
  implementation; deterministic-current KL protects it across the tested RNGs.
- **Frozen-replay crossover active:** `21430551` training -> `21430552`
  integrity verifier -> `21430553` endpoints. Seventeen treatments are complete;
  three Rover treatments are active. Ten source checkpoints and all 600 replay
  files were checksum-frozen before submission.

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

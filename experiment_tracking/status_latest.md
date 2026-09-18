# Current experiment status

Updated: 2026-09-18T10:52:00+03:00.

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State and scientific progress | Allocated resources |
|---|---|---:|
| MPrime final Stage-2 fixed/PW70 exact recovery | All 40 original tasks are scheduler-terminal: 25 completed normally and 15 OOM. Scientific reconciliation found 784/800 terminal identities: 677 successes, 107 exact six-hour timeouts and 16 genuinely unclassified fixed-search instances. The first two operational recovery attempts failed safely before inference because they used the pre-fix checkout. Corrected array `21449642_[0-15]` is now verified on commit `e5b44c25` and writes separate recovery ledgers; `21449643` performs the final reconciliation. Current lower bounds are fixed/off >=14.2/15.8/16.6, fixed/on >=12.9/14.6/16.0, PW70/off 16.0/17.7/17.9 and PW70/on 15.2/17.0/17.2 at 30m/2h/6h. | 32 CPU / 1,920 GiB, plus one dependency-pending finalizer |
| Counters strict tie-break exact tail | The last Action-ID arm reconciles to 58/59 terminal identities: 28 successes and 30 exact six-hour timeouts. Evaluator slot 59 is a terminal timeout; slot 44 alone remains unclassified. Exact recovery `21449257_[1]` is running, followed by full-root trace controller `21449258`. A newer exact duplicate, `21449584_[1]` plus controller `21449585`, targets the same identity and completion ledger; the newer pair should be cancelled after exact-ID authorization to avoid a shared-ledger collision. | 4 CPU / 400 GiB across the original and accidental duplicate; controllers allocate zero while pending |
| MCTS first-divergence audit | The first smoke correctly rejected a stale step-1 fixture and automatically cancelled its dependents. Comparison against historical log `21178321` proved that step 0 is the actual first divergence. Corrected smoke `21449781` passed on commit `24a7c280`: recorder-disabled/enabled traces match and reproduce step 0, policy 52, search 101, root visits 20 and edge visits 19. Ten fixed grouped tasks `21449782` and two optional FO-Counters PW70 tasks `21449783` are now running. | 24 CPU / 1,440 GiB; 26-hour task hard limits |

At the snapshot, active work requested 60 CPU / 3,760 GiB, including the
accidental duplicate Counters recovery. Dependency-pending controllers allocate
nothing until released.

## Scientific interpretation

- MPrime policy endpoints remain complete at 16.5/20 VH-off and 16.7/20
  VH-on. PW70 is already scientifically complete and its descriptive means
  exceed policy in both modes by two and six hours. Fixed search has eight
  missing instances per VH mode, so paired CIs, p-values and the RQ2/RQ4
  update remain frozen until exact recovery finishes.
- PRIMARY-50 is complete for all 50 independent primary checkpoints. It is a
  conditional update-effect diagnostic because deterministic-current KL
  replays each checkpoint's own legacy-generated data. The only
  Holm-significant source-to-treatment contrast is FO Counters under legacy
  KL (4.2 to 2.1; Holm p=.00977); deterministic-current attenuates the loss to
  2.7 but no deterministic-minus-legacy contrast is Holm-significant.
- The Counters primary RQ values remain frozen until instance 44 and the exact
  full-root trace follow-up complete.
- The compact MCTS first-divergence recorder passed its known-root compute
  smoke. Ten grouped fixed-search tasks and two optional FO-PW tasks are live;
  they cover 40 frozen candidates without submitting 40 separate jobs.
  The value-head V1 design now uses identical shared state manifests for each
  matched Stage-1/Stage-2 pair, but checkpoint hashes and eight state manifests
  remain deliberately preflight-blocking. Neither is yet a scientific result.

## RQ impact

No primary RQ value changes at this snapshot. MPrime fixed-search values remain
explicit lower bounds, and the KL result is a method diagnostic rather than a
replacement for the historical primary pipeline.

## Canonical sources

- `experiment_tracking/cluster_workload_latest.csv`
- `experiment_tracking/live_experiment_status_latest.csv`
- `experiment_tracking/mprime_final_stage2_search_20260916/`
- `experiment_tracking/imperfect_kl_primary_lineage_crossover_20260917/`
- `experiment_tracking/imperfect_kl_frozen_replay_crossover_20260917/`
- `experiment_tracking/mcts_policy_divergence_cause_audit/`
- `experiment_tracking/value_head_quality_audit/`

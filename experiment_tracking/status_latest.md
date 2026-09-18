# Current experiment status

Updated: 2026-09-18T19:39:00+03:00.

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State and scientific progress | Allocated resources |
|---|---|---:|
| MPrime final Stage-2 fixed/PW70 | **Complete: 800/800 scientific identities**. Final reconciliation contains 677 successes and 123 exact six-hour timeouts, with no action-limit failures, conflicts or unclassified cases. Exact 30m/2h/6h means are fixed/off 14.2/15.8/16.6, fixed/on 12.9/14.6/16.0, PW70/off 16.0/17.7/17.9 and PW70/on 15.2/17.0/17.2. The sixteen last wrappers have scheduler state `FAILED` only because the historical timeout path did not append JSONL after producing valid exact timeout evidence. | Terminal; zero live resources |
| Counters strict tie-break exact tail | Canonical recovery `21453481_[0]` is running the last two exact identities; controller `21453482` will reconcile them and release the full-root trace union. Accidental slower duplicates `21453491`/`21453492` were cancelled and are never counted. | 4 CPU / 200 GiB; two parallel six-hour instance allowances |
| Endogenous 100-epoch KL pilot | Eight same-build Drone/FO-Counters arms `21453460_[0-7]` are running. Obsolete jobs `21453461`/`21453462` were cancelled. Canonical watcher `21455207` discovered 60 stable checkpoints and submitted exact policy array `21455208`; it keeps appending ready checkpoints and will automatically freeze validation-best endpoints and launch eight fixed 20/70 plus eight PW70 evaluations. | Training 48 CPU / 384 GiB; current policy batch 360 CPU / 1.2 TiB; watcher 2 CPU / 2 GiB |
| MCTS first-divergence audit | Corrected smoke `21453647` passed on commit `a46e0a84`. Reconciliation preserved/materialized 23 outcomes without rerun and submitted exactly 17 recoveries across seven source-task groups. **37/40 candidates are terminal**: 26 successes, five finished-unsolved and six exact timeouts. Three candidates remain (two BG/on fixed and one FO/off fixed). Interim mechanisms among 28 observed divergences are 16 Q-aligned visit winners, four exact visit ties, one other visit winner and seven goal-chase overrides; no policy action was excluded from expansion and none was uniquely exploration-U-aligned. FO-PW is complete at 6/6. | Two remaining grouped tasks allocate 4 CPU / 240 GiB; 26h task limits are conservative sequential-group bounds |
| Value-head V1 state-quality audit | Repaired smoke `21455175` passed. Capture arrays `21455176`-`21455179` are running for the eight paired domain/seed state manifests. Finalizer `21455180` and post-capture controller `21455181` automatically gate the measured resource preflight and 16 scientific checkpoint tasks. | 48 allocated CPU / 656 GiB during capture; downstream resources remain dependency-pending |

At this snapshot, running work allocates approximately 464 CPU / 2,562 GiB:
KL training 48/384, ready policy evaluations 360/1,200, the KL watcher 2/2,
V1 captures 48/656, Counters 4/200 and divergence recovery 2/120.
Dependency-pending controllers allocate nothing until their gates release.

## Scientific interpretation

- MPrime policy endpoints are 16.5/20 VH-off and 16.7/20 VH-on. Fixed and PW70
  search are now exact. PW70 exceeds fixed search in both modes at every
  cutoff; its 49 six-hour failures are all exact timeouts. MPrime now enters
  the Stage-2 RQ2/RQ4 extension.
- PRIMARY-50 is complete for all 50 independent primary checkpoints. It is a
  conditional update-effect diagnostic because deterministic-current KL
  replays each checkpoint's own legacy-generated data. The only
  Holm-significant source-to-treatment contrast is FO Counters under legacy
  KL (4.2 to 2.1; Holm p=.00977); deterministic-current attenuates the loss to
  2.7 but no deterministic-minus-legacy contrast is Holm-significant.
- The Counters primary RQ values remain frozen until task-0 identities 58/59
  and the exact full-root trace follow-up complete. This is distinct from the
  now-terminal task-1 slot 44 recovery.
- The compact MCTS first-divergence recorder passed its known-root compute
  smoke. The first grouped run preserved valid records and motivated an exact
  filtered repair rather than a broad rerun. The value-head V1 design now has
  all 16 exact checkpoint hashes and the exact search-time ENHSP transform;
  eight paired shared-state manifests still require new capture because the
  historical logs do not contain restorable physical+history state payloads.
  Neither audit is yet a primary-RQ result.

## RQ impact

RQ2 and RQ4 gain exact MPrime Stage-2 fixed/PW rows. RQ1/RQ3 do not change.
The PRIMARY-50 and endogenous-KL work remain method-development evidence and
do not replace the historical primary pipeline at this snapshot.

## Canonical sources

- `experiment_tracking/cluster_workload_latest.csv`
- `experiment_tracking/live_experiment_status_latest.csv`
- `experiment_tracking/mprime_final_stage2_search_20260916/`
- `experiment_tracking/imperfect_kl_primary_lineage_crossover_20260917/`
- `experiment_tracking/imperfect_kl_frozen_replay_crossover_20260917/`
- `experiment_tracking/mcts_policy_divergence_cause_audit/`
- `experiment_tracking/value_head_quality_audit/`

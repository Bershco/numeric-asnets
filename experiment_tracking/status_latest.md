# Current experiment status

Updated: 2026-09-18T18:05:00+03:00.

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State and scientific progress | Allocated resources |
|---|---|---:|
| MPrime final Stage-2 fixed/PW70 | **Complete: 800/800 scientific identities**. Final reconciliation contains 677 successes and 123 exact six-hour timeouts, with no action-limit failures, conflicts or unclassified cases. Exact 30m/2h/6h means are fixed/off 14.2/15.8/16.6, fixed/on 12.9/14.6/16.0, PW70/off 16.0/17.7/17.9 and PW70/on 15.2/17.0/17.2. The sixteen last wrappers have scheduler state `FAILED` only because the historical timeout path did not append JSONL after producing valid exact timeout evidence. | Terminal; zero live resources |
| Counters strict tie-break exact tail | Both duplicate task-1 recoveries finished evaluator slot 44 as an exact six-hour timeout; they are counted once. Their controllers exposed a separate stale task-0 gap: seed `534933607` has exactly evaluator identities 58 and 59 unclassified. Parallel pair `21453481_[0]` -> `21453482` and accidental retry pair `21453491_[0]` -> `21453492` currently target those same two identities and shared ledger. The later pair requires exact-ID cancellation authorization; no result will be double-counted. | Two duplicate recoveries currently allocate 8 CPU / 400 GiB; each is expected within the original six-hour per-instance allowance plus setup, with an 8h allocation |
| Endogenous 100-epoch KL pilot | Smoke `21453459` passed. Eight same-build Drone/FO-Counters training arms `21453460_[0-7]` are running: two canonical seeds per domain x legacy/deterministic-current KL, each generating its own replay. Policy-curve array `21453461_[0-167]` and finalizer `21453462` are dependency-gated. | 48 CPU / 384 GiB while training; historical estimate 3.4-11.3h per lineage, 18h allocation |
| MCTS first-divergence audit | Corrected smoke `21453647` passed on commit `a46e0a84`. Reconciliation preserved/materialized 23 outcomes without rerun and submitted exactly 17 recoveries across seven source-task groups. **37/40 candidates are terminal**: 26 successes, five finished-unsolved and six exact timeouts. Three candidates remain (two BG/on fixed and one FO/off fixed). Interim mechanisms among 28 observed divergences are 16 Q-aligned visit winners, four exact visit ties, one other visit winner and seven goal-chase overrides; no policy action was excluded from expansion and none was uniquely exploration-U-aligned. FO-PW is complete at 6/6. | Two remaining grouped tasks allocate 4 CPU / 240 GiB; 26h task limits are conservative sequential-group bounds |
| Value-head V1 state-quality audit | All 16 checkpoint hashes and the exact search-time `exp(-h)` ENHSP transform are frozen; 17 local tests pass. Smoke `21453653` exposed the missing ignored native operator. Smoke `21453918` then exposed a lossy serializer that omitted MDPSim special fluents; commit `2b86060b` now preserves the complete fluent vector and passes 17/17 tests. Repaired smoke `21454176` is resource-pending. Capture arrays `21454177`-`21454180` and finalizer `21454181` are correctly dependency-gated behind it and will create the paired 60-state manifests. | No V1 allocation while smoke waits. After admission: smoke 4 CPU / 48 GiB; then eight captures request 40 CPU / 656 GiB; finalizer 1 CPU / 8 GiB |

At this snapshot, the eight KL-training tasks allocate 48 CPU / 384 GiB, two
duplicate Counters recoveries allocate 8 CPU / 400 GiB, divergence recovery
allocates 4 CPU / 240 GiB, while the repaired V1 smoke is resource-pending:
60 CPU / 1,024 GiB total. Dependency-pending arrays and controllers allocate nothing
until their gates release.

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

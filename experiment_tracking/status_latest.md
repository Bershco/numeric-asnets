# Current experiment status

Updated: 2026-09-16T20:34:05+03:00 (live scheduler snapshot plus immediate retry verification)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State | Running / pending / complete | Concurrent CPU / RAM | Expected / hard bound |
|---|---|---:|---:|---|
| MPrime Phase-B-A validation rescore | 419/420 exact checkpoint evaluations durable; retry `21414946_100` running | 1 / 0 / 419 | 4 CPU / 20 GiB | normal rescores 16--29 m; retry 8 h hard limit |
| MPrime finalizer and downstream | `21412366` waits for the retry; `21412367` waits for finalization | 0 / 2 / 0 | 1 CPU / 2 GiB each when active | 30 m controller limits; downstream submits only exact missing policy endpoints, fixed 20/70 and PW70 |
| TPP KL multi-RNG | all 12 one-epoch trainings complete; endpoints 2 complete, 7 running, 3 exact environment-independent retries pending | 7 / 3 / 2 endpoints | currently 42 CPU / 140 GiB; retry maximum 15 CPU / 60 GiB | endpoint 2 h hard limit; observed completed endpoints about 6--7 m |
| Imperfect-domain KL screen v2 | exact historical-module replacement training active; endpoints dependency-pending | 20 / 20 / 0 | training 120 CPU / 2,400 GiB; endpoints later 100 CPU / 400 GiB | one epoch only; 12 h/2 h hard limits, not expected runtimes |
| Missing true-legacy Counters epoch-0 policy | exact historical checkpoint evaluation running | 1 / 0 / 0 | 5 requested CPU / 20 GiB | 2 h hard limit |

## Scientific status

- **MPrime:** this is validation rescoring, not policy inference. Twenty
  lineages x twenty-one saved epochs are scored on the frozen thirty-instance
  Phase-B-A validator. When the final score arrives, the fail-closed finalizer
  chooses maximum Phase-B-A coverage with earliest-epoch tie-break. The
  downstream controller joins exact checkpoint hashes to existing policy
  evidence, evaluates only genuinely missing selected endpoints, and builds
  matched fixed 20/70 and PW70 campaigns. No MPrime RQ row changes before that
  chain finishes.
- **TPP KL semantics:** the completed selected-pair result isolates a specific
  implementation mismatch: the legacy current-policy side of the anchor KL
  used dropout while the Stage-1 anchor was deterministic. Under RNG 314159,
  legacy scored 11/20 and deterministic-current scored 20/20 on the selected
  catastrophic seed; the stable control stayed 20/20. The live multi-RNG
  screen adds three optimizer-step RNG schedules, reuses the completed fourth,
  and tests repeatability without a guard or 100-epoch retraining.
- **Imperfect-domain KL semantics:** historical logs contain no decomposed
  anchor-gradient evidence, so a prospective screen is necessary. It runs two
  deliberately selected VH-off lineages in each of Block Grouping, Drone, FO
  Counters, Rover and Counters under same-build legacy versus deterministic-
  current KL for one epoch, then evaluates all endpoints. The first convenience-
  module attempt `21415079`/`21415080` was cancelled after Drone/Rover correctly
  failed on input-dimension mismatch. Only exact historical-module v2
  `21415209`/`21415210` is scientific evidence. Endpoint differences are
  descriptive unless all sixty replay batches, targets and pre-treatment
  gradients match across each treatment pair.
- **Advisor follow-up designs:** the methods/configuration inventory,
  outcome-stratified MCTS-policy divergence audit and gated value-head quality
  experiment are documented locally. No broad divergence or value-greedy jobs
  were submitted. The divergence audit starts with existing traces; value-head
  work starts with offline calibration/ranking before any greedy-value policy.

## RQ impact

There is no new primary RQ result at this snapshot. MPrime final checkpoint
selection and downstream policy/fixed/PW evaluation are still live. The TPP
and imperfect-domain KL experiments are implementation/mechanism diagnostics,
not replacements for historical primary rows.

## Canonical sources

- Live scheduler snapshot: `experiment_tracking/live_experiment_status_latest.csv`
- MPrime rescore: `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/README.md`
- TPP multi-RNG: `experiment_tracking/tpp_kl_multirng_susceptibility_20260916/`
- Imperfect-domain KL screen: `experiment_tracking/imperfect_domains_kl_semantics_screen_20260916/`
- Methods inventory: `docs/thesis_methods_configuration_inventory_20260916.md`
- Divergence audit: `experiment_tracking/mcts_policy_divergence_cause_audit/README.md`
- Value-head audit: `experiment_tracking/value_head_quality_audit/README.md`
- Experiment registry: `experiment_tracking/experiment_registry.csv`

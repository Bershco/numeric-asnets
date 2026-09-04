# Thesis experiment status — refreshed 2026-09-04 19:16 IDT

This is the authoritative narrative snapshot for this refresh. The newest
verified scheduler observation remains **2026-09-04 12:08:47 IDT**: three
correctly profiled and correctly spaced SSH attempts during this refresh timed
out before authentication, including a 120-second connection grace period. Dynamic
scheduler rows are therefore frozen in `cluster_workload_latest.csv`; grouped
resources are in `cluster_workload_summary_latest.csv`. They must not be read as
a 19:16 queue claim. All scores below are static evidence and remain current.
Counters is out of 59; every other domain is out of 20. MCTS scores are always
shown as 30-minute / 2-hour / 6-hour per-instance cutoffs when elapsed-time
evidence exists.

## 19:16 refresh decisions

- No Slurm job was submitted, cancelled, released or altered while the current
  queue could not be verified.
- MPRIME-VAL-ADEQUACY Phase B was **not** submitted. Readiness confidence is
  35%, below the user-approved 85% threshold. The protocol is sound, but the two
  frozen replicates, checksum manifests, planner sanity evidence, 60-lineage
  manifest and parameterized resumable wrapper do not yet exist.
- The readiness audit found a more important generator issue: the present
  validation generator inserts a direct two-action witness for every goal.
  Merely increasing object counts can still yield behaviorally easy instances.
  Phase B now requires a frozen planner-side plan-difficulty gate before any
  network checkpoint is viewed.
- MPrime policy-versus-MCTS completion is now explicit in
  `mprime_mcts_table_gap_20260904.csv`: 20 selected Stage-1 MCTS jobs and 40
  selected Stage-2 MCTS jobs are required after Phase B freezes defensible
  checkpoints. The separate declared Stage-1 final-endpoint contract adds 20
  more MCTS jobs, but those are not needed for the selected-checkpoint table.

## Actions and incident recovery in this refresh

- Delivery terminal-led policy evidence is complete. All twelve previously scoreless learning-curve/endpoint rows were exact checkpoint retries; all twelve scored successfully and no training was repeated.
- TPP/on training is terminal at valid Stage-2 snapshot 83. The original dependency controller failed only because the timeout log lacked its usual final-checkpoint footer. A continuation-aware replacement materialized all 214 every-five/selected/final policy identities.
- TPP/on has 206/214 unique policy-curve identities scored and the remaining eight are running; every selected endpoint is recovered and VAL-valid, so the ten-seed terminal-led endpoint result is complete.
- TPP/off has 206/213 unique policy-curve identities scored and the remaining seven are running. Both previously missing selected endpoints are recovered and VAL-valid, so its ten-seed terminal-led endpoint result is complete.
- The first Block Grouping/FO completion release produced nine scored endpoints and nine pre-inference ENOSPC failures. Every failed identity was recreated. Four first retries also landed on `ise-cpu128-04`; exact second retries were submitted.
- Rover was conditionally submitted when running requested memory was 5,380 GiB, below the user's 6-TiB gate. Thirteen originals failed on `ise-cpu128-04`; seven first retries then failed with native exit `-4` on six specific 40-core `ise-cpu-intl` nodes. Every one of the nineteen missing identities now has an active original or exact replacement: eleven are running and eight are resource-pending.
- Counters terminal-led Stage-2 MCTS was **not** submitted: the second gate requires all nineteen Rover identities to have a live running replacement and remaining memory capacity. That condition has not occurred.
- Node exclusions are deliberately exact rather than family-wide: `ise-cpu128-03`, `ise-cpu128-04`, and the directly observed SIGILL nodes `ise-cpu-intl-08,09,10,13,14,26,28`. Successful work remains eligible for the other 48–72-core `intl` nodes. Both controllers explicitly apply and verify `ExcNodeList` after submission.

## Last verified cluster workload

| State | Jobs | Requested CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 59 | 414 | 5,580 GiB |
| Pending | 10 | 60 | 1,200 GiB |
| Held | 0 | 0 | 0 GiB |

| Experiment | Running | Pending | CPUs running / pending | RAM running / pending |
|---|---:|---:|---:|---:|
| TPP terminal policy, VH-off | 7 | 0 | 70 / 0 | 140 / 0 GiB |
| TPP terminal policy, VH-on | 8 | 0 | 80 / 0 | 160 / 0 GiB |
| FO Counters terminal-led Stage-2 MCTS | 13 | 0 | 78 / 0 | 1,560 / 0 GiB |
| Block Grouping branch completion | 4 | 0 | 24 / 0 | 480 / 0 GiB |
| FO Counters validation-branch completion | 3 | 2 | 18 / 12 | 360 / 240 GiB |
| Rover validation-branch completion | 11 | 8 | 66 / 48 | 1,320 / 960 GiB |
| Counters exact-snapshot PW20/PW70 divergence recovery | 6 | 0 | 36 / 0 | 720 / 0 GiB |
| PW70 five-seed confirmation | 4 | 0 | 24 / 0 | 480 / 0 GiB |
| PW70 two-seed correction | 2 | 0 | 12 / 0 | 240 / 0 GiB |
| Counters binding-Horizon | 1 | 0 | 6 / 0 | 120 / 0 GiB |

The queue can change minute by minute as short policy jobs finish. The CSV snapshot—not this prose—is the machine-readable source.

## Live-experiment timing

| Experiment | Current position | Realistic timing |
|---|---|---|
| TPP terminal policy | Training and all ten selected endpoints per VH are complete; off curve 192/213, on curve 204/214 scored | Most remaining policy jobs should clear within roughly an hour after starting; four-hour hard cap. |
| FO terminal Stage-2 MCTS | 7/20 terminal, 13 running | Similar jobs have median about 48.3h and range 30.1–72h. Running jobs are about 9–52h old; expect staggered completions today through their 72h bounds. |
| BG validation completion | 8/10 off and 8/10 on static; four missing identities running | Comparable median 12.5h, range 10.2–19.2h. Current replacements started under one hour ago. |
| FO validation completion | 6/10 off and 9/10 on static; five missing identities have active replacements | Comparable median 48.3h, range 30.1–72h. |
| Rover validation completion | Historical evidence is 1/10 off and 0/10 on; all nineteen missing identities now have running/pending originals or exact replacements | Comparable median 30.1h, range 26.8–42.6h once inference actually starts. Pending start times are not predictable. |
| PW70 correction | 10/12 terminal; two Counters Stage-1/off running | One is near its 72h bound; the other has roughly 52h of hard allocation left. |
| PW70 confirmation | 14/18 new jobs terminal; four Counters Stage-2 jobs running | Running 33–49h; 23–39h remain to hard bounds. |
| Counters divergence recovery | Six exact snapshots, PW20 and PW70, all running | Running roughly 20–27h; 45–52h remain to hard bounds. |
| Counters Horizon | 7/8 terminal; one unaware VH-on job running | About four hours remain to its 72h hard bound. |

## PRESERVE-3 validation-led — completed

| Domain/VH | Stage-1 selected | Stage-2 selected all 10 | Held-out 8 | Tuning 2 | Change, 95% CI | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -0.2 [-1.01, 0.61] | 1.0 | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +0.4 [-0.37, 1.17] | .5 | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59, 1.39] | 1.0 | Not uniformly preserved |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -0.5 [-1.63, 0.63] | 1.0 | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0, 0] | 1.0 | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -0.1 [-0.33, 0.13] | 1.0 | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 scores 9/20. This is an audited seed-specific catastrophic-forgetting event, not a broad average decline.

## PRESERVE-3 terminal-led — current endpoint evidence

| Domain/VH | n | Stage-1 final | Stage-2 selected | Held-out | Tuning | Change, 95% CI | Raw p | Status |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 10 | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52, 5.12] | .496 | Complete |
| Delivery/on | 10 | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-0.91, 4.31] | .188 | Complete |
| TPP/off | 10 | 17.1 | 17.5 | 16.875 | 20.0 | +0.4 [-2.13, 2.93] | 1.0 | Complete endpoints; curve tail live |
| TPP/on | 10 | 18.9 | 18.3 | 17.875 | 20.0 | -0.6 [-3.77, 2.57] | 1.0 | Complete endpoints; curve tail live |
| Zenotravel/off | 10 | 20.0 | 19.8 | 19.875 | 19.5 | -0.2 [-0.50, 0.10] | .5 | Complete |
| Zenotravel/on | 10 | 19.8 | 20.0 | 20.0 | 20.0 | +0.2 [-0.10, 0.50] | .5 | Complete |

Delivery/on's held-out mean (18.625) is higher than its two tuning seeds (17.5), which is reassuring: its confirmation result is not an artifact of favourable tuning seeds. It is not a powered held-out-versus-tuning comparison because the tuning group has only two seeds.

## Stage-1 selected policy versus preferred MCTS — completed

| Domain/VH | Search | Policy | MCTS 30m / 2h / 6h | 6h change, 95% CI | Raw p | Holm p |
|---|---|---:|---:|---|---:|---:|
| Block Grouping/off | narrow 5/20 | 16.3 | 11.6 / 14.8 / 15.4 | -0.9 [-1.88, 0.08] | .109 | .547 |
| Block Grouping/on | narrow 5/20 | 15.9 | 12.0 / 14.0 / 16.2 | +0.3 [-0.29, 0.89] | .453 | .750 |
| Drone/off | normal 20/70 | 5.9 | 6.9 / 6.9 / 6.9 | +1.0 [0.05, 1.95] | .074 | .445 |
| Drone/on | normal 20/70 | 5.1 | 10.0 / 10.4 / 10.4 | +5.3 [3.42, 7.18] | .002 | .020 |
| FO Counters/off | normal 20/70 | 4.2 | 7.5 / 7.8 / 7.8 | +3.6 [2.20, 5.00] | .004 | .035 |
| FO Counters/on | normal 20/70 | 3.7 | 5.3 / 5.7 / 5.7 | +2.0 [1.05, 2.95] | .004 | .035 |
| Rover/off | normal 20/70 | 4.0 | 4.8 / 5.0 / 5.0 | +1.0 [0.25, 1.75] | .031 | .219 |
| Rover/on | normal 20/70 | 3.8 | 4.4 / 4.4 / 4.4 | +0.6 [-0.00, 1.20] | .125 | .547 |
| Counters/off | narrow 5/20 | 32.5 | 24.9 / 25.6 / 25.7 | -6.8 [-17.73, 4.13] | .250 | .750 |
| Counters/on | narrow 5/20 | 18.6 | 20.3 / 22.1 / 22.5 | +3.9 [-4.33, 12.13] | .328 | .750 |

Drone/on and both FO Counters cells remain significant after ten-cell Holm correction.

## Stage-2 policy versus MCTS: complete rows and live branch matrix

| Domain/VH | Branch | Search | n | Policy | MCTS 30m / 2h / 6h | 6h change, 95% CI | Raw p | Holm p |
|---|---|---|---:|---:|---:|---|---:|---:|
| BG/off | terminal | narrow 5/20 | 10 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22, -0.38] | .031 | .219 |
| BG/on | terminal | narrow 5/20 | 10 | 11.6 | 4.4 / 8.7 / 9.4 | -2.2 [-4.60, 0.20] | .031 | .219 |
| Drone/off | validation | normal 20/70 | 10 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [-0.43, 2.43] | .195 | .391 |
| Drone/on | validation | normal 20/70 | 10 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33, 8.07] | .002 | .020 |
| Drone/off | terminal | normal 20/70 | 10 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [0.65, 3.55] | .020 | .156 |
| Drone/on | terminal | normal 20/70 | 10 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [5.47, 7.73] | .002 | .020 |
| Rover/off | terminal | normal 20/70 | 10 | 3.8 | 4.2 / 4.2 / 4.2 | +0.4 [0.03, 0.77] | .125 | .375 |
| Rover/on | terminal | normal 20/70 | 10 | 4.0 | 4.5 / 4.5 / 4.5 | +0.5 [0.12, 0.88] | .063 | .313 |
| Counters/off | validation | narrow 5/20 | 10 | 36.9 | 34.9 / 36.7 / 36.7 | -0.2 [-2.87, 2.47] | .969 | .969 |
| Counters/on | validation | narrow 5/20 | 10 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-0.70, 11.30] | .082 | .328 |

Incomplete rows are tracked explicitly rather than presented as zero:

| Domain/VH | Validation branch terminal / live / unsubmitted | Terminal branch terminal / live / unsubmitted |
|---|---:|---:|
| BG/off | 8 / 2 / 0 | 10 / 0 / 0 |
| BG/on | 8 / 2 / 0 | 10 / 0 / 0 |
| Drone/off | 10 / 0 / 0 | 10 / 0 / 0 |
| Drone/on | 10 / 0 / 0 | 10 / 0 / 0 |
| FO/off | 6 / 4 / 0 | 4 / 6 / 0 |
| FO/on | 9 / 1 / 0 | 3 / 7 / 0 |
| Rover/off | 1 / 9 / 0 | 10 / 0 / 0 |
| Rover/on | 0 / 10 / 0 | 10 / 0 / 0 |
| Counters/off | 10 / 0 / 0 | 0 / 0 / 10 |
| Counters/on | 10 / 0 / 0 | 0 / 0 / 10 |

## Progressive widening

### PW70 five-seed confirmation — four cells complete

| Domain/VH | n | Policy | Fixed normal 20/70 | PW70 30m / 2h / 6h | PW-policy change, 95% CI; raw p | PW-fixed change, 95% CI; raw p |
|---|---:|---:|---:|---:|---|---|
| FO/off | 5 | 3.6 | 8.6 | 8.0 / 8.0 / 8.0 | +4.4 [2.52, 6.28]; .0625 | -0.6 [-1.28, 0.08]; .25 |
| FO/on | 5 | 3.2 | 5.4 | 7.2 / 7.2 / 7.2 | +4.0 [2.76, 5.24]; .0625 | +1.8 [0.18, 3.42]; .125 |
| Rover/off | 5 | 4.0 | 4.8 | 4.8 / 4.8 / 4.8 | +0.8 [-0.56, 2.16]; .50 | 0.0 [-1.96, 1.96]; 1.0 |
| Rover/on | 5 | 3.6 | 4.6 | 4.8 / 4.8 / 4.8 | +1.2 [-0.84, 3.24]; .50 | +0.2 [-2.19, 2.59]; 1.0 |

FO and Rover therefore satisfy the practical PW gate: PW improves on policy and is close to or better than fixed normal search, with no successful plan requiring more than 30 minutes. Five seeds still give a minimum two-sided exact sign-flip p-value of .0625, so these are strong directional but not conventional 0.05 significance results.

### PW70 correction and Counters confirmation — live

| Cell | n | Status | PW70 30m / 2h / 6h |
|---|---:|---|---:|
| BG S1/off | 2 | terminal | 10.5 / 11.5 / 13.0 |
| BG S1/on | 2 | terminal | 9.0 / 11.5 / 13.5 |
| Counters S1/off | 2 | running lower bounds | >=21.0 / >=21.0 / >=21.0 |
| Counters S1/on | 2 | terminal fixed-budget outcomes | 12.0 / 12.5 / 12.5 |
| Counters S2/off | 2 | terminal fixed-budget outcomes | 34.0 / 39.5 / 43.5 |
| Counters S2/on | 2 | terminal fixed-budget outcomes | 13.5 / 14.0 / 14.0 |

The six exact Counters snapshots selected for fixed-search regressions are also live under both PW20 and PW70. Current aggregate lower bounds across those six jobs are 125 / 139 / 148 successes at 30m / 2h / 6h over 251 classified instance records; no final comparison is yet valid.

## MPrime and validation adequacy

MPrime policy Stage-2 is complete but scientifically provisional because checkpoint validation is saturated:

| Branch/VH | Stage-1 | Stage-2 selected | Change, 95% CI | Raw p |
|---|---:|---:|---|---:|
| Validation/off | 15.0 | 15.2 | +0.2 [-1.18, 1.58] | .88 |
| Validation/on | 14.6 | 15.2 | +0.6 [-1.46, 2.66] | .58 |
| Terminal/off | 13.5 | 14.0 | +0.5 [-1.29, 2.29] | .672 |
| Terminal/on | 13.2 | 14.5 | +1.3 [-0.49, 3.09] | .172 |

Phase A of MPRIME-VAL-ADEQUACY is complete: 290/290 Stage-1 and 840/840 Stage-2 checkpoint records. Every Stage-2 checkpoint scores 30/30 on the current validation set, so the set cannot rank Stage-2 checkpoints or anchors. Phase B remains the highest-priority active preparation task. It must replace the generator's guaranteed direct-witness difficulty with two independently seeded, planner-difficulty-stratified and checksum-frozen validation replicates before rescoring existing checkpoints without new training.

## Horizon and determinism

- Drone Horizon is a completed non-result: same-commit aware/unaware mean difference 0.0, 95% CI [-0.89, 0.89], raw p=1.0; almost no cutoffs bound.
- Counters Horizon has seven terminal jobs and one running. Existing compact completion evidence has 138 classified instances, but emitted cutoff counters remain zero. No SAFE2 activation is justified unless nonzero cutoffs and cross-horizon state reuse are demonstrated.
- The determinism audit is complete and archived: same CPU-family repeats were action-identical; different CPU families changed prediction/statistic checksums but not selected actions or outcomes. It does not explain the old uncontrolled Horizon difference.

## Held designs, priority order

1. `MPRIME-VAL-ADEQUACY` Phase B — top priority; validation is demonstrably saturated.
2. `MCTS-PW-30M` — the FO/Rover five-seed results now satisfy its activation gate; submit a fresh matched hard-30m comparison when current memory pressure clears and explicit release is given.
3. `ANCHOR-KL-CONTROL` — literature-grounded constant versus adaptive target-KL versus scheduled KL test for Stage-2 collapse.
4. `MCTS-RESOURCE` — two workers/160 GiB to separate algorithm quality from resource censoring.
5. `STOP-ORIG` — original stopping-rule replication.
6. `LONG-DRONE-SELECTED-MCTS` — three held side-experiment endpoints.
7. `ACT-HISTORY-ABLATION` — user-held fresh-network campaign on Drone, Counters and mostly monotone TPP control.
8. `MCTS-PW-PATHBATCH` — nonstandard multiple-expansion design.
9. `PUCT-EST` — deferred causal sensitivity grid.
10. `MCTS-SAFE2` — low priority until the Horizon contamination gate is actually observed.

`MCTS-STAGE2-BRANCH-COMPLETION` Counters is an operational gate, not a scientific design hold: its 20-row narrow5/20 terminal-led manifest is ready locally but remains unsubmitted until the user's Rover-running-and-capacity condition is met.

## Provenance contract

The structural provenance audit covers every score-bearing CSV through direct log/job/path columns or an explicit row-level companion; zero result files are intentionally left without a backtrace. New authoritative files include:

- `preserve3_terminal_led/delivery_policy_retry_results_20260904.csv`
- `preserve3_terminal_led/tpp_on_policy_results_latest_20260904.csv`
- `preserve3_terminal_led/tpp_on_policy_node_retry_results_20260904.csv`
- `preserve3_terminal_led/terminal_stage2_tpp_off_policy_results_20260904.csv`
- `mcts_progressive_widening_cross_domain/pw70_followup_cutoffs_20260904.csv`
- `mcts_progressive_widening_cross_domain/pw70_confirmatory_seed_results_20260904.csv`
- `mcts_progressive_widening_cross_domain/pw70_confirmatory_statistics_20260904.csv`
- `mcts_progressive_widening_cross_domain/counters_divergence_cutoffs_20260904.csv`
- `stage2_mcts_branch_coverage_20260904.csv`

Every row-level file retains original cluster log paths and Slurm job IDs; aggregate tables point to those row-level companions instead of duplicating raw logs.

# Complete experiment snapshot — 2026-09-08T10:58:35+03:00

This report joins a fresh Slurm query with bounded reads of every live MCTS log,
the MPrime Phase-B result tree, and the authoritative static result ledgers.
`>=` denotes a conservative lower bound from an active or interrupted run.
Policy-only scores are invariant across the 30-minute/2-hour/6-hour MCTS cutoffs.

## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 70 | 294 | 2,480 GiB |
| Ordinary pending | 0 | 0 | 0 GiB |
| Slurm-held | 0 | 0 | 0 GiB |

| Live experiment | Running | CPU / RAM | Current evidence | Remaining bound / expectation |
|---|---:|---:|---|---|
| MPrime validation adequacy Phase B | 60 | 240 / 1,200 GiB | 1,050/2,260 checkpoint-replicates complete; all 60 lineages active | 7.1–8.6h allocation remains; continuation is likely because 53.5% remains |
| Counters terminal-led Stage-2 narrow MCTS | 3 | 18 / 360 GiB | Current live seed lower bounds: off 35/59; on 6/59 and 8/59 | 2.0–2.2h hard bound |
| TPP catastrophic-seed MCTS | 2 | 12 / 240 GiB | fixed-normal >=4; PW70 >=7; PW20 is terminal at 10/20; all versus policy 9/20 | <=53.2h; likely sooner for already-classified tails |
| FO terminal-led Stage-2 MCTS tail | 1 | 6 / 120 GiB | >=5/20; 17/20 classified | <=26.2h |
| PW70 correction tail | 1 | 6 / 120 GiB | Counters S1/off >=21/59; 42/59 classified | <=26.6h |
| Counters exact-snapshot PW70 recovery | 1 | 6 / 120 GiB | seed 534933607 >=22/59; 45/59 classified | <=19.1h |
| Rover exact-instance recovery | 2 | 6 / 320 GiB | 25/29 original opportunities classified as 6h timeouts; two active in task 7; exact instances 16/17 relaunched separately | original <=2.6h; follow-up <=12.3h |

No additional broad campaign was released. The only new job is Rover follow-up
21114871: one worker, two CPUs, 160 GiB, exactly instances 16 and 17 from
source job 20430090. The first submission 21114859 failed in zero seconds due
to a quoted TSV index and produced no evaluation evidence.

## Stage-1 validation-selected policy versus preferred fixed MCTS

Each cutoff cell is `MCTS mean; paired change [95% CI]; raw/Holm p`. Holm is
computed separately across the ten domain/VH cells at each cutoff.

| Domain/VH | Search | Policy | 30 minutes | 2 hours | 6 hours | Conclusion |
|---|---|---:|---|---|---|---|
| BG/off | narrow 5/20 | 16.3 | 11.6; -4.7 [-5.53,-3.87]; .002/.020 | 14.8; -1.5 [-2.68,-.32]; .035/.188 | 15.4; -.9 [-1.88,.08]; .109/.438 | Short budgets harm; no 6h gain |
| BG/on | narrow 5/20 | 15.9 | 12.0; -3.9 [-4.94,-2.86]; .002/.020 | 14.0; -1.9 [-2.94,-.86]; .010/.068 | 16.2; +.3 [-.29,.89]; .453/.906 | 6h neutral |
| Drone/off | normal 20/70 | 5.9 | 6.9; +1.0 [.05,1.95]; .074/.313 | 6.9; +1.0 [.05,1.95]; .074/.250 | 6.9; +1.0 [.05,1.95]; .074/.375 | Positive, not corrected-significant |
| Drone/on | normal 20/70 | 5.1 | 10.0; +4.9 [3.07,6.73]; .002/.020 | 10.4; +5.3 [3.42,7.18]; .002/.020 | 10.4; +5.3 [3.42,7.18]; .002/.020 | Significant at every cutoff |
| FO/off | normal 20/70 | 4.2 | 7.5; +3.3 [2.04,4.56]; .004/.027 | 7.8; +3.6 [2.20,5.00]; .004/.035 | 7.8; +3.6 [2.20,5.00]; .004/.035 | Significant gain |
| FO/on | normal 20/70 | 3.7 | 5.3; +1.6 [.83,2.37]; .004/.027 | 5.7; +2.0 [1.05,2.95]; .004/.035 | 5.7; +2.0 [1.05,2.95]; .004/.035 | Significant gain |
| Rover/off | normal 20/70 | 4.0 | 4.8; +.8 [.06,1.54]; .063/.313 | 5.0; +1.0 [.25,1.75]; .031/.188 | 5.0; +1.0 [.25,1.75]; .031/.219 | Positive, not Holm-significant |
| Rover/on | normal 20/70 | 3.8 | 4.4; +.6 [-.00,1.20]; .125/.313 | 4.4; +.6 [-.00,1.20]; .125/.250 | 4.4; +.6 [-.00,1.20]; .125/.438 | Neutral-positive |
| Counters/off | narrow 5/20 | 32.5 | 24.9; -7.6 [-18.52,3.32]; .203/.406 | 25.6; -6.9 [-17.95,4.15]; .250/.500 | 25.7; -6.8 [-17.73,4.13]; .250/.750 | Negative and highly variable |
| Counters/on | narrow 5/20 | 18.6 | 20.3; +1.7 [-8.94,12.34]; .787/.787 | 22.1; +3.5 [-5.56,12.56]; .453/.500 | 22.5; +3.9 [-4.33,12.13]; .328/.750 | Positive but highly variable |

**Conclusion:** fixed-search MCTS is robustly useful for Drone/on and both FO
cells. It is not a safe blanket replacement: Block Grouping needs the full
budget merely to reach parity and Counters/off regresses badly.

## Stage-2 policy versus fixed MCTS

The inferential endpoint is the declared six-hour result; lower cutoffs are
deterministic recensorings. Full cutoff-specific seed rows and provenance are
retained in the companion ledgers; interrupted lower-bound rows have no p-value.

| Domain/VH | Branch | Search | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | Raw/Holm p | Status/conclusion |
|---|---|---|---:|---:|---|---|---|
| BG/off | validation | narrow 5/20 | 16.0 | 11.4 / 15.0 / 15.7 | -.3 [-.89,.29] | .500 / family pending | Neutral at 6h |
| BG/on | validation | narrow 5/20 | 12.8 | 10.1 / 10.6 / 12.6 | -.2 [-1.08,.68] | .813 / family pending | Neutral at 6h |
| BG/off | terminal | narrow 5/20 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22,-.38] | .031/.219 | Negative |
| BG/on | terminal | narrow 5/20 | 11.6 | 4.4 / 8.7 / 9.4 | -2.2 [-4.60,.20] | .031/.219 | Negative mean |
| Drone/off | validation | normal 20/70 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [-.43,2.43] | .195/.391 | Positive, nonsignificant |
| Drone/on | validation | normal 20/70 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33,8.07] | .002/.020 | Significant gain |
| Drone/off | terminal | normal 20/70 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [.65,3.55] | .020/.156 | Positive, not Holm-significant |
| Drone/on | terminal | normal 20/70 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [5.47,7.73] | .002/.020 | Significant gain |
| FO/off | validation | normal 20/70 | 2.9 | >=6.0 / >=6.0 / >=6.0 | >=+3.1 | withheld | Two interrupted partial seeds |
| FO/on | validation | normal 20/70 | 3.1 | >=5.3 / >=5.4 / >=5.4 | >=+2.3 | withheld | Positive lower bound |
| FO/off | terminal | normal 20/70 | 2.8 | 5.3 / 5.3 / 5.3 | +2.5 [1.73,3.27] | .002/.010 | Significant gain |
| FO/on | terminal | normal 20/70 | 3.8 | >=4.5 / >=4.5 / >=4.5 | >=+.7 | withheld | 9 terminal + 1 live, current tail >=5 |
| Rover/off | validation | normal 20/70 | 4.0 | 4.5 / 4.5 / 4.5 | +.5 [-.01,1.01] | .125 / pending | Small nonsignificant gain |
| Rover/on | validation | normal 20/70 | 3.9 | 4.4 / 4.5 / 4.5 | +.6 [-.09,1.29] | .156 / pending | Small nonsignificant gain |
| Rover/off | terminal | normal 20/70 | 3.8 | 4.2 / 4.2 / 4.2 | +.4 [.03,.77] | .125/.375 | Small nonsignificant gain |
| Rover/on | terminal | normal 20/70 | 4.0 | 4.5 / 4.5 / 4.5 | +.5 [.12,.88] | .063/.313 | Small nonsignificant gain |
| Counters/off | validation | narrow 5/20 | 36.9 | 34.9 / 36.7 / 36.7 | -.2 [-2.87,2.47] | .969/.969 | Neutral |
| Counters/on | validation | narrow 5/20 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-.70,11.30] | .082/.328 | Positive, variable |
| Counters/off | terminal | narrow 5/20 | 37.9 | >=34.9 / >=37.8 / >=38.4 | >=+.5 | withheld | Three jobs live; lower bound above policy at 6h |
| Counters/on | terminal | narrow 5/20 | 16.8 | >=19.9 / >=22.0 / >=22.4 | >=+5.6 | withheld | Three jobs live across both modes |

**Conclusion:** Stage-2 inference-time MCTS is established for Drone/on and
FO/off, harmful for terminal-led Block Grouping, modest for Rover, and still
live for Counters/FO-on.

## PW70 ten-seed FO/Rover expansion

All 20 expansion allocations are terminal: 16 completed and four ended OOM.
The only unusable inferential seed is FO/off 2082152039, which has a 7/20 lower
bound with one unclassified instance. Thus FO/off is n=9; the other cells are
n=10. Every statistic below is paired; Holm is across the four cells at the
same cutoff and comparator.

| Cell | n | Policy | PW70 30m / 2h / 6h | PW-policy at 6h [95% CI]; raw/Holm p | PW-fixed 20/70 at 6h [95% CI]; raw/Holm p | Conclusion |
|---|---:|---:|---:|---|---|---|
| FO/off | 9 | 4.11 | 8.56 / 8.56 / 8.56 | +4.44 [3.58,5.31]; .004/.012 | +.78 [-.65,2.20]; .328/.984 | Significant policy gain; fixed parity; one OOM-partial seed |
| FO/on | 10 | 3.70 | 7.30 / 7.30 / 7.30 | +3.60 [2.70,4.50]; .002/.008 | +1.60 [.52,2.68]; .023/.094 | Significant over policy; raw gain over fixed |
| Rover/off | 10 | 4.00 | 4.70 / 4.70 / 4.70 | +.70 [.02,1.38]; .125/.250 | -.30 [-1.37,.77]; .688/1.0 | Fixed-search parity, not significant |
| Rover/on | 10 | 3.80 | 4.50 / 4.60 / 4.60 | +.80 [-.08,1.68]; .125/.250 | +.20 [-.80,1.20]; .828/1.0 | Fixed-search parity, not significant |

FO/on remains significant versus policy at 30m and 2h as well (Holm p=.008 at
both); its fixed-search comparison is Holm p=.031 at 30m and .094 at 2h.
FO/off is unchanged across cutoffs. Rover remains nonsignificant at every
cutoff. Full cutoff CIs/p-values are in `pw70_ten_seed_statistics_latest.csv`.

**Conclusion:** PW70 is a strong FO Counters result and a Rover parity result.
It is not a universal replacement for fixed search.

## Other live diagnostics

### TPP catastrophic seed

The source policy is 9/20. Fixed narrow terminated OOM after 18 classified
instances at 4/20; this is worse than policy. Fixed normal is >=4/20. PW20 is
now terminal at 5/20 under 30m and 10/20 under 2h/6h; all ten printed plans are
VAL-valid, so it recovers one net policy failure. PW70 is >=4/5/7. This is
one-seed mechanism evidence, not a TPP mean.

### Counters exact-snapshot widening

| Seed | Policy | Fixed narrow | PW20 30m / 2h / 6h | PW70 30m / 2h / 6h | Status |
|---|---:|---:|---:|---:|---|
| 534933607 | 59 | 23 | 19 / 21 / 21 | >=19 / >=21 / >=22 | PW70 live; neither recovers policy |
| 923500475 | 59 | 29 | 32 / 41 / 48 | 18 / 20 / 21 | PW20 partially recovers fixed-search loss |
| 2082152039 | 35 | 18 | 18 / 18 / 18 | 19 / 19 / 19 | Neither recovers policy |

### MPrime Phase B

The two independently frozen harder validation replicates are now being scored
against all 1,130 saved checkpoints (2,260 checkpoint-replicates). At this
snapshot 1,050 (46.5%) are complete, every lineage has results, and no lineage
is complete. The work is resumable. No MPrime MCTS should be submitted until
the two replicates yield stable checkpoint and anchor rankings.

## Best demonstrated result by domain

| Domain | Best current configuration | 30m / 2h / 6h or policy | Matched S1 policy | Paper | Delta vs S1 / paper | Conclusion |
|---|---|---:|---:|---:|---:|---|
| Delivery | S1 selected policy, off | 19.8 / 19.8 / 19.8 | 19.8 | 20 | 0 / -.2 | S2 preserves but does not improve the best S1 result |
| TPP | S1 policy | 20 / 20 / 20 | 20 | 20 | 0 / 0 | Perfect; one S2/off seed catastrophically forgets |
| Zenotravel | S1 and multiple S2 policy cells | 20 / 20 / 20 | 20 | 17 | 0 / +3 | Solved and preserved |
| MPrime | validation-led S2 policy, provisional | 15.2 / 15.2 / 15.2 | 15.0 | 19 | +.2 / -3.8 | Phase B may change checkpoint selection |
| Block Grouping | S1 selected policy, off | 16.3 / 16.3 / 16.3 | 16.3 | 17 | 0 / -.7 | Neither S2 nor MCTS improves the best policy |
| Drone | terminal-led S2 normal MCTS, on | 12.9 / 13.1 / 13.1 | 7.2 | 9 | +5.9 / +4.1 | Best established search result; Holm-significant |
| FO Counters | S1 PW70/off, n=9 | 8.56 / 8.56 / 8.56 | 4.11 | 6 | +4.45 / +2.56 | Current maximum; one missing complete seed |
| Rover | S1 fixed normal MCTS/off | 4.8 / 5.0 / 5.0 | 4.0 | 7 | +1.0 / -2.0 | Modest gain, still below paper |
| Counters | terminal-led S2 narrow/off, live | >=34.9 / >=37.8 / >=38.4 | 21.5 | 17 | >=+16.9 / >=+21.4 | Largest gain, but still a live lower bound |

## Completed RQs

### RQ1 — does MCTS-guided Stage-2 training improve VH-off policy?

| Domain | Validation-led change [95% CI]; Holm p | Terminal-led change [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -.3 [-1.20,.60]; 1.0 | 0 [-.95,.95]; 1.0 | No improvement |
| Drone | +.8 [-1.27,2.87]; 1.0 | +.4 [-1.33,2.13]; 1.0 | No reliable improvement |
| FO | -1.3 [-2.37,-.23]; .234 | -.8 [-1.46,-.14]; .219 | Raw negative; not Holm-significant |
| Rover | 0 [0,0]; 1.0 | 0 [-.34,.34]; 1.0 | No change |
| Counters | +4.4 [-17.44,26.24]; 1.0 | +16.4 [3.09,29.71]; .137 | Large noisy mean; not corrected-significant |

No original five-domain RQ1 result survives Holm correction.

### RQ3 — does the value head improve refinement?

| Domain | Validation-led DiD [95% CI]; Holm p | Terminal-led DiD [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -2.8 [-4.61,-.99]; .088 | -3.4 [-5.13,-1.67]; .020 | Terminal-led VH significantly worsens refinement |
| Drone | -.9 [-2.88,1.08]; 1.0 | -1.1 [-3.57,1.37]; .836 | No reliable effect |
| FO | +.7 [-.26,1.66]; .813 | +1.3 [.08,2.52]; .188 | Positive raw tendency only |
| Rover | +.1 [-.31,.51]; 1.0 | -.1 [-.51,.31]; 1.0 | No effect |
| Counters | -1.2 [-25.87,23.47]; 1.0 | -16.8 [-29.64,-3.96]; .078 | Large negative tendency, high variance |

Only terminal-led Block Grouping is significant after correction.

### RQ2/RQ4 — inference-time MCTS and value-head interaction

The complete branch-aware table is above. Current firm claims are: strong
Drone/on and FO/off gains; terminal-led BG loss; small Rover gains; live
Counters and FO/on cells. MPrime enters only after Phase B freezes checkpoints.

## Completed preservation experiments

### PRESERVE-3 validation-led

| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Conclusion |
|---|---:|---:|---:|---:|---|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -.2 [-1.01,.61] | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +.4 [-.37,1.17] | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | Not uniform |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 scores 9/20.

### PRESERVE-3 terminal-led

| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Conclusion |
|---|---:|---:|---:|---:|---|---|
| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52,5.12] | Preserved relative to source |
| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91,4.31] | Positive mean |
| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +.4 [-2.13,2.93] | Preserved on average |
| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -.6 [-3.77,2.57] | No reliable change |
| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -.2 [-.50,.10] | Essentially preserved |
| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +.2 [-.10,.50] | Preserved |

## Other completed experiments

| Experiment | Conclusion |
|---|---|
| MCTS-PW Drone | Faster with far fewer retained nodes, but unacceptable coverage loss under the original schedule |
| PW Kmin=3 Drone extension | 9.5/20 versus policy 7.0 and fixed 10.5; faster tail, still inferior to fixed |
| MCTS-SAFE-1 | Repaired 2/4 targeted Drone dead ends; useful guard, incomplete quality fix |
| MCTS-SAFE-CONTEXT | Negative: node splitting harmed coverage, especially VH-on |
| MCTS-HORIZON Drone/Counters | Non-results with zero effective cutoffs |
| Determinism audit | CPU families change numerical checksums, but audited actions/outcomes were identical |
| MCTS-WIDTH | BG and Counters confirmatory fixed search is narrow 5/20 |
| LONG-DRONE | Policy and final-checkpoint MCTS complete; three selected-checkpoint endpoints remain held |
| Storage audit/compaction | 17 oversized logs compacted; 27.90 GiB reclaimed in addition to earlier cleanup |

## Design-held experiments in priority order

| Priority | Experiment | Activation condition |
|---:|---|---|
| 1 | MCTS-PW-30M | Optional fresh whole-job efficiency validation after final PW tails; post-hoc coverage is already defensible |
| 2 | ANCHOR-KL-CONTROL | Implement/smoke-test adaptive or scheduled KL controller; use TPP collapse as diagnostic |
| 3 | MCTS-PW-PATHBATCH | Freeze nonstandard multiple-expansion update semantics |
| 4 | MCTS-RESOURCE | Use only for unresolved OOM/lifecycle endpoints |
| 5 | ACT-HISTORY-ABLATION | Separate fresh Stage-1/Stage-2 campaign on Drone, Counters, TPP |
| 6 | LONG-DRONE selected MCTS | Reprioritize side question |
| 7 | STOP-ORIG | Finalize compatibility manifest |
| 8 | PUCT-EST | Run only after higher-priority evidence closes |
| 9 | MCTS-SAFE2 | Require nonzero cutoffs and demonstrated cross-horizon statistic contamination |

## Provenance contract

- Current jobs/resources: `cluster_workload_latest.csv` and
  `cluster_workload_summary_latest.csv`.
- Every dynamically inspected job, direct stdout and completion ledger:
  `dynamic_experiment_jobs_latest.csv`.
- Stage-1 seed-level cutoff scores and both original logs:
  `stage1_policy_mcts_seed_cutoffs_latest.csv`.
- Stage-1 all-cutoff inference: `stage1_policy_mcts_all_cutoff_statistics_latest.csv`.
- PW70 seed/job/log rows: `mcts_progressive_widening_cross_domain/pw70_ten_seed_results_latest.csv`.
- PW70 all-cutoff inference: `mcts_progressive_widening_cross_domain/pw70_ten_seed_statistics_latest.csv`.
- Rover exact recovery scope: `rover_interrupted_mcts_recovery_manifest_20260908.csv`
  and `rover_interrupted_mcts_recovery_followup_manifest_20260908.csv`.
- Rover follow-up submission/job/log trace:
  `rover_interrupted_mcts_recovery_followup_submission_20260908.csv`.
- Full lifecycle registry: `experiments.csv` and `experiment_registry.csv`.

Aggregate CSVs either carry direct log paths or point through their
`row_level_provenance`/`results_file` companion to job-, checkpoint-, training-
and evaluation-log rows.

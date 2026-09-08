# Complete experiment snapshot — 2026-09-08T16:03:21+03:00

This report joins a fresh Slurm query with bounded reads of every live MCTS log,
the MPrime Phase-B result tree, and the authoritative static result ledgers.
`>=` denotes a conservative lower bound from an active or interrupted run.
Policy-only scores are invariant across the 30-minute/2-hour/6-hour MCTS cutoffs.

## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 65 | 270 | 2,040 GiB |
| Ordinary pending | 1 | 4 | 160 GiB |
| Slurm-held | 0 | 0 | 0 GiB |

| Live experiment | Running | CPU / RAM | Current evidence | Remaining bound / expectation |
|---|---:|---:|---|---|
| MPrime validation adequacy Phase B | 58 | 232 / 1,160 GiB | 1,359/2,260 checkpoint-replicates complete (60.1%); all 60 lineages represented, 2 complete | Original allocations have roughly 2.2–3.8h left; resumable continuation still required |
| TPP catastrophic-seed MCTS | 2 | 12 / 240 GiB | fixed-normal 4/20 after 13 classified; PW70 7/20 after 10 classified; policy 9/20 | Fixed-normal about 12h; PW70 likely <=6h because only its final three workers remain active |
| FO terminal-led Stage-2 MCTS tail | 1 | 6 / 120 GiB | Five persisted resumable successes; instances 6–8 active. Historical concatenated attempts contain 19 classifications but cannot all be skipped | Next three classifications <=2.6h; allocation has ~20.5h left and cannot guarantee all 15 unresolved instances |
| PW70 correction tail | 1 | 6 / 120 GiB | Counters S1/off seed 2011206605: >=21/59; 44 classified | <=21.4h allocation bound; likely scheduler-limited |
| Counters exact-snapshot PW70 recovery | 1 | 6 / 120 GiB | seed 534933607: >=22/59; 48 classified | <=14h allocation bound; likely scheduler-limited |
| Rover exact-instance recovery | 1 running + 1 pending | 2 / 160 GiB running; 4 / 160 GiB pending | 25/29 classified, all six-hour timeouts; four exact opportunities remain | Running instance 16 should classify shortly, then 17 within 6h; pending task runs only 17/19 together in <=6h after start |

Two exact MPrime replacements were released as array 21128409 (tasks 18 and
53). Both predecessor tasks failed immediately on `ise-cpu-intl-01` with
native evaluator exit -4; the replacements are running on other nodes and
skip every already validated checkpoint-replicate. Rover task 7's pending
manifest now contains only instances 17 and 19, so its already observed
six-hour timeouts on instances 15 and 16 will not be repeated.

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
| FO/on | terminal | normal 20/70 | 3.8 | >=4.5 / >=4.5 / >=4.5 | >=+.7 | withheld | One restarted tail remains; five persisted successes, instances 6–8 active |
| Rover/off | validation | normal 20/70 | 4.0 | 4.5 / 4.5 / 4.5 | +.5 [-.01,1.01] | .125 / pending | Small nonsignificant gain |
| Rover/on | validation | normal 20/70 | 3.9 | 4.4 / 4.5 / 4.5 | +.6 [-.09,1.29] | .156 / pending | Small nonsignificant gain |
| Rover/off | terminal | normal 20/70 | 3.8 | 4.2 / 4.2 / 4.2 | +.4 [.03,.77] | .125/.375 | Small nonsignificant gain |
| Rover/on | terminal | normal 20/70 | 4.0 | 4.5 / 4.5 / 4.5 | +.5 [.12,.88] | .063/.313 | Small nonsignificant gain |
| Counters/off | validation | narrow 5/20 | 36.9 | 34.9 / 36.7 / 36.7 | -.2 [-2.87,2.47] | .969/.969 | Neutral |
| Counters/on | validation | narrow 5/20 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-.70,11.30] | .082/.328 | Positive, variable |
| Counters/off | terminal | narrow 5/20 | 37.9 | 34.9* / 37.8* / 38.4* | +.5 [-.34,1.34] | .313/.313† | Scheduler-terminal; no significant 6h change |
| Counters/on | terminal | narrow 5/20 | 16.8 | 19.9* / 22.0* / 22.4* | +5.6 [1.84,9.36] | .0078/.0156† | Significant positive effect within the two-mode Counters family |

`*` Declared-budget means include all ten seeds. OOM/scheduler-timeout
allocations count unclassified instances as failures rather than disappearing
from the mean. `†` Holm correction is across the two Counters VH modes; the
complete cross-domain Stage-2 family correction will be frozen after FO/on.

**Conclusion:** Stage-2 inference-time MCTS is established for Drone/on,
FO/off and Counters/on; harmful for terminal-led Block Grouping; modest for
Rover; and still live only for FO/on. Counters/off reaches a slightly higher
mean than policy at six hours, but the +0.5 difference is not significant.

## PW70 ten-seed FO/Rover expansion

All 20 expansion allocations are terminal: 16 completed and four ended OOM.
Every declared-budget seed is retained. FO/off seed 2082152039 contributes
7/20* after ten classified instances; the ten unclassified instances count as
failures. A complete-allocation sensitivity analysis excluding that one seed
remains available (n=9, mean 8.56), but it is not the headline mean. Every
statistic below is paired; Holm is across the four cells at the same cutoff
and comparator.

| Cell | n | Policy | PW70 30m / 2h / 6h | PW-policy at 6h [95% CI]; raw/Holm p | PW-fixed 20/70 at 6h [95% CI]; raw/Holm p | Conclusion |
|---|---:|---:|---:|---|---|---|
| FO/off | 10* | 4.20 | 8.40 / 8.40 / 8.40 | +4.20 [3.26,5.14]; .002/.008 | +.60 [-.71,1.91]; .422/1.0 | Significant policy gain; fixed parity; one OOM-partial seed |
| FO/on | 10 | 3.70 | 7.30 / 7.30 / 7.30 | +3.60 [2.70,4.50]; .002/.008 | +1.60 [.52,2.68]; .023/.094 | Significant over policy; raw gain over fixed |
| Rover/off | 10 | 4.00 | 4.70 / 4.70 / 4.70 | +.70 [.02,1.38]; .125/.250 | -.30 [-1.37,.77]; .688/1.0 | Fixed-search parity, not significant |
| Rover/on | 10 | 3.80 | 4.50 / 4.60 / 4.60 | +.80 [-.08,1.68]; .125/.250 | +.20 [-.80,1.20]; .828/1.0 | Fixed-search parity, not significant |

`*` FO/off seed 2082152039 is the 7/20 OOM-partial declared result; all
unclassified instances count unsuccessful. Excluding it gives the secondary
complete-allocation estimate 8.56/20 over nine seeds.

FO/off and FO/on are significant versus policy at 30m, 2h and 6h (Holm
p=.008 at each cutoff). FO/on exceeds fixed search at 30m after Holm correction
(p=.031), but its 2h/6h fixed-search differences do not survive correction.
FO/off is unchanged across cutoffs. Rover remains nonsignificant at every
cutoff. Full cutoff CIs/p-values are in `pw70_ten_seed_statistics_latest.csv`.

**Conclusion:** PW70 is a strong FO Counters result and a Rover parity result.
It is not a universal replacement for fixed search.

## Other live diagnostics

### TPP catastrophic seed

The source policy is 9/20. Fixed narrow terminated OOM after 18 classified
instances at 4/20; this is worse than policy. PW20 is terminal at 5/20 under
30m and 10/20 under 2h/6h; all ten printed plans are VAL-valid, so it recovers
one net policy failure. Fixed normal is 4/20 after 13 classified and is now on
instances 14–16; its rolling three-worker schedule needs about 12 more hours
to classify all 20 if every remaining instance times out. PW70 is 4/5/7 after
ten classified; only instances 18–20 remain active, so it should terminate in
at most about six hours, but earlier worker failures mean it may remain
partially classified. This is one-seed mechanism evidence, not a TPP mean.

### Counters exact-snapshot widening

| Seed | Policy | Fixed narrow | PW20 30m / 2h / 6h | PW70 30m / 2h / 6h | Status |
|---|---:|---:|---:|---:|---|
| 534933607 | 59 | 23 | 19 / 21 / 21 | >=19 / >=21 / >=22 | PW70 live at 48/59 classified; neither recovers policy |
| 923500475 | 59 | 29 | 32 / 41 / 48 | 18 / 20 / 21 | PW20 partially recovers fixed-search loss |
| 2082152039 | 35 | 18 | 18 / 18 / 18 | 19 / 19 / 19 | Neither recovers policy |

### MPrime Phase B

The two independently frozen harder validation replicates are now being scored
against all 1,130 saved checkpoints (2,260 checkpoint-replicates). At this
snapshot 1,359 (60.1%) are complete; every lineage has results and two lineages
are complete. Tasks 18 and 53 failed within two minutes on
`ise-cpu-intl-01` with native evaluator exit -4. Their exact resumable
replacements, array 21128409, are running on other nodes and skip every valid
result. The original allocations end in roughly 2.2–3.8 hours, so a bounded
continuation over only remaining checkpoint-replicates will still be needed.
No MPrime Stage-2 training/MCTS should be submitted until the two replicates
yield stable checkpoint and anchor rankings.

### Rover interrupted-instance recovery

Twenty-five of 29 previously unclassified opportunities have now been
classified, and every one reached the declared six-hour per-instance timeout;
none added a plan. This leaves all existing Rover aggregate scores unchanged.
The four remaining opportunities are source 20430090 instances 16/17 and
source 20430103 instances 17/19. Job 21114871 is handling the first pair
sequentially with one worker; pending task 21107687_7 was narrowed to the
second pair and will use two independent workers. Two workers are defensible
because instances run in separate processes with instance-derived fixed seeds
and share no MCTS tree or statistics; it only parallelizes independent trials.

## Best demonstrated result by domain

| Domain | Best current configuration | 30m / 2h / 6h or policy | Matched S1 policy | Paper | Delta vs S1 / paper | Conclusion |
|---|---|---:|---:|---:|---:|---|
| Delivery | S1 selected policy, off | 19.8 / 19.8 / 19.8 | 19.8 | 20 | 0 / -.2 | S2 preserves but does not improve the best S1 result |
| TPP | S1 policy | 20 / 20 / 20 | 20 | 20 | 0 / 0 | Perfect; one S2/off seed catastrophically forgets |
| Zenotravel | S1 and multiple S2 policy cells | 20 / 20 / 20 | 20 | 17 | 0 / +3 | Solved and preserved |
| MPrime | validation-led S2 policy, provisional | 15.2 / 15.2 / 15.2 | 15.0 | 19 | +.2 / -3.8 | Phase B is 60.1% complete and may change checkpoint selection |
| Block Grouping | S1 selected policy, off | 16.3 / 16.3 / 16.3 | 16.3 | 17 | 0 / -.7 | Neither S2 nor MCTS improves the best policy |
| Drone | terminal-led S2 normal MCTS, on | 12.9 / 13.1 / 13.1 | 7.2 | 9 | +5.9 / +4.1 | Best established search result; Holm-significant |
| FO Counters | S1 PW70/off, n=10 declared-budget* | 8.40 / 8.40 / 8.40 | 4.20 | 6 | +4.20 / +2.40 | Current maximum; includes one 7/20 OOM-partial allocation |
| Rover | S1 fixed normal MCTS/off | 4.8 / 5.0 / 5.0 | 4.0 | 7 | +1.0 / -2.0 | Modest gain, still below paper |
| Counters | terminal-led S2 narrow/off* | 34.9 / 37.8 / 38.4 | 21.5 | 17 | +16.9 / +21.4 | Largest result; versus its own S2 policy37.9 the 6h change is only +.5 and nonsignificant |

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
Drone/on, FO/off and Counters/on gains; terminal-led BG loss; small Rover
gains; and one still-live FO/on cell. MPrime enters only after Phase B freezes
checkpoints.

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
- Counters terminal-led Stage-2 ten-seed cutoff rows and direct logs:
  `counters_terminal_stage2_narrow_seed_results_latest.csv`.
- Counters terminal-led Stage-2 cutoff CIs/tests:
  `counters_terminal_stage2_narrow_statistics_latest.csv`.
- Rover exact recovery scope: `rover_interrupted_mcts_recovery_manifest_20260908.csv`
  and `rover_interrupted_mcts_recovery_followup_manifest_20260908.csv`.
- Rover follow-up submission/job/log trace:
  `rover_interrupted_mcts_recovery_followup_submission_20260908.csv`.
- Full lifecycle registry: `experiments.csv` and `experiment_registry.csv`.
- Best demonstrated domain configurations: `best_configuration_by_domain_latest.csv`.

Aggregate CSVs either carry direct log paths or point through their
`row_level_provenance`/`results_file` companion to job-, checkpoint-, training-
and evaluation-log rows.

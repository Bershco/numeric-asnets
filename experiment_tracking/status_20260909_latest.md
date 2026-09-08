# Complete experiment snapshot — 2026-09-09T00:43:18+03:00

This report joins the 00:43 IDT Slurm query with bounded reads of every live MCTS log,
the MPrime Phase-B result tree, and the authoritative static result ledgers.
`>=` denotes a conservative lower bound from an active or interrupted run.
Policy-only scores are invariant across the 30-minute/2-hour/6-hour MCTS cutoffs.

## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 59 | 248 | 1,776 GiB |
| Ordinary pending | 0 | 0 | 0 GiB |
| Slurm-held | 0 | 0 | 0 GiB |

| Live experiment | Running | CPU / RAM | Current evidence | Remaining bound / expectation |
|---|---:|---:|---|---|
| MPrime validation adequacy Phase B | 52 | 208 / 1,040 GiB | 1,608/2,260 checkpoint-replicates complete (71.2%); all 60 lineages represented, 7 complete | Current tasks have roughly 15–23h to allocation limits; aggregate throughput suggests another ~13h, but slow lineage tails may require another idempotent continuation |
| Adaptive KL-control TPP/off | 2 | 12 / 96 GiB | First adaptive checkpoints exist: outlier validation 22/30 with post-update KL .0484; stable control 30/30 with KL .0189; coefficient remains at floor 3 | Existing comparable runs took ~16h for the stable seed and ~30h for the outlier; 72h hard cap |
| TPP catastrophic-seed MCTS | 1 | 6 / 120 GiB | Fixed-normal 4/20 after 19 classified; policy 9/20 | One instance remains; <=6h |
| FO terminal-led Stage-2 MCTS tail | 1 | 6 / 120 GiB | 5/20 after 19 durably classified | One instance remains; <=6h |
| PW70 correction tail | 1 | 6 / 120 GiB | Counters S1/off seed 2011206605: >=21/59; 48 classified | ~13h allocation remains; exact continuation likely if the tail does not fit |
| Counters exact-snapshot PW70 recovery | 1 | 6 / 120 GiB | Seed 534933607: >=22/59; 52 classified | <6h allocation remains; exact continuation likely |
| Rover exact-instance recovery | 1 | 4 / 160 GiB | 27/29 classified, all as six-hour timeouts; final two active | <=4.3h from this snapshot |

There are no ordinary-pending or deliberately held Slurm jobs. The two adaptive
jobs are the only newly released work in this refresh. The current allocation is
well below the approximately 6 TiB effective user ceiling, but no other held
design has an equally unambiguous activation decision; those remain a user
priority choice rather than being silently released.

## Stage-1 validation-selected policy versus preferred fixed MCTS

Each cutoff cell is `MCTS mean; paired change [95% CI]; raw/Holm p`. Holm is
computed separately across the ten domain/VH cells at each cutoff.

| Domain/VH | Search | Policy | 30 minutes | 2 hours | 6 hours | Conclusion |
|---|---|---:|---|---|---|---|
| BG/off | narrow 5/20 | 16.3 | 11.6; -4.7 [-5.53,-3.87]; .002/.020 | 14.8; -1.5 [-2.68,-.32]; .035/.188 | 15.4; -.9 [-1.88,.08]; .109/.547 | Short budgets harm; no 6h gain |
| BG/on | narrow 5/20 | 15.9 | 12.0; -3.9 [-4.94,-2.86]; .002/.020 | 14.0; -1.9 [-2.94,-.86]; .010/.068 | 16.2; +.3 [-.29,.89]; .453/.750 | 6h neutral |
| Drone/off | normal 20/70 | 5.9 | 6.9; +1.0 [.05,1.95]; .074/.313 | 6.9; +1.0 [.05,1.95]; .074/.297 | 6.9; +1.0 [.05,1.95]; .074/.445 | Positive, not corrected-significant |
| Drone/on | normal 20/70 | 5.1 | 10.0; +4.9 [3.07,6.73]; .002/.020 | 10.4; +5.3 [3.42,7.18]; .002/.020 | 10.4; +5.3 [3.42,7.18]; .002/.020 | Significant at every cutoff |
| FO/off | normal 20/70 | 4.2 | 7.5; +3.3 [2.04,4.56]; .004/.027 | 7.8; +3.6 [2.20,5.00]; .004/.035 | 7.8; +3.6 [2.20,5.00]; .004/.035 | Significant gain |
| FO/on | normal 20/70 | 3.7 | 5.3; +1.6 [.83,2.37]; .004/.027 | 5.7; +2.0 [1.05,2.95]; .004/.035 | 5.7; +2.0 [1.05,2.95]; .004/.035 | Significant gain |
| Rover/off | normal 20/70 | 4.0 | 4.8; +.8 [.06,1.54]; .063/.313 | 5.0; +1.0 [.25,1.75]; .031/.188 | 5.0; +1.0 [.25,1.75]; .031/.219 | Positive, not Holm-significant |
| Rover/on | normal 20/70 | 3.8 | 4.4; +.6 [-.00,1.20]; .125/.375 | 4.4; +.6 [-.00,1.20]; .125/.375 | 4.4; +.6 [-.00,1.20]; .125/.547 | Neutral-positive |
| Counters/off | narrow 5/20 | 32.5 | 24.9; -7.6 [-18.52,3.32]; .203/.406 | 25.6; -6.9 [-17.95,4.15]; .250/.500 | 25.7; -6.8 [-17.73,4.13]; .250/.750 | Negative and highly variable |
| Counters/on | narrow 5/20 | 18.6 | 20.3; +1.7 [-8.94,12.34]; .787/.787 | 22.1; +3.5 [-5.56,12.56]; .453/.500 | 22.5; +3.9 [-4.33,12.13]; .328/.750 | Positive but highly variable |

**Conclusion:** fixed-search MCTS is robustly useful for Drone/on and both FO
cells. It is not a safe blanket replacement: Block Grouping needs the full
budget merely to reach parity and Counters/off regresses badly.

## Stage-2 policy versus fixed MCTS

Each cutoff cell is `MCTS mean; paired change [95% CI]; raw/Holm p`.
Holm is computed across the 17 currently complete domain/VH/branch cells at
each cutoff. It remains interim until FO/on terminal-led becomes the 18th complete cell.

| Domain/VH | Branch | Search | n | Policy | 30 minutes | 2 hours | 6 hours | Visible conclusion |
|---|---|---|---:|---:|---|---|---|---|
| BG/off | terminal | narrow5/20 | 10 | 16.3 | 11.1; -5.20 [-6.14,-4.26]; 0.002/0.0332 | 13.1; -3.20 [-4.01,-2.39]; 0.002/0.0332 | 14.5; -1.80 [-3.22,-0.38]; 0.0312/0.375 | MCTS loses coverage at every cutoff; per-instance cutoff reconstruction corrected 2026-09-08 |
| BG/on | terminal | narrow5/20 | 10 | 11.6 | 9.2; -2.40 [-3.17,-1.63]; 0.002/0.0332 | 9.5; -2.10 [-3.02,-1.18]; 0.0078/0.1016 | 10.5; -1.10 [-2.08,-0.12]; 0.0625/0.6875 | MCTS loses mean coverage; per-instance cutoff reconstruction corrected 2026-09-08 |
| BG/off | validation | narrow5/20 | 10 | 16 | 11.4; -4.60 [-5.37,-3.83]; 0.002/0.0332 | 15; -1.00 [-2.01,0.01]; 0.0938/0.75 | 15.7; -0.30 [-0.89,0.29]; 0.5/1 | No reliable gain or loss at6h;30m is materially worse |
| BG/on | validation | narrow5/20 | 10 | 12.8 | 10.1; -2.70 [-3.38,-2.02]; 0.002/0.0332 | 10.6; -2.20 [-3.26,-1.14]; 0.0078/0.1016 | 12.6; -0.20 [-1.08,0.68]; 0.8125/1 | No reliable gain or loss at6h;30m is materially worse |
| Drone/off | terminal | normal20/70 | 10 | 7.8 | 9.7; +1.90 [0.49,3.31]; 0.0254/0.2539 | 9.8; +2.00 [0.53,3.47]; 0.0254/0.2539 | 9.9; +2.10 [0.65,3.55]; 0.0195/0.2539 | Positive paired effect but not Holm-significant |
| Drone/on | terminal | normal20/70 | 10 | 6.5 | 12.9; +6.40 [5.32,7.48]; 0.002/0.0332 | 13.1; +6.60 [5.47,7.73]; 0.002/0.0332 | 13.1; +6.60 [5.47,7.73]; 0.002/0.0332 | Large significant gain |
| Drone/off | validation | normal20/70 | 10 | 6.7 | 7.5; +0.80 [-0.58,2.18]; 0.3594/1 | 7.7; +1.00 [-0.43,2.43]; 0.1953/0.875 | 7.7; +1.00 [-0.43,2.43]; 0.1953/1 | Positive mean but not significant |
| Drone/on | validation | normal20/70 | 10 | 5 | 10.9; +5.90 [4.23,7.57]; 0.002/0.0332 | 11.2; +6.20 [4.33,8.07]; 0.002/0.0332 | 11.2; +6.20 [4.33,8.07]; 0.002/0.0332 | Large significant gain |
| FO/off | terminal | normal20/70 | 10 | 2.8 | 5.3; +2.50 [1.73,3.27]; 0.002/0.0332 | 5.3; +2.50 [1.73,3.27]; 0.002/0.0332 | 5.3; +2.50 [1.73,3.27]; 0.002/0.0332 | Strong significant gain; duplicated retry output deduplicated and VAL-certified |
| Rover/off | terminal | normal20/70 | 10 | 3.8 | 4.2; +0.40 [0.03,0.77]; 0.125/1 | 4.2; +0.40 [0.03,0.77]; 0.125/0.875 | 4.2; +0.40 [0.03,0.77]; 0.125/1 | Small non-significant gain |
| Rover/on | terminal | normal20/70 | 10 | 4 | 4.5; +0.50 [0.12,0.88]; 0.0625/0.5625 | 4.5; +0.50 [0.12,0.88]; 0.0625/0.5625 | 4.5; +0.50 [0.12,0.88]; 0.0625/0.6875 | Small non-significant gain |
| Rover/off | validation | normal20/70 | 10 | 4 | 4.5; +0.50 [-0.01,1.01]; 0.125/1 | 4.5; +0.50 [-0.01,1.01]; 0.125/0.875 | 4.5; +0.50 [-0.01,1.01]; 0.125/1 | Positive mean; not significant |
| Rover/on | validation | normal20/70 | 10 | 3.9 | 4.4; +0.50 [-0.2,1.2]; 0.25/1 | 4.5; +0.60 [-0.09,1.29]; 0.1562/0.875 | 4.5; +0.60 [-0.09,1.29]; 0.1562/1 | Positive mean; not significant |
| Counters/off | terminal | narrow5/20 | 10 | 37.9 | 34.9; -3.00 [-7.54,1.54]; 0.2266/1 | 37.8; -0.10 [-1.59,1.39]; 1/1 | 38.4; +0.50 [-0.34,1.34]; 0.3125/1 | Declared-budget mean includes OOM/timeout allocations as failures for unclassified instances; small positive mean |
| Counters/on | terminal | narrow5/20 | 10 | 16.8 | 19.9; +3.10 [-3.43,9.63]; 0.3398/1 | 22; +5.20 [1.02,9.38]; 0.0195/0.2148 | 22.4; +5.60 [1.84,9.36]; 0.0078/0.1094 | Declared-budget mean includes OOM/timeout allocations as failures for unclassified instances; positive but variable mean |
| Counters/off | validation | narrow5/20 | 10 | 36.9 | 34.9; -2.00 [-6.93,2.93]; 0.5/1 | 36.7; -0.20 [-2.87,2.47]; 0.9688/1 | 36.7; -0.20 [-2.87,2.47]; 0.9688/1 | No meaningful change |
| Counters/on | validation | narrow5/20 | 10 | 21.8 | 22.6; +0.80 [-8.28,9.88]; 0.8691/1 | 26.4; +4.60 [-1.83,11.03]; 0.1523/0.875 | 27.1; +5.30 [-0.69,11.29]; 0.082/0.7383 | Positive mean but high variance |
| FO/on | terminal | normal20/70 | 9 terminal + 1 live | 3.8 | >=4.5 lower bound | >=4.5 lower bound | >=4.5 lower bound | One live seed remains; one terminal seed timed out at4/20 and current seed is at5/20 |

**Current conclusion:** after the single 17-cell family correction, Drone/on is
significantly improved in both branches and FO/off is significantly improved in its
terminal-led branch at all three cutoffs (Holm p=.0332). Counters/on terminal-led
has a large positive mean but no longer survives this broader correction at six hours
(Holm p=.1094). Block Grouping is negative, Rover is small/neutral, and FO/on remains live.

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
one net policy failure. Fixed normal is still 4/20 but now has 19 classified,
leaving one active instance and at most six hours. PW70 terminated OOM at
4/5/7 under 30m/2h/6h after 12 classified; the eight unclassified instances
count unsuccessful in this declared allocation. This is one-seed mechanism
evidence, not a TPP population mean.

### Counters exact-snapshot widening

| Seed | Policy | Fixed narrow | PW20 30m / 2h / 6h | PW70 30m / 2h / 6h | Status |
|---|---:|---:|---:|---:|---|
| 534933607 | 59 | 23 | 19 / 21 / 21 | >=19 / >=21 / >=22 | PW70 live at 52/59 classified; neither recovers policy |
| 923500475 | 59 | 29 | 32 / 41 / 48 | 18 / 20 / 21 | PW20 partially recovers fixed-search loss |
| 2082152039 | 35 | 18 | 18 / 18 / 18 | 19 / 19 / 19 | Neither recovers policy |

### MPrime Phase B

The two independently frozen harder validation replicates are now being scored
against all 1,130 saved checkpoints (2,260 checkpoint-replicates). At this
snapshot 1,608 (71.2%) are complete; every lineage has results and seven lineages
are complete. Exact continuation array `21143254` now covers every inactive
lineage while tasks 2, 18, 52 and 53 continue in their existing allocations.
It requests 4 CPUs and 20 GiB per task, skips existing results, and all submitted
tasks were admitted immediately. No MPrime Stage-2 retraining or MCTS should
be submitted until the two replicates yield stable checkpoint and anchor
rankings.

### Rover interrupted-instance recovery

Twenty-seven of 29 previously unclassified opportunities have now been
classified, and every one reached the declared six-hour per-instance timeout;
none added a plan. This leaves all existing Rover aggregate scores unchanged.
Job `21114871` classified source 20430090 instances 16/17 as timeouts. Job
`21107687_7` is now running the exact final source-20430103 instances 17/19 in
two independent processes; it does not share MCTS state or alter per-instance
seeds and should finish within six hours.

### Adaptive KL-control screen

The adaptive-only screen is now live. Existing constant-anchor runs are reused,
so only two new training jobs are required:

| Role | Seed | Constant evidence | Adaptive job | Resources | State |
|---|---:|---|---:|---:|---|
| Catastrophic outlier | 1972442430 | Stage-2 epoch 0 fell from 20/20 to 10/20; selected endpoint 9/20 | 21144388 | 6 CPU / 48 GiB | Epoch 0 saved; validation 22/30; post-update KL .0484 |
| Stable control | 1963100312 | Stable constant-anchor training | 21144389 | 6 CPU / 48 GiB | Epoch 0 saved; validation 30/30; post-update KL .0189 |

The first submissions, `21144210` and `21144211`, failed before training after
about one minute because a fresh Git worktree does not contain the ignored
compiled TensorFlow operator `_asnet_ops_impl.so`. The isolated checkout now
links the checksum-verified production build. Strengthened compute smoke
`21144340` imported that operator, ran all five controller tests, verified both
CLI options and completed successfully. The retry ledger explicitly links each
new job to its failed precursor, source Stage-1 training job, checkpoint and log.

The decision criterion remains: adaptive control should prevent or materially
reduce the outlier's first-update collapse without damaging the stable seed.
The current jobs are training-only; matched policy evaluation is materialized
after their checkpoints exist.

The first realized KL values are below the target band, so the controller made
no upward adjustment; because coefficient 3 is the declared floor, it also did
not reduce protection. These are validation diagnostics, not test-policy scores.

## Best demonstrated result by domain

| Domain | Best current configuration | 30m / 2h / 6h or policy | Matched S1 policy | Paper | Delta vs S1 / paper | Conclusion |
|---|---|---:|---:|---:|---:|---|
| Delivery | S1 selected policy, off | 19.8 / 19.8 / 19.8 | 19.8 | 20 | 0 / -.2 | S2 preserves but does not improve the best S1 result |
| TPP | S1 policy | 20 / 20 / 20 | 20 | 20 | 0 / 0 | Perfect; one S2/off seed catastrophically forgets |
| Zenotravel | S1 and multiple S2 policy cells | 20 / 20 / 20 | 20 | 17 | 0 / +3 | Solved and preserved |
| MPrime | validation-led S2 policy, provisional | 15.2 / 15.2 / 15.2 | 15.0 | 19 | +.2 / -3.8 | Phase B is 71.2% complete and may change checkpoint selection |
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

The branch-aware all-cutoff table above is authoritative. Under one interim Holm
family across the 17 complete cells, Drone/on (both branches) and FO/off terminal-led
remain significant at six hours. Counters/on terminal-led is a large positive mean
that does not survive the broader family correction; terminal-led BG is negative,
Rover is small/neutral, and FO/on remains live. MPrime enters only after Phase B
freezes defensible checkpoints.

## Completed preservation experiments

### PRESERVE-3 validation-led

| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -.2 [-1.01,.61] | 1.0 | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +.4 [-.37,1.17] | .5 | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | 1.0 | Not uniform |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | 1.0 | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | 1.0 | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | 1.0 | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 scores 9/20.

### PRESERVE-3 terminal-led

| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52,5.12] | .496 | Preserved relative to source |
| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91,4.31] | .188 | Positive mean |
| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +.4 [-2.13,2.93] | 1.0 | Preserved on average |
| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -.6 [-3.77,2.57] | 1.0 | No reliable change |
| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -.2 [-.50,.10] | .5 | Essentially preserved |
| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +.2 [-.10,.50] | .5 | Preserved |

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
| 1 | MCTS-PW-30M | Optional fresh whole-job efficiency validation; post-hoc cutoff coverage is already defensible |
| 2 | MCTS-PW-PATHBATCH | Freeze nonstandard multiple-expansion update semantics |
| 3 | MCTS-RESOURCE | Use only for unresolved OOM/lifecycle endpoints |
| 4 | ACT-HISTORY-ABLATION | Separate fresh Stage-1/Stage-2 campaign on Drone, Counters and TPP |
| 5 | LONG-DRONE selected MCTS | Reprioritize the remaining side question |
| 6 | STOP-ORIG | Finalize compatibility manifest |
| 7 | PUCT-EST | Run only after higher-priority evidence closes |
| 8 | MCTS-SAFE2 | Require nonzero cutoffs and demonstrated cross-horizon statistic contamination |

`ANCHOR-KL-CONTROL` is no longer held; it is the two-job live experiment above.

## Provenance contract

- Current jobs/resources: `cluster_workload_latest.csv` and
  `cluster_workload_summary_latest.csv`.
- Every dynamically inspected job, direct stdout and completion ledger:
  `dynamic_experiment_jobs_latest.csv`.
- Stage-1 seed-level cutoff scores and both original logs:
  `stage1_policy_mcts_seed_cutoffs_latest.csv`.
- Stage-1 all-cutoff inference: `stage1_policy_mcts_all_cutoff_statistics_latest.csv`.
- Stage-2 seed-level cutoff scores and direct/companion log provenance:
  `stage2_policy_mcts_seed_cutoffs_latest.csv`.
- Stage-2 cutoff-specific paired CIs, exact tests, and interim Holm families:
  `stage2_policy_mcts_all_cutoff_statistics_latest.csv`.
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
- Adaptive KL frozen rows: `anchor_kl_control_tpp_screen_20260908.csv`.
- Adaptive KL initial and retry job/log lineage: `anchor_kl_control_submissions_20260909.tsv` and `anchor_kl_control_retry_submissions_20260909.tsv`.
- Adaptive KL frozen rows: `anchor_kl_control_tpp_screen_20260908.csv`.
- Adaptive KL initial and retry job/log lineage: `anchor_kl_control_submissions_20260909.tsv` and `anchor_kl_control_retry_submissions_20260909.tsv`.
- Adaptive KL frozen rows: `anchor_kl_control_tpp_screen_20260908.csv`.
- Adaptive KL initial and retry job/log lineage: `anchor_kl_control_submissions_20260909.tsv` and `anchor_kl_control_retry_submissions_20260909.tsv`.
- Adaptive KL frozen rows: `anchor_kl_control_tpp_screen_20260908.csv`.
- Adaptive KL initial and retry job/log lineage: `anchor_kl_control_submissions_20260909.tsv` and `anchor_kl_control_retry_submissions_20260909.tsv`.
- Adaptive KL frozen rows: `anchor_kl_control_tpp_screen_20260908.csv`.
- Adaptive KL initial and retry job/log lineage: `anchor_kl_control_submissions_20260909.tsv` and `anchor_kl_control_retry_submissions_20260909.tsv`.

Aggregate CSVs either carry direct log paths or point through their
`row_level_provenance`/`results_file` companion to job-, checkpoint-, training-
and evaluation-log rows.

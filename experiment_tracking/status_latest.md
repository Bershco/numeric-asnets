# Complete RQ and experiment snapshot — 2026-09-09T10:46:40+03:00

This report uses the fresh live scheduler capture only for jobs that were live in the prior report. Completed results come from locally cached authoritative ledgers. `>=` always denotes an active-run lower bound.

## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 37 | 156 | 996 GiB |
| Ordinary pending | 0 | 0 | 0 GiB |
| Slurm-held | 0 | 0 | 0 GiB |

| Live experiment | Jobs | CPU / RAM | Latest evidence | Realistic timing |
|---|---:|---:|---|---|
| MPrime validation adequacy Phase B | 33 | 132 / 660 GiB | 2,014/2,260 checkpoint-replicates (89.1%); 25/60 lineages complete | Aggregate throughput suggests ~6h; older allocations have under6h hard bounds, so a small idempotent tail may still be required |
| Adaptive KL-control TPP/off | 2 | 12 / 96 GiB | Outlier through epoch37; stable control through epoch53; zero coefficient changes | Stable control ~8h to epoch100; outlier ~17h at current rates; both may stop earlier |
| FO Counters terminal-led S2 MCTS tail | 1 | 6 / 120 GiB | 5/20, 19/20 classified, 14 explicit timeouts | One instance; <=2.5h allocation bound |
| PW70 cross-domain correction tail | 1 | 6 / 120 GiB | Counters S1/off seed2011206605: >=21/59, 52 classified | <=3.4h allocation bound |

There are no ordinary-pending or Slurm-held jobs. Design-held experiments appear later and do not occupy the scheduler.

## Research questions

### RQ1 — does MCTS-guided Stage-2 training improve VH-off policy?







| Domain | Validation-led change [95% CI]; Holm p | Terminal-led change [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -.3 [-1.20,.60]; 1.0 | 0 [-.95,.95]; 1.0 | No improvement |
| Drone | +.8 [-1.27,2.87]; 1.0 | +.4 [-1.33,2.13]; 1.0 | No reliable improvement |
| FO | -1.3 [-2.37,-.23]; .234 | -.8 [-1.46,-.14]; .219 | Raw negative; not Holm-significant |
| Rover | 0 [0,0]; 1.0 | 0 [-.34,.34]; 1.0 | No change |
| Counters | +4.4 [-17.44,26.24]; 1.0 | +16.4 [3.09,29.71]; .137 | Large noisy mean; not corrected-significant |

No original five-domain RQ1 result survives Holm correction.

### RQ2 — does inference-time MCTS improve coverage?

The complete Stage-1 and branch-aware Stage-2 tables below answer this question at 30m, 2h and 6h. The p-values are paired exact sign-flip tests; Holm correction is recomputed separately at each cutoff.

#### Stage-1 validation-selected policy versus preferred fixed MCTS

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

##

##

#### Stage-2 policy versus fixed MCTS

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

### RQ3 — does the value head improve Stage-2 refinement?







| Domain | Validation-led DiD [95% CI]; Holm p | Terminal-led DiD [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -2.8 [-4.61,-.99]; .088 | -3.4 [-5.13,-1.67]; .020 | Terminal-led VH significantly worsens refinement |
| Drone | -.9 [-2.88,1.08]; 1.0 | -1.1 [-3.57,1.37]; .836 | No reliable effect |
| FO | +.7 [-.26,1.66]; .813 | +1.3 [.08,2.52]; .188 | Positive raw tendency only |
| Rover | +.1 [-.31,.51]; 1.0 | -.1 [-.51,.31]; 1.0 | No effect |
| Counters | -1.2 [-25.87,23.47]; 1.0 | -16.8 [-29.64,-3.96]; .078 | Large negative tendency, high variance |

Only terminal-led Block Grouping is significant after correction.

### RQ4 — does the value head alter the benefit of inference-time MCTS?

The Stage-2 table under RQ2 is also the authoritative RQ4 evidence. The strongest reproducible pattern is Drone/on: large Holm-significant gains in both validation- and terminal-led branches. FO/off is also significant in the terminal-led branch. Block Grouping is negative, Rover is small/neutral, Counters is variable, FO/on is awaiting one final instance, and MPrime awaits Phase B checkpoint freezing.

## Learning curves — locally cached and immediately re-plottable

The source rows, aggregate rows, figures and direct training/evaluation-log
mapping are frozen under `experiment_tracking/learning_curves/latest/`.

| View | File | Use |
|---|---|---|
| RQ1 + RQ3 combined | `by_rq_rq1_rq3.png` / `.svg` | Advisor overview; five domains x two RQs |
| RQ1 only | `by_rq_rq1.png` / `.svg` | Stage-2 policy refinement without VH interaction |
| RQ3 only | `by_rq_rq3.png` / `.svg` | Value-head comparison |
| Per domain | `by_domain_<domain>.svg` | One-domain discussion or slide |
| Tidy source | `rq1_rq3_policy_curve_source.csv` | Fast restyling/replotting without cluster access |
| Aggregate source | `rq1_rq3_five_domain_learning_curve_aggregates.csv` | Mean/median/best/min-max curves |
| Provenance | `learning_curve_provenance.csv` | Direct training job, evaluation job and log paths |

The figures support the numerical conclusions below: Stage-2 does not provide a
consistent policy-only gain across domains; Block Grouping shows the clearest
negative VH interaction, while Counters is high-variance and should never be
summarized only by its mean.


## Experiments

### Live and recently completed experiment updates

#### MPRIME-VAL-ADEQUACY Phase B

Two independently frozen harder validation replicates are being evaluated on
all 1,130 saved checkpoints: **2,014/2,260 results (89.1%)**, all 60 lineages
represented, 25 lineages complete, 33 jobs active. No new training is involved.
The next decision is whether both replicates agree on checkpoint and anchor
rankings. Only then should MPrime Stage-2 and its eventual 40 MCTS comparisons
be released.

#### ANCHOR-KL-CONTROL — adaptive KL screen

Purpose: test whether controlling actual policy drift prevents TPP/off seed
1972442430 from collapsing after the first Stage-2 update without damaging a
stable control seed.

| Role | Job | Epochs recorded | Validation first -> latest (range) | Max post-update KL | Coefficient | Adjustments |
|---|---:|---:|---|---:|---:|---:|
| Catastrophic outlier | 21144388 | 38 | 22/30 -> 29/30 (21–30/30) | 0.089548 | 3.0 | 0 |
| Stable control | 21144389 | 54 | 30/30 -> 30/30 (30–30/30) | 0.021586 | 3.0 | 0 |

The intervention is currently a **controller no-op**: coefficient 3 has never
changed. The upper trigger is approximately .17145 (target .1143 x tolerance
1.5), while the largest post-update KL observed is below .090. Values below the
lower band cannot reduce the coefficient because 3 is the configured floor.
The live runs therefore add diagnostics but have not yet tested a different
training objective from the constant-anchor baseline. Test-policy scores are
still required; validation alone cannot establish repair.

#### Remaining MCTS tails

| Experiment | Current result | What remains | Consequence |
|---|---|---|---|
| FO terminal-led S2/on | >=5/20 at every cutoff, 19 classified | One instance, <=2.5h allocation bound | Final eighteenth Stage-2 cell and final Holm family update |
| PW70 correction, Counters S1/off seed2011206605 | >=21/59, 52 classified | Seven classifications, <=3.4h allocation bound | Finishes the corrected PW70 narrow-domain screen |

#### Diagnostics completed since the prior report

| Experiment | 30m / 2h / 6h | Conclusion |
|---|---|---|
| TPP catastrophic seed: policy | 9 / 9 / 9 | Collapsed Stage-2 policy baseline |
| Fixed narrow 5/20 | 4 / 4 / 4 | Worse than policy; OOM after18 classified |
| PW20 | 5 / 10 / 10 | Only arm to improve, by one plan at2h/6h; OOM after19 classified |
| Fixed normal 20/70 | 4 / 4 / 4 | Complete20-instance run; no recovery |
| PW70 | 4 / 5 / 7 | OOM after12 classified; no recovery |
| Rover interrupted-instance recovery | 0 / 0 / 0 new plans | All29 exact opportunities became six-hour timeouts; aggregate Rover scores unchanged |
| Counters exact-snapshot widening | see focused ledger | Neither PW20 nor PW70 restores all policy successes; PW20 seed923500475 reaches48/59 and is the strongest partial recovery |


### PW70 ten-seed FO/Rover expansion

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

#

#

### Completed preservation experiments

###### PRESERVE-3 validation-led

| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -.2 [-1.01,.61] | 1.0 | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +.4 [-.37,1.17] | .5 | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | 1.0 | Not uniform |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | 1.0 | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | 1.0 | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | 1.0 | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 scores 9/20.

###### PRESERVE-3 terminal-led

| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52,5.12] | .496 | Preserved relative to source |
| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91,4.31] | .188 | Positive mean |
| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +.4 [-2.13,2.93] | 1.0 | Preserved on average |
| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -.6 [-3.77,2.57] | 1.0 | No reliable change |
| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -.2 [-.50,.10] | .5 | Essentially preserved |
| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +.2 [-.10,.50] | .5 | Preserved |

#

#

### Other completed experiments

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

#

#

### Design-held experiments in priority order

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

#

#

### Complete experiment catalog

The row-level catalog is `experiment_catalog_latest.csv`. The table below keeps every registered experiment visible while using the detailed sections above for scores.

| Experiment | State | What it tested | Result / next decision |
|---|---|---|---|
| MAIN-VAL | completed or inactive | Does Stage 2 improve a validation-selected Stage-1 network? | Policy experiment complete at 300/300 endpoints; do not routinely reprint unless requested |
| MAIN-TERM | completed or inactive | Does Stage 2 improve the terminal Stage-1 network? | Policy experiment complete; do not routinely reprint unless requested |
| PRESERVE-3-VAL | completed or inactive | Do the three stable domains preserve their Stage-1 performance after validation-led Stage 2? | Zenotravel is preserved and Delivery is essentially preserved; TPP is not fully preserved because of one audited9/20 off-mode collapse |
| PRESERVE-3-TERM | completed or inactive | Does Stage 2 preserve coverage when initialized from the Stage-1 final checkpoint? | All 60 training lineages and all selected endpoints complete; all nineteen final TPP curve repair jobs completed; freeze as completed. |
| MPRIME-VAL | completed or inactive | Repair structurally unrepresentative validation generation and rerun Stage 1 | Corrected Stage1 and all 290 policy jobs are terminal; validation improves selected versus final means but pooled Spearman agreement is about 0.25; preservation failed so MPrime moves to the six-domain imperfect extension |
| MAIN-EXT6-MPRIME | completed or inactive | Do MAIN-VAL conclusions extend from the original five imperfect domains to MPrime as a sixth? | Retain results as provisional extension and complete MPRIME-VAL-ADEQUACY before further MPrime training |
| MAIN-TERM-EXT6-MPRIME | completed or inactive | Does Stage 2 improve the corrected terminal Stage-1 MPrime network? | All twenty training and420 policy evaluations terminal; selected epoch0 means14.0 off and14.5 on and neither paired change is significant |
| ANCHOR-4 | completed or inactive | Select Stage-2 anchor coefficient without test-set tuning | MPrime anchor10 wins both VH; validation-stability audit remains separate |
| MCTS-WIDTH | completed or inactive | Does width 5 avoid waste/timeouts relative to width 20? | Counters Stage1 and Stage2 are terminal; Stage2 off is 36.7/59 at 2h/6h versus policy 36.9 and on is 26.4/59 at 2h or 27.1/59 at 6h versus policy 21.8; post-hoc VAL jobs 20768679--20768681 found zero invalid plans |
| MCTS-PW | completed or inactive | Can adaptive policy-ordered widening improve coverage/runtime/memory? | 30/30 terminal and VAL-valid; PW is significantly faster and retains far fewer nodes but significantly loses coverage |
| MCTS-SAFE | completed or inactive | Can safe external action masking prevent MCTS from converting policy successes into battery dead ends? | SAFE-1 is useful but incomplete; MCTS-SAFE-CONTEXT and MCTS-SAFE-2 are separately documented follow-ups |
| MCTS-SAFE2 | held design | Does indexing statistics by physical state and remaining executable horizon prevent cross-horizon value contamination? | Activate only after nonzero cutoffs and measurable cross-horizon state reuse establish a contamination mechanism |
| MCTS-SAFE-CONTEXT | completed or inactive | Does physical-state aliasing across different action-count network inputs distort priors values coverage and efficiency? | Ten matched pairs are terminal; contextual nodes reduce VH-off by 0.4 and VH-on by 3.4 plans; neither VH result survives Holm correction; retain diagnostics but do not promote behavior |
| MCTS-HORIZON | completed or inactive | Does a binding finite executable horizon prevent unreachable-depth waste? | All twenty arms and post-hoc VAL are complete; paired mean change is zero with 95 percent CI -0.89 to 0.89 and sign-flip p 1.0; almost no cutoffs bound so Counters is the proper efficacy follow-up |
| MCTS-HORIZON-COUNTERS | completed or inactive | Does horizon enforcement help where executions genuinely approach ten thousand actions? | Freeze as a non-result: both VH modes have zero aware-minus-unaware change at6h and zero recorded cutoffs. |
| MCTS-PW-SAFE | completed or inactive | Can extra initial width faster widening or 140 simulations recover PW coverage while retaining efficiency? | All eight Kmin3 jobs are terminal; Kmin3 averages 9.5 versus policy 7.0 and top20 10.5; freeze rather than silently expanding |
| MCTS-PW-CROSS-DOMAIN | completed or inactive | Does Kmin3 retain fixed-search coverage while reducing runtime outside Drone? | 20/20 terminal and printed plans VAL-confirmed; never label the whole screen PW20 because the eight FO Counters/Rover rows use PW70 |
| MCTS-PW70-CROSS-DOMAIN | live | What changes when the accidental 20-simulation PW screen is rerun with the intended 70 simulations? | Only Counters Stage1/off seed2011206605 remains active: >=21/59 after52 classified; allocation hard bound approximately3.4h. |
| MCTS-PW70-CONFIRMATORY | completed or inactive | Do promising two-seed cells replicate with enough matched seeds for paired intervals and tests? | All five seed cells terminal; expand FO/Rover to ten with separately registered20 jobs; Counters PW70 does not match policy consistently. |
| MCTS-PW-30M | held design | Does PW retain most MCTS benefit under a paper-style 30-minute instance budget? | Post-hoc cutoffs suffice for solutions-within-budget coverage; fresh runs needed only for whole-job resource/time validation or missing/censored timing evidence. |
| MAIN-VAL-S2-MCTS | completed or inactive | Does MCTS improve validation-selected Stage2 policies? | All twenty exact endpoints terminal with VAL evidence |
| MCTS-LEGACY-ROVER | completed or inactive | Complete the already-submitted Rover endpoint evidence before FO Counters | Off policy3.8 to MCTS4.2 CI[0.03 0.77] rawp.125; on policy4.0 to MCTS4.5 CI[0.12 0.88] rawp.0625; OOM allocations retained as fixed-budget outcomes |
| MCTS-LEGACY-FO | live | Complete the fifth mainstream Stage2 policy/MCTS domain comparison | Final terminal-led VH-on seed remains5/20 after19 classified and14 explicit timeouts; one exact instance is active with approximately2.5h allocation bound. |
| MCTS-STAGE2-BRANCH-COMPLETION | completed or inactive | Complete branch-balanced Stage2 policy/MCTS evidence without treating absent jobs as zero | Counters terminal-led narrow is scheduler-terminal at off38.4 and on22.4 at6h; FO terminal/on retains one live resumable tail. |
| MCTS-PW-PATHBATCH | held design | Can collecting all widening-eligible nodes on one selection path recover batching and useful breadth without excessive successor generation? | Freeze update semantics and compare coverage runtime generated states batch size retained nodes and memory |
| MCTS-RESOURCE | held design | Can lifecycle-safe 2-worker/160GB continuations finish without OOM? | Deploy tested lifecycle commit, then resume |
| LONG-DRONE | held design | Does training beyond 24h improve the declared endpoint? | All six policy endpoints and three final-checkpoint MCTS endpoints are complete; release the three selected-checkpoint MCTS only if this side comparison is prioritized |
| STOP-ORIG | held design | Effect of original training-success early stopping rather than validation stopping | Finalize compatibility implementation and launch manifest |
| PUCT-EST | held design | Separate PUCT and estimator contributions | Do not submit until released |
| ENHSP-LEAF | completed or inactive | Can a stronger ENHSP configuration improve leaf valuation? | Archive final logs/results |
| BG-HIST | completed or inactive | Recover and explain historical 18/20 | Archive provenance; do not use width-5/3 as primary |
| MCTS-DETERMINISM-AUDIT | completed or inactive | Where does same-seed same-configuration MCTS first diverge? | CPU-family numerical differences change internal checksums but no observed action; aware/unaware replay is identical with zero cutoffs |
| ACT-HISTORY-ABLATION | held design | Does the cumulative grounded-action count feature help policy while harming search through contextual aliasing? | Keep held; TPP is the control because bought is monotone nondecreasing and on-sale monotone nonincreasing; only drive location can cycle |
| MCTS-PW-COUNTERS-DIVERGENCE | completed or inactive | Can PW recover policy successes lost when fixed narrow MCTS diverges or times out? | All PW20/PW70 exact-snapshot arms terminal. Neither widening mode fully recovers the policy; PW20 seed923500475 gives the strongest partial recovery at48/59. |
| ANCHOR-KL-CONTROL | live | Can literature-grounded nonconstant KL control prevent first-update collapse without blocking later improvement? | Jobs21144388/21144389 reached Stage2 epochs34/52. Coefficient remains3 in every epoch and controller adjustments remain zero; evaluate matched test checkpoints before any efficacy claim. |
| MPRIME-VAL-ADEQUACY | live | Does validation rank checkpoints and anchors reliably without saturating? | 2014/2260 checkpoint-replicates complete;25/60 lineages complete and33 tasks active. Finish or resume only missing points, compare replicate rankings, then freeze checkpoints/anchors. |
| MCTS-PW70-TEN-SEED | completed or inactive | Do five-seed PW70 findings survive all ten original seeds? | All four cells contain ten declared-budget seeds; FO/off includes one explicitly starred 7/20 OOM-partial seed and has mean8.4. |
| TPP-CATASTROPHIC-MCTS | completed or inactive | Can search recover the eleven policy failures caused by seed-specific Stage2 catastrophic forgetting? | Policy9/20; fixed narrow4; PW20 5/10/10; fixed normal4; PW70 4/5/7 at30m/2h/6h. Only PW20 recovers one net policy success. |
| ROVER-MCTS-INTERRUP-REC | completed or inactive | Classify only instances omitted by scheduler or OOM interruption | All29 formerly unclassified instances reached the declared six-hour timeout; Rover aggregates are unchanged. |

### Best demonstrated result by domain

| Domain | Best current configuration | 30m / 2h / 6h or policy | Matched S1 policy | Paper | Delta vs S1 / paper | Conclusion |
|---|---|---:|---:|---:|---:|---|
| Delivery | S1 selected policy, off | 19.8 / 19.8 / 19.8 | 19.8 | 20 | 0 / -.2 | S2 preserves but does not improve the best S1 result |
| TPP | S1 policy | 20 / 20 / 20 | 20 | 20 | 0 / 0 | Perfect; one S2/off seed catastrophically forgets |
| Zenotravel | S1 and multiple S2 policy cells | 20 / 20 / 20 | 20 | 17 | 0 / +3 | Solved and preserved |
| MPrime | validation-led S2 policy, provisional | 15.2 / 15.2 / 15.2 | 15.0 | 19 | +.2 / -3.8 | Phase B is 89.1% complete and may change checkpoint selection |
| Block Grouping | S1 selected policy, off | 16.3 / 16.3 / 16.3 | 16.3 | 17 | 0 / -.7 | Neither S2 nor MCTS improves the best policy |
| Drone | terminal-led S2 normal MCTS, on | 12.9 / 13.1 / 13.1 | 7.2 | 9 | +5.9 / +4.1 | Best established search result; Holm-significant |
| FO Counters | S1 PW70/off, n=10 declared-budget* | 8.40 / 8.40 / 8.40 | 4.20 | 6 | +4.20 / +2.40 | Current maximum; includes one 7/20 OOM-partial allocation |
| Rover | S1 fixed normal MCTS/off | 4.8 / 5.0 / 5.0 | 4.0 | 7 | +1.0 / -2.0 | Modest gain, still below paper |
| Counters | terminal-led S2 narrow/off* | 34.9 / 37.8 / 38.4 | 21.5 | 17 | +16.9 / +21.4 | Largest result; versus its own S2 policy37.9 the 6h change is only +.5 and nonsignificant |

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

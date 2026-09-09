# Advisor-ready RQ and experiment snapshot — 2026-09-09T22:08:19+03:00

This report uses the fresh live scheduler capture only for jobs that were live in the prior report. Completed results come from locally cached authoritative ledgers. `>=` always denotes an active-run lower bound.

## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 9 | 38 | 336 GiB |
| Ordinary pending | 11 | 44 | 220 GiB |
| Slurm-held | 0 | 0 | 0 GiB |

| Live experiment | Jobs | CPU / RAM | Latest evidence | Realistic timing |
|---|---:|---:|---|---|
| MPrime validation adequacy Phase B | 6 running + 11 dependency-pending | 24 / 120 GiB running; 44 / 220 GiB pending | 2,222/2,260 checkpoint-replicates (98.3%); the last structured lineage rebuild had 50/60 complete at 2,219 results | Original tasks have 59m hard bound; idempotent tail 21157796 starts afterwards and computes only missing points |
| Adaptive KL-control TPP/off | 2 running | 12 / 96 GiB | Outlier through Stage-2 epoch34; stable control through epoch84; zero coefficient changes; epoch-0 test scores are 10/20 and 20/20 | Stable control roughly 4h to epoch100; outlier roughly 13h at current pace; either may stop earlier |
| FO/off PW70 exact-instance correction | 1 running | 2 / 120 GiB | Correct evaluator slot for `instance_15.pddl`; the earlier 40-GiB job targeted slot15=`instance_16.pddl` and OOMed | At most about 6h instance work plus validation; seed remains 7/20 until this run proves an eighth success |

The only queued work is shown above. Design-held experiments appear later and do not occupy Slurm.

## Compact advisor story

1. **Start from the original split.** Numeric ASNets were already near-perfect
   on Delivery, TPP and Zenotravel, but remained imperfect on six domains. The
   first question was whether MCTS-generated Stage-2 data could improve the
   learned policy itself without destroying the three stable domains.
2. **Policy refinement is not a universal win.** Across the five completed
   mainstream domains, Stage-2 policy changes are mostly neutral or negative;
   no VH-on Stage-2 policy gain survives correction. PRESERVE-3 does retain
   near-ceiling means, but exposes one important TPP/off catastrophic seed.
3. **Inference-time search is the concrete success.** Fixed MCTS produces large,
   Holm-significant gains for Drone/VH-on and FO Counters/off, including at a
   30-minute counterfactual cutoff. The best demonstrated results beat the
   published means in Drone, FO Counters and Counters.
4. **Efficiency can be recovered selectively.** PW70 is significant over policy
   in both FO Counters modes and reaches fixed-search parity, while Rover reaches
   parity without a significant gain. The original Drone PW schedule was faster
   but lost too much coverage, so PW is a domain/configuration choice rather than
   a universal replacement.
5. **The remaining methodological blocker is checkpoint selection.** MPrime's
   original validator weakly ranks Stage-1 checkpoints and gives every Stage-2
   checkpoint 30/30. Phase B is now 98.2% complete on two harder, independently
   frozen sets. The live completion count is now 98.3%. This is the only experiment likely to change the main thesis
   story before writing.

The optimistic but defensible headline is: **MCTS does not reliably improve the
policy through a second training stage, but it does materially improve planning
coverage at inference time in the domains that need it most, and progressive
widening can retain those gains efficiently in FO Counters.**

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

#### Stage-2 policy versus fixed MCTS

Every row now has ten matched seeds. Each cutoff cell is `MCTS mean; paired
change [95% CI]; raw/Holm p`; Holm is computed across all 18 complete
domain/VH/branch cells separately at each cutoff.

| Domain/VH | Branch | Search | Policy | 30m: MCTS; change [95% CI]; raw/Holm p | 2h: MCTS; change [95% CI]; raw/Holm p | 6h: MCTS; change [95% CI]; raw/Holm p |
|---|---|---|---:|---|---|---|
| BG/off | terminal | narrow5/20 | 16.3 | 11.1; -5.2 [-6.14,-4.26]; .002/.035 | 13.1; -3.2 [-4.01,-2.39]; .002/.035 | 14.5; -1.8 [-3.22,-.38]; .031/.406 |
| BG/on | terminal | narrow5/20 | 11.6 | 9.2; -2.4 [-3.17,-1.63]; .002/.035 | 9.5; -2.1 [-3.02,-1.18]; .008/.109 | 10.5; -1.1 [-2.08,-.12]; .063/.750 |
| Counters/off | terminal | narrow5/20 | 37.9 | 34.9; -3.0 [-7.54,1.54]; .227/1 | 37.8; -.1 [-1.59,1.39]; 1/1 | 38.4; +.5 [-.34,1.34]; .313/1 |
| Counters/on | terminal | narrow5/20 | 16.8 | 19.9; +3.1 [-3.43,9.63]; .340/1 | 22.0; +5.2 [1.02,9.38]; .020/.234 | 22.4; +5.6 [1.84,9.36]; .008/.117 |
| Drone/off | terminal | normal20/70 | 7.8 | 9.7; +1.9 [.49,3.31]; .025/.279 | 9.8; +2.0 [.53,3.47]; .025/.279 | 9.9; +2.1 [.65,3.55]; .020/.273 |
| Drone/on | terminal | normal20/70 | 6.5 | 12.9; +6.4 [5.32,7.48]; .002/.035 | 13.1; +6.6 [5.47,7.73]; .002/.035 | 13.1; +6.6 [5.47,7.73]; .002/.035 |
| FO/off | terminal | normal20/70 | 2.8 | 5.3; +2.5 [1.73,3.27]; .002/.035 | 5.3; +2.5 [1.73,3.27]; .002/.035 | 5.3; +2.5 [1.73,3.27]; .002/.035 |
| FO/on | terminal | normal20/70 | 3.8 | 4.4; +.6 [-.09,1.29]; .156/1 | 4.4; +.6 [-.09,1.29]; .156/1 | 4.4; +.6 [-.09,1.29]; .156/1 |
| Rover/off | terminal | normal20/70 | 3.8 | 4.2; +.4 [.03,.77]; .125/1 | 4.2; +.4 [.03,.77]; .125/1 | 4.2; +.4 [.03,.77]; .125/1 |
| Rover/on | terminal | normal20/70 | 4.0 | 4.5; +.5 [.12,.88]; .063/.625 | 4.5; +.5 [.12,.88]; .063/.625 | 4.5; +.5 [.12,.88]; .063/.750 |
| BG/off | validation | narrow5/20 | 16.0 | 11.4; -4.6 [-5.37,-3.83]; .002/.035 | 15.0; -1.0 [-2.01,.01]; .094/.844 | 15.7; -.3 [-.89,.29]; .500/1 |
| BG/on | validation | narrow5/20 | 12.8 | 10.1; -2.7 [-3.38,-2.02]; .002/.035 | 10.6; -2.2 [-3.26,-1.14]; .008/.109 | 12.6; -.2 [-1.08,.68]; .813/1 |
| Counters/off | validation | narrow5/20 | 36.9 | 34.9; -2.0 [-6.93,2.93]; .500/1 | 36.7; -.2 [-2.87,2.47]; .969/1 | 36.7; -.2 [-2.87,2.47]; .969/1 |
| Counters/on | validation | narrow5/20 | 21.8 | 22.6; +.8 [-8.28,9.88]; .869/1 | 26.4; +4.6 [-1.83,11.03]; .152/1 | 27.1; +5.3 [-.69,11.29]; .082/.820 |
| Drone/off | validation | normal20/70 | 6.7 | 7.5; +.8 [-.58,2.18]; .359/1 | 7.7; +1.0 [-.43,2.43]; .195/1 | 7.7; +1.0 [-.43,2.43]; .195/1 |
| Drone/on | validation | normal20/70 | 5.0 | 10.9; +5.9 [4.23,7.57]; .002/.035 | 11.2; +6.2 [4.33,8.07]; .002/.035 | 11.2; +6.2 [4.33,8.07]; .002/.035 |
| Rover/off | validation | normal20/70 | 4.0 | 4.5; +.5 [-.01,1.01]; .125/1 | 4.5; +.5 [-.01,1.01]; .125/1 | 4.5; +.5 [-.01,1.01]; .125/1 |
| Rover/on | validation | normal20/70 | 3.9 | 4.4; +.5 [-.20,1.20]; .250/1 | 4.5; +.6 [-.09,1.29]; .156/1 | 4.5; +.6 [-.09,1.29]; .156/1 |

**Conclusion:** the complete family retains two clean Stage-2 MCTS wins:
Drone/VH-on in both branches and FO/off terminal-led at every cutoff (all Holm
p=.035). Counters/on has a large positive mean but high seed variance; Block
Grouping is negative at short budgets; Rover and FO/on are small/neutral.

### RQ3 — does the value head improve Stage-2 refinement?







| Domain | Validation-led DiD [95% CI]; Holm p | Terminal-led DiD [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -2.8 [-4.61,-.99]; .088 | -3.4 [-5.13,-1.67]; .020 | Terminal-led VH significantly worsens refinement |
| Drone | -.9 [-2.88,1.08]; 1.0 | -1.1 [-3.57,1.37]; .836 | No reliable effect |
| FO | +.7 [-.26,1.66]; .813 | +1.3 [.08,2.52]; .188 | Positive raw tendency only |
| Rover | +.1 [-.31,.51]; 1.0 | -.1 [-.51,.31]; 1.0 | No effect |
| Counters | -1.2 [-25.87,23.47]; 1.0 | -16.8 [-29.64,-3.96]; .078 | Large negative tendency, high variance |

Only terminal-led Block Grouping is significant after correction.

The difference-in-differences above does not conceal a positive VH-on training
result. Looking only at VH-on policy change gives:

| Domain | Validation-led VH-on change [95% CI]; Holm p | Terminal-led VH-on change [95% CI]; Holm p |
|---|---|---|
| BG | -3.1 [-5.16,-1.04]; .156 | -3.4 [-5.25,-1.55]; .078 |
| Drone | -.1 [-1.34,1.14]; 1.0 | -.7 [-1.77,.37]; 1.0 |
| FO | -.6 [-1.50,.30]; 1.0 | +.5 [-.41,1.41]; 1.0 |
| Rover | +.1 [-.31,.51]; 1.0 | -.1 [-.33,.13]; 1.0 |
| Counters | +3.2 [-6.15,12.55]; 1.0 | -.4 [-11.44,10.64]; 1.0 |

Thus, VH-on Stage-2 training has no Holm-significant policy-only benefit in any
domain. The defensible positive value-head story is instead inference-time:
Drone/VH-on receives the largest, most reproducible MCTS gains.

### RQ4 — does the value head alter the benefit of inference-time MCTS?

The Stage-2 table under RQ2 is also the authoritative RQ4 evidence. The strongest reproducible pattern is Drone/on: large Holm-significant gains in both validation- and terminal-led branches. FO/off is also significant in the terminal-led branch. Block Grouping is negative, Rover and FO/on are small/neutral, Counters is variable, and MPrime awaits Phase B checkpoint freezing.

## Advisor figures — before and after independent review

The advisor package is `experiment_tracking/advisor_meeting_20260910/`.
`before_review/` preserves the first version unchanged. A same-model reviewer
received only the general project description and those five PNGs; its comments
are stored in `independent_plot_review.md`. `after_review/` fixes the identified
ambiguities without changing the underlying result rows.

| Revised view | File | Question answered |
|---|---|---|
| Best observed domain scorecard | `after_review/01_domain_scorecard.png` | Where did we beat Stage 1 and the published mean? |
| Full two-stage learning dynamics | `after_review/02_two_stage_learning_dynamics.png` | How do complete Stage-1 and validation-led Stage-2 policy curves differ? |
| Stage-1 inference-time MCTS | `after_review/03a_stage1_mcts_cutoff_forest.png` | Which policy cells benefit at 30m, 2h and 6h? |
| Stage-2 inference-time MCTS | `after_review/03b_stage2_mcts_cutoff_forest.png` | Does search recover or improve refined policies? |
| PRESERVE-3 validation-led robustness | `after_review/04_preserve3_validation_seed_robustness.png` | Are near-perfect policies preserved seed-by-seed? |
| MPrime validation failure | `after_review/05_mprime_validation_problem.png` | What does the saturated validator cost, and why is Phase B necessary? |

The full Stage-1 and Stage-2 curve data remain locally re-plottable from
`advisor_meeting_20260910/before_review/02_two_stage_learning_dynamics.csv`.
That CSV is built from 1,980 cached Stage-1 policy rows plus the cached Stage-2
aggregates and retains the direct job/log provenance through the underlying
learning-curve ledgers; no cluster reread is needed to restyle it.


## Experiments

### Live and recently completed experiment updates

#### MPRIME-VAL-ADEQUACY Phase B

Two independently frozen harder validation replicates are being evaluated on
all 1,130 saved checkpoints: **2,222/2,260 results (98.3%)**. The last full
structured lineage rebuild, at 2,219 results, had all 60 lineages represented
and 50 lineages complete. Six original tasks remain active with about 59 minutes
to their allocation limits. Idempotent tail array 21157796 is
already dependency-gated behind them and covers the eleven lineages that were
incomplete when it was submitted; it will skip any point completed meanwhile.
No new training is involved. The next decision is whether both frozen
replicates agree on checkpoint and anchor rankings. Only then should MPrime
checkpoints/coefficients be frozen and the Stage-2 branches released; the
eventual 40 MCTS comparisons remain deliberately unsubmitted.

#### ANCHOR-KL-CONTROL — adaptive KL screen

Purpose: test whether controlling actual policy drift prevents TPP/off seed
1972442430 from collapsing after the first Stage-2 update without damaging a
stable control seed.

| Role | Job | Epochs recorded | Validation first -> latest (range) | Max post-update KL | Coefficient | Adjustments |
|---|---:|---:|---|---:|---:|---:|
| Catastrophic outlier | 21144388 | 35 (epoch 34; cumulative 49) | latest 29/30 (range 18–30/30) | 0.057600 | 3.0 | 0 |
| Stable control | 21144389 | 85 (epoch 84; cumulative 87) | 30/30 throughout | 0.021586 | 3.0 | 0 |

The completed first-update test-policy evaluations are **10/20 for the
catastrophic seed and 20/20 for the stable control**. The adaptive run therefore
did not repair the original first-update 20->10 collapse.

The reason is now concrete: the intervention has been a **controller no-op**.
Coefficient 3 has never changed. Three was chosen as the floor because it is the
already frozen constant-anchor baseline for TPP/off; the adaptive arm was
designed never to weaken that baseline. The KL target .1143 is the stable
control's predeclared first-Stage-2-update KL, not a value tuned on the bad
seed. The standard PPO-style 1.5x tolerance gives a target band of approximately
[.0762,.1715], with factor-two coefficient changes outside it. This band is
intentionally not tighter because minibatch KL is noisy and a narrow band would
encourage oscillation. Both new runs stayed below the upper trigger, while the
floor prevented downward adjustment.

The outlier's automatic requeue did **not** redo Stage-1 training and did not
mean "resume at epoch 0." It loaded the exact Stage-1 validation-selected source
(Stage-1 epoch 15) and began a fresh Stage-2 output directory. The previous 58
adaptive Stage-2 checkpoints remain intact. The wrapper failed to discover that
continuation because an explicit Stage-1 `--train-from` source was already
present. A manual continuation is technically possible, but interrupting the
healthy replacement now would lose more work; the correct wrapper fix belongs
before any larger follow-up.

#### Single live MCTS classification

| Experiment | Current result | What remains | Consequence |
|---|---|---|---|
| FO/off PW70 seed 2082152039 | Declared seed remains 7/20 | Correct `instance_15.pddl` recovery job 21157787 is running | Success would move this seed to 8/20 and the ten-seed mean from 8.40 to 8.50; failure leaves all published statistics unchanged |

The prior 40-GiB attempt 21154686 did not test the intended instance: evaluator
slot 15 maps to `instance_16.pddl` because FO's test list begins at instance 2.
It then OOMed. The corrected job uses slot 14 and 120 GiB. The old 40-GiB choice
was exactly one third of the original 120-GiB/three-worker request, because one
worker evaluates one independent instance; the failure showed that proportional
scaling was insufficient for this hard case, hence the full-memory correction.

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
7/20* after 19 classified instances; the single unclassified instance counts as
a failure. A complete-allocation sensitivity analysis excluding that entire
OOM-labelled seed remains available (n=9, mean 8.56), but it is not the headline
mean. Every
statistic below is paired; Holm is across the four cells at the same cutoff
and comparator.

| Cell | n | Policy | PW70 30m / 2h / 6h | PW-policy at 6h [95% CI]; raw/Holm p | PW-fixed 20/70 at 6h [95% CI]; raw/Holm p | Conclusion |
|---|---:|---:|---:|---|---|---|
| FO/off | 10* | 4.20 | 8.40 / 8.40 / 8.40 | +4.20 [3.26,5.14]; .002/.008 | +.60 [-.71,1.91]; .422/1.0 | Significant policy gain; fixed parity; one OOM-partial seed |
| FO/on | 10 | 3.70 | 7.30 / 7.30 / 7.30 | +3.60 [2.70,4.50]; .002/.008 | +1.60 [.52,2.68]; .023/.094 | Significant over policy; raw gain over fixed |
| Rover/off | 10 | 4.00 | 4.70 / 4.70 / 4.70 | +.70 [.02,1.38]; .125/.250 | -.30 [-1.37,.77]; .688/1.0 | Fixed-search parity, not significant |
| Rover/on | 10 | 3.80 | 4.50 / 4.60 / 4.60 | +.80 [-.08,1.68]; .125/.250 | +.20 [-.80,1.20]; .828/1.0 | Fixed-search parity, not significant |

`*` FO/off seed 2082152039 is the 7/20 OOM-partial declared result: 19/20
instances were classified and the one unclassified instance counts unsuccessful.
The 8.56/20 sensitivity estimate excludes that entire seed, not merely its one
unclassified instance.

FO/off and FO/on are significant versus policy at 30m, 2h and 6h (Holm
p=.008 at each cutoff). FO/on exceeds fixed search at 30m after Holm correction
(p=.031), but its 2h/6h fixed-search differences do not survive correction.
FO/off is unchanged across cutoffs. Rover remains nonsignificant at every
cutoff. Full cutoff CIs/p-values are in `pw70_ten_seed_statistics_latest.csv`.

**Conclusion:** PW70 is a strong FO Counters result and a Rover parity result.
It is not a universal replacement for fixed search.

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
| MCTS-PW70-CROSS-DOMAIN | completed or inactive | What changes when the accidental 20-simulation PW screen is rerun with the intended 70 simulations? | Corrected screen is terminal; no active Slurm work remains. |
| MCTS-PW70-CONFIRMATORY | completed or inactive | Do promising two-seed cells replicate with enough matched seeds for paired intervals and tests? | All five seed cells terminal; expand FO/Rover to ten with separately registered20 jobs; Counters PW70 does not match policy consistently. |
| MCTS-PW-30M | held design | Does PW retain most MCTS benefit under a paper-style 30-minute instance budget? | Post-hoc cutoffs suffice for solutions-within-budget coverage; fresh runs needed only for whole-job resource/time validation or missing/censored timing evidence. |
| MAIN-VAL-S2-MCTS | completed or inactive | Does MCTS improve validation-selected Stage2 policies? | All twenty exact endpoints terminal with VAL evidence |
| MCTS-LEGACY-ROVER | completed or inactive | Complete the already-submitted Rover endpoint evidence before FO Counters | Off policy3.8 to MCTS4.2 CI[0.03 0.77] rawp.125; on policy4.0 to MCTS4.5 CI[0.12 0.88] rawp.0625; OOM allocations retained as fixed-budget outcomes |
| MCTS-LEGACY-FO | completed or inactive | Complete the fifth mainstream Stage2 policy/MCTS domain comparison | Terminal-led VH-on is now ten seeds: policy3.8 to MCTS4.4 at every cutoff, change+.6 CI[-.09,1.29], raw p=.156, Holm p=1.0. |
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
| ANCHOR-KL-CONTROL | live | Can literature-grounded nonconstant KL control prevent first-update collapse without blocking later improvement? | Jobs21144388/21144389 reached Stage2 epochs34/84. First-update test scores remain10/20 and20/20; coefficient remains3 and adjustments remain zero, so the adaptive mechanism has not activated. |
| MPRIME-VAL-ADEQUACY | live | Does validation rank checkpoints and anchors reliably without saturating? | 2222/2260 checkpoint-replicates complete (98.3%); the last structured rebuild at2219 results had50/60 lineages complete; six tasks active and an idempotent11-lineage tail is dependency-pending. Compare replicate rankings, then freeze checkpoints/anchors. |
| MCTS-PW70-TEN-SEED | completed or inactive | Do five-seed PW70 findings survive all ten original seeds? | All four cells contain ten declared-budget seeds; FO/off includes one explicitly starred 7/20 OOM-partial seed and has mean8.4. |
| TPP-CATASTROPHIC-MCTS | completed or inactive | Can search recover the eleven policy failures caused by seed-specific Stage2 catastrophic forgetting? | Policy9/20; fixed narrow4; PW20 5/10/10; fixed normal4; PW70 4/5/7 at30m/2h/6h. Only PW20 recovers one net policy success. |
| ROVER-MCTS-INTERRUP-REC | completed or inactive | Classify only instances omitted by scheduler or OOM interruption | All29 formerly unclassified instances reached the declared six-hour timeout; Rover aggregates are unchanged. |

### Best demonstrated result by domain

| Domain | Best current configuration | 30m / 2h / 6h or policy | Matched S1 policy | Paper | Delta vs S1 / paper | Conclusion |
|---|---|---:|---:|---:|---:|---|
| Delivery | S1 selected policy, off | 19.8 / 19.8 / 19.8 | 19.8 | 20 | 0 / -.2 | S2 preserves but does not improve the best S1 result |
| TPP | S1 policy | 20 / 20 / 20 | 20 | 20 | 0 / 0 | Perfect; one S2/off seed catastrophically forgets |
| Zenotravel | S1 and multiple S2 policy cells | 20 / 20 / 20 | 20 | 17 | 0 / +3 | Solved and preserved |
| MPrime | validation-led S2 policy, provisional | 15.2 / 15.2 / 15.2 | 15.0 | 19 | +.2 / -3.8 | Phase B is 98.3% complete and may change checkpoint selection |
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
- MPrime live marker count: `mprime_phase_b_live_count_20260909.csv`; structured per-lineage progress: `mprime_validation_phase_b_progress_latest.csv`.

Aggregate CSVs either carry direct log paths or point through their
`row_level_provenance`/`results_file` companion to job-, checkpoint-, training-
and evaluation-log rows.

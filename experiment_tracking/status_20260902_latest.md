# Experiment status — 2 September 2026, 14:26 IDT

This is the current authoritative status report. Static results come from the
CSV ledgers linked in each section. Live state comes from
`cluster_workload_latest.csv`, `live_training_progress_latest.csv`, and
`live_mcts_progress_latest.csv`. Every result-bearing CSV is covered by the
provenance audit and either contains direct source-log columns or names a
row-level companion ledger.

## Actions completed in this refresh

- Replaced the single validation-led MPrime policy controller with independent
  VH-off job 20863175 and VH-on job 20863176. The terminal-led controller had
  already completed and submitted 420 policy evaluations.
- Submitted the approved six-job Counters divergence-recovery experiment as
  jobs 20863184--20863189 at Nice=10000. Each of the three exact snapshots has
  separate PW20 and PW70 arms; the original PW screen remains intact.
- Added a two-step remaining-horizon regression test. With remaining horizon
  two and twenty simulations it records a cutoff at search depth two.
- Produced a complete sixty-row stable-domain seed ledger, paired statistics,
  and a detailed TPP/off training/outcome audit.
- Updated the experiment registry, coverage plan, live workload, training and
  MCTS ledgers, and result-provenance index.

## Cluster workload

| State | Jobs | Requested CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 89 | 658 | 6,140 GiB |
| Pending | 432 | 4,200 | 10,764 GiB |
| Held | 0 | 0 | 0 GiB |

| Experiment | Running | Pending | CPUs running / pending | RAM running / pending |
|---|---:|---:|---:|---:|
| Counters Horizon aware/unaware | 8 | 0 | 48 / 0 | 960 / 0 GiB |
| FO Counters terminal-led Stage-2 MCTS | 20 | 0 | 120 / 0 | 2,400 / 0 GiB |
| MPrime validation-led Stage-2 training | 6 | 0 | 36 / 0 | 288 / 0 GiB |
| MPrime validation-led policy controllers | 0 | 2 dependency | 0 / 4 | 0 / 2 GiB |
| MPrime terminal-led policy evaluation | 14 | 406 | 140 / 4,060 | 280 / 8,120 GiB |
| PRESERVE-3 terminal-led Stage-2 training | 14 | 0 | 84 / 0 | 672 / 0 GiB |
| PRESERVE-3 policy evaluation | 17 | 0 | 170 / 0 | 340 / 0 GiB |
| PRESERVE-3 policy controllers | 0 | 2 dependency | 0 / 4 | 0 / 2 GiB |
| PW70 cross-domain correction | 8 | 0 | 48 / 0 | 960 / 0 GiB |
| PW70 five-seed confirmation | 2 | 16 low priority | 12 / 96 | 240 / 1,920 GiB |
| Counters exact-snapshot PW recovery | 0 | 6 low priority | 0 / 36 | 0 / 720 GiB |

No job is scientifically or operationally held inside Slurm. The 24 low-Nice
PW jobs are submitted and eligible, but ordinary-priority work ranks ahead of
them. The large MPrime policy queue is ordinary resource-pending work.

## Live training and policy pipeline

| Experiment/cell | Terminal | Running | Policy work | Current training progress | Realistic estimate |
|---|---:|---:|---|---|---|
| MPrime validation-led Stage 2, VH-off | 8/10 | 2 | controller 20863175 depends only on this cell | epochs 96, 97 | about 1--2 h |
| MPrime validation-led Stage 2, VH-on | 6/10 | 4 | controller 20863176 depends only on this cell | epochs 88--92 | about 2--5 h |
| MPrime terminal-led Stage 2 | 20/20 | 0 | 14 running, 406 pending of 420 | training complete | policy jobs drain as memory becomes available |
| PRESERVE-3 Delivery | 20/20 | 0 | rolling curve/endpoint evaluation active | training complete | active policy jobs generally minutes each |
| PRESERVE-3 TPP, VH-off | 5/10 | 5 | cell controller installed | epochs 59, 82, 85, 97, 97 | two near-finished jobs about 1--2 h; others roughly 8--25 h |
| PRESERVE-3 TPP, VH-on | 1/10 | 9 | cell controller installed | epochs 55--98 | epoch 98 about 1 h; middle jobs roughly 5--22 h; epoch 55 may hit the 72 h limit |
| PRESERVE-3 Zenotravel | 20/20 | 0 | complete | training and policy compute complete | done |

The policy controllers are cell-specific: VH-off cannot wait for VH-on or vice
versa. No newly available policy endpoint is currently absent from a manifest;
the active controllers and 420 live MPrime policy jobs are the mechanism that
materializes every-five learning curves plus selected/final endpoints.

## PRESERVE-4 validation-led policy result — complete

All ten seeds are included. The two tuning seeds are displayed separately from
the eight held-out confirmation seeds, but the first Stage-2 column is always
the complete ten-seed result. Statistics use paired seed differences.

| Domain/VH | S1 selected | S2 selected all 10 | Held-out 8 | Tuning 2 | Change, 95% CI | Raw p | Holm p |
|---|---:|---:|---:|---:|---|---:|---:|
| Delivery/off | 19.8 | 19.6 | 19.5 | 20.0 | -0.2 [-1.01, 0.61] | 1.000 | 1.000 |
| Delivery/on | 19.2 | 19.6 | 19.5 | 20.0 | +0.4 [-0.37, 1.17] | .500 | 1.000 |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59, 1.39] | 1.000 | 1.000 |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -0.5 [-1.63, 0.63] | 1.000 | 1.000 |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0.0 [0.0, 0.0] | 1.000 | 1.000 |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -0.1 [-0.33, 0.13] | 1.000 | 1.000 |

`*` TPP/off is not a broad 1.1-plan decline: nine seeds are 20/20 and seed
1972442430 is 9/20. That job used the same 100 Stage-2 epochs as its peers and
did not train longer in update count. It took 30.17 h versus a 16.06 h peer
mean, emitted six worker-timeout warnings versus two for every peer, had higher
mean total loss (1.891 versus 1.612), higher policy loss (1.273 versus 1.203),
and about twice the anchor KL (0.1059 versus 0.0516). Test coverage was already
10/20 at epoch 0, peaked at 13/20 at epoch 5, and the validation-selected epoch
12 scored 9/20. Thus validation misranking cost roughly four plans relative to
the observed test-best checkpoint, but the severe collapse existed before that
selection. Source: `tpp_stage2_vh_off_regression_audit_20260902.md` and the
training-comparison CSV.

## Stage-1 validation-selected policy versus MCTS — complete

The cutoff columns are deterministic reconstructions from recorded
per-instance runtimes. Six hours is the declared result. Counters and Block
Grouping use the intended narrow 5/20 search; normal Counters 20/70 is retained
only as a separately labelled sensitivity arm.

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

## Stage-2 policy versus MCTS — exhaustive branch matrix

Both Stage-2 branches are explicit. A result in one branch does not substitute
for the other branch's checkpoint. `n` is the number of exact seed-level MCTS
results found or currently running.

| Domain/VH | Branch | Search | n/10 | Policy | MCTS 30m / 2h / 6h | 6h change, 95% CI; raw p | State/reason |
|---|---|---|---:|---:|---|---|---|
| Block Grouping/off | validation | narrow 5/20 | 0 | 16.0 | not run | not available | August release never included this cell |
| Block Grouping/on | validation | narrow 5/20 | 8 | 12.8 | partial result only | not final | only jobs 20559482--89 were released |
| Block Grouping/off | terminal | narrow 5/20 | 10 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22, -0.38]; .031 | complete |
| Block Grouping/on | terminal | narrow 5/20 | 9 exact logs | 11.6 | 4.4 / 8.7 / 9.4 in prior ten-score aggregate | provenance conflict | seed 2082152039 lacks an exact canonical log |
| Drone/off | validation | normal 20/70 | 10 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [-0.43, 2.43]; .195 | complete |
| Drone/on | validation | normal 20/70 | 10 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33, 8.07]; .002 | complete |
| Drone/off | terminal | normal 20/70 | 10 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [0.65, 3.55]; .020 | complete |
| Drone/on | terminal | normal 20/70 | 10 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [5.47, 7.73]; .002 | complete |
| FO Counters/off | validation | normal 20/70 | 6 | 2.9 | partial historical evidence | not final | manually released exploratory subset |
| FO Counters/on | validation | normal 20/70 | 9 | 3.1 | partial historical evidence | not final | manually released exploratory subset |
| FO Counters/off | terminal | normal 20/70 | 10 running | 2.8 | >=5.3 / >=6.3 retained / >=6.3 retained | lower bound only | current logs were truncated by outage requeue; prior 63-success evidence retained |
| FO Counters/on | terminal | normal 20/70 | 10 running | 3.8 | >=4.4 / >=4.4 / >=4.4 | lower bound only | live |
| Rover/off | validation | normal 20/70 | 1 | 4.0 | partial historical evidence | not final | only job 20559495 was released |
| Rover/on | validation | normal 20/70 | 0 | 4.22 | not run | not available | validation branch was never released |
| Rover/off | terminal | normal 20/70 | 10 | 3.8 | 4.2 / 4.2 / 4.2 | +0.4 [0.03, 0.77]; .125 | complete |
| Rover/on | terminal | normal 20/70 | 10 | 4.0 | 4.5 / 4.5 / 4.5 | +0.5 [0.12, 0.88]; .063 | complete |
| Counters/off | validation | narrow 5/20 | 10 | 36.9 | 34.9 / 36.7 / 36.7 | -0.2 [-2.87, 2.47]; .969 | complete |
| Counters/on | validation | narrow 5/20 | 10 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-0.70, 11.30]; .082 | complete |
| Counters/off | terminal | narrow 5/20 | 0 | 37.9 | not materialized | not available | no manifest/submission/log ever existed |
| Counters/on | terminal | narrow 5/20 | 0 | 16.8 | not materialized | not available | no manifest/submission/log ever existed |

The 36 validation-led gaps are Block Grouping 12, FO Counters 5 and Rover 19.
A fresh hard-two-hour campaign cannot honestly fill a declared six-hour matrix;
it would be a different fixed-budget experiment and might still require later
six-hour reruns. Therefore these gaps were documented, not submitted as 2 h.
Comparable medians are 12.5 h for Block Grouping, 48.3 h for FO Counters and
30.1 h for Rover. Rover's existing terminal results are identical at 2 h and
6 h, FO's Stage-1 analogue is unchanged, but Block Grouping loses about 2.2
plans/off and 0.7/on when capped at 2 h.

## Progressive widening — complete and live evidence

The table keeps PW20 and PW70 separate. Fixed search is labelled narrow 5/20
or normal 20/70. Values prefixed `>=` are current lower bounds.

| Cell | Policy | Fixed 30m / 2h / 6h | PW20 30m / 2h / 6h | PW70 30m / 2h / 6h |
|---|---:|---|---|---|
| BG S1/off | 16.5 | narrow 11.0 / 13.5 / 15.0 | 11.5 / 15.0 / 15.0 | 10.5 / 11.5 / 13.0 final |
| BG S1/on | 17.0 | narrow 11.5 / 13.5 / 17.0 | 12.5 / 17.0 / 18.0 | 9.0 / 11.5 / 13.5 final |
| Counters S1/off | 18.0 | narrow 21.5 / 21.5 / 21.5 | 20.5 / 20.5 / 20.5 | >=21.0 / >=21.0 / >=21.0 |
| Counters S1/on | 5.0 | narrow 13.5 / 13.5 / 13.5 | 12.5 / 12.5 / 12.5 | >=12.0 / >=12.5 / >=12.5 |
| Counters S2/off | 49.0 | narrow >=37.0 / >=41.5 / 44.5 | 46.0 / 49.5 / 49.5 | >=34.0 / >=39.5 / >=43.5 |
| Counters S2/on | 5.0 | narrow 16.5 / 17.5 / 17.5 | 15.0 / 15.0 / 15.0 | >=13.5 / >=14.0 / >=14.0 |
| FO Counters S1/off | 3.5 | normal 9.0 / 9.5 / 9.5 | not run | 9.5 / 9.5 / 9.5 final |
| FO Counters S1/on | 3.5 | normal 6.0 / 7.5 / 7.5 | not run | 8.0 / 8.0 / 8.0 final |
| Rover S1/off | 4.0 | normal 4.0 / 4.5 / 4.5 | not run | 5.0 / 5.0 / 5.0 final |
| Rover S1/on | 4.0 | normal 4.5 / 4.5 / 4.5 | not run | 5.5 / 5.5 / 5.5 final |

The PW70 correction has 205 classified instances and current totals
>=161/174/182 at 30 m/2 h/6 h. The eight Counters jobs have run about 22--36 h;
the hard remainder is 36--50 h, while realistic completion remains instance
dependent. The 18-job confirmation is exactly six cells times three additional
seeds: two Counters S2/off jobs run and sixteen wait at low priority, extending
each two-seed screen cell to five matched seeds.

The six exact-snapshot Counters recovery jobs are additional to this screen.
They test seeds 534933607, 923500475 and 2082152039 under both PW20 and PW70,
and will compare the same snapshot against policy, normal 20/70 and fixed narrow
5/20 at all three cutoffs.

## Counters Horizon efficacy

Eight normal-20/70 jobs are running: two seeds, two VH modes, aware and unaware.
Across all arms, 138 instances are classified and the aggregate lower bounds
are 131/135/138 at 30 m/2 h/6 h. Each current aware/unaware pair ties at the
6 h cutoff. No emitted summary reports a nonzero cutoff.

The zero count is plausible for the currently terminal evidence: maximum
successful plan length is 1,105, far from 10,000. The user's depth argument is
nevertheless a valid audit trigger near action 9,998. The minimum largest root
visit count is `ceil(simulations / width)`—four for both 20/70 and 5/20—not 8
or 28, and visits can terminate or spread before a grandchild. Even so, with a
remaining horizon of two, repeated simulations should eventually hit depth two.
The new regression test proves that the current counter records exactly that
case. If an aware run reaches 10,000 external actions and still reports a
cumulative zero, it will be treated as a propagation/instrumentation defect,
not dismissed as normal behavior.

## Story holes and defensible closure

| Hole | Why it matters | Defensible closure |
|---|---|---|
| 57 exact Stage-2 MCTS branch gaps, including 20 live FO jobs | RQ2/RQ4 need branch-matched networks | finish FO; jointly decide whether the remaining 37 six-hour jobs justify their cost; do not disguise a 2 h campaign as the 6 h result |
| TPP/off one-seed collapse | average hides bimodality and validation misranking | retain asterisk and seed audit; inspect replay/state distribution before any retraining |
| FO requeue truncated current stdout | current log alone no longer contains all pre-outage evidence | preserve the prior 63-success ledger as a lower bound and reconcile by instance identity at termination |
| PW screen starts with two seeds | good for screening, insufficient for robust CI/p-values | current 18-job expansion supplies five matched seeds in six promising cells |
| Horizon cutoff remains zero | the mechanism has not yet been activated by evidence | finish Counters; require a nonzero audited cutoff and cross-horizon state reuse before SAFE2 |
| MPrime policy results pending | sixth-domain RQ1/RQ2/RQ4 extension not yet testable | cell-specific controllers now release curves without cross-VH blocking |

## Held designs and frozen experiments

Design-held only; none occupies Slurm:

- `MCTS-PW-30M`: keep held until PW70 correction/confirmation selects cells.
- `MCTS-PW-PATHBATCH`: requires frozen nonstandard multi-expansion semantics.
- `MCTS-SAFE2`: held behind the nonzero-cutoff and cross-horizon-reuse gate.
- `ACT-HISTORY-ABLATION`: fresh Stage-1 training for Drone, Counters and TPP;
  intentionally held because it changes network dimensionality.
- `MCTS-RESOURCE`: two-worker/160-GiB FO Counters/Rover sensitivity.
- `STOP-ORIG`: original stopping-rule replication.
- `PUCT-EST`: PUCT/estimator causal sensitivity.

Completed experiments remain registered in `experiments.csv`, including
MAIN-VAL, MAIN-TERM, PRESERVE-4, MPrime corrected Stage 1, all anchor tuning,
fixed-width sensitivity, original PW, SAFE-PW Kmin3, SAFE-1, Drone Horizon,
SAFE-CONTEXT, Drone/Rover completed Stage-2 MCTS, Long Drone and ENHSP leaf
sensitivity. The determinism audit is retained in the registry but omitted from
routine reports as requested.

## Authoritative files

- `cluster_workload_latest.csv` and `cluster_workload_summary_latest.csv`
- `live_training_progress_latest.csv` and `live_mcts_progress_latest.csv`
- `four_domain_preservation/stable_domain_stage2_seed_pairs_20260902.csv`
- `four_domain_preservation/stable_domain_stage2_statistics_20260902.csv`
- `stage2_mcts_branch_coverage_20260902.csv`
- `stage2_mcts_gap_decisions_20260902.csv`
- `mcts_progressive_widening_cross_domain/counters_stage1_divergence_recovery_submissions.tsv`
- `result_provenance_audit_20260902.csv` and `result_provenance_index_20260902.csv`

# Experiment status — 3 September 2026, 10:21 IDT

This snapshot joins the live Slurm queue, current training/MCTS logs, and frozen
static ledgers. Live sources are `cluster_workload_latest.csv`,
`live_training_progress_latest.csv`, and `live_mcts_progress_latest.csv`.
Every static result cited here either has absolute row-level cluster-log paths or
is mapped to its row-level evidence in `result_provenance_index_20260902.csv`.

## Cluster workload

| State | Jobs | Requested CPU | Requested RAM |
|---|---:|---:|---:|
| Running | 53 | 318 | 6,072 GiB |
| Pending | 6 | 32 | 601 GiB |
| Held | 0 | 0 | 0 GiB |

| Experiment | Running | Pending | CPU running/pending | RAM running/pending |
|---|---:|---:|---:|---:|
| Counters Binding Horizon | 8 | 0 | 48/0 | 960/0 GiB |
| Counters PW divergence recovery | 6 | 0 | 36/0 | 720/0 GiB |
| FO Counters terminal-led S2 MCTS | 11 | 5 | 66/30 | 1,320/600 GiB |
| PRESERVE-3 terminal-led S2 training | 4 | 0 | 24/0 | 192/0 GiB |
| PRESERVE-3 policy controller | 0 | 1 | 0/2 | 0/1 GiB |
| PW70 confirmatory expansion | 18 | 0 | 108/0 | 2,160/0 GiB |
| PW70 two-seed correction | 6 | 0 | 36/0 | 720/0 GiB |

The five ordinary MCTS jobs are pending for `QOSMaxMemoryPerUser`; the one
controller is dependency-pending. No job is Slurm-held.

## Live training and policy pipeline

| Experiment/cell | Terminal | Running | Current point | Realistic estimate |
|---|---:|---:|---|---|
| PRESERVE-3 terminal-led TPP/off | 10/10 | 0 | training terminal; cell controller failed on continuation provenance | controller repair only; no retraining required |
| PRESERVE-3 terminal-led TPP/on | 6/10 | 4 | epochs 99, 98, 74, 99 | three about 1–3 h; epoch-74 lineage likely hits the 72 h limit in about 9.5 h |
| MPrime validation-led S2 | 20/20 | 0 | all policy endpoints complete and reconciled | complete, but validation adequacy is under audit |
| MPrime terminal-led S2 | 20/20 | 0 | all policy endpoints complete and reconciled | complete, but validation adequacy is under audit |
| PRESERVE-3 Delivery | 20/20 | 0 | all policy curves/endpoints complete | complete |
| PRESERVE-3 Zenotravel | 20/20 | 0 | all policy curves/endpoints complete | complete |

TPP/off continuation job 20834985 resumed from cumulative S2 epoch 84 and
finished at epoch 100. Its final checkpoint is continuation snapshot 15. Its
inherited validation-best checkpoint is parent job 20755752 snapshot 25
(cumulative S2 epoch 80). Controller 20859876 incorrectly required both to be
declared in the continuation log and failed before writing a manifest. The
scientific evidence exists; the materializer needs continuation-aware
provenance before the replacement controller is submitted.

## Mainstream policy results — completed

All cells contain ten paired seeds. `pH` is Holm-adjusted across the ten cells.

| Domain/VH | MAIN-VAL S1 selected -> S2 selected | change [95% CI]; raw/pH | MAIN-TERM S1 final -> S2 selected | change [95% CI]; raw/pH |
|---|---:|---|---:|---|
| Block Grouping/off | 16.3 -> 16.0 | -.3 [-1.20,.60]; .625/1 | 16.3 -> 16.3 | 0 [-.95,.95]; 1/1 |
| Block Grouping/on | 15.9 -> 12.8 | -3.1 [-5.16,-1.04]; .0156/.156 | 15.0 -> 11.6 | -3.4 [-5.25,-1.55]; .0078/.078 |
| Drone/off | 5.9 -> 6.7 | +.8 [-1.27,2.87]; .504/1 | 7.4 -> 7.8 | +.4 [-1.33,2.13]; .688/1 |
| Drone/on | 5.1 -> 5.0 | -.1 [-1.34,1.14]; 1/1 | 7.2 -> 6.5 | -.7 [-1.77,.37]; .250/1 |
| FO Counters/off | 4.2 -> 2.9 | -1.3 [-2.37,-.23]; .0469/.422 | 3.6 -> 2.8 | -.8 [-1.46,-.14]; .0547/.438 |
| FO Counters/on | 3.7 -> 3.1 | -.6 [-1.50,.30]; .250/1 | 3.3 -> 3.8 | +.5 [-.41,1.41]; .344/1 |
| Rover/off | 4.0 -> 4.0 | 0 [0,0]; 1/1 | 3.8 -> 3.8 | 0 [-.34,.34]; 1/1 |
| Rover/on | 3.8 -> 3.9 | +.1 [-.31,.51]; 1/1 | 4.1 -> 4.0 | -.1 [-.33,.13]; 1/1 |
| Counters/off | 32.5 -> 36.9 | +4.4 [-17.44,26.24]; .660/1 | 21.5 -> 37.9 | +16.4 [3.09,29.71]; .0273/.246 |
| Counters/on | 18.6 -> 21.8 | +3.2 [-6.15,12.55]; .563/1 | 17.2 -> 16.8 | -.4 [-11.44,10.64]; .961/1 |

No Stage-1-to-Stage-2 policy change survives ten-cell Holm correction.

## MPrime six-domain extension

The corrected validation set improved substantially over the old epoch-1-biased
set, but current evidence does not establish that it is adequate.

| Branch/VH | S1 source -> S2 selected | change [95% CI] | exact sign-flip p |
|---|---:|---|---:|
| Validation-led/off | 15.0 -> 15.2 | +.2 [-1.18,1.58] | .88 |
| Validation-led/on | 14.6 -> 15.2 | +.6 [-1.46,2.66] | .58 |
| Terminal-led/off | 13.5 -> 14.0 | +.5 [-1.29,2.29] | .672 |
| Terminal-led/on | 13.2 -> 14.5 | +1.3 [-.49,3.09] | .172 |

All twenty validation-led S2 lineages selected epoch 0 because validation was
already 30/30. For Stage 1, pooled validation/test Spearman correlation is only
.255/off and .252/on; only 5/10 off and 7/10 on lineages have positive
within-lineage correlation. Selection regret versus the retrospectively observed
test-best checkpoint averages 3.1 plans/off and 2.5/on. The current set is thus
weak for Stage-1 checkpoint ranking and saturated for Stage-2 selection.

The adequacy test is:

1. Use the existing 290 Stage-1 and 840 Stage-2 checkpoint records to measure
   Spearman/Kendall agreement, score saturation, plateau width, selected-epoch
   stability, and test regret. Test performance is diagnostic only.
2. Freeze two independent, structurally stratified IPC-scale validation
   replicates with a harder upper tail before viewing network scores.
3. Rescore existing checkpoints only: at most 1,130 validations grouped into 60
   lineage jobs. Require low saturation, agreement between replicates, stable
   selected epochs, and consistent anchor rankings.
4. Retrain only if the replacement passes those criteria and proves that the old
   early-stopping decision discarded useful later checkpoints.

## PRESERVE-3 validation-led — completed

| Domain/VH | S1 selected | S2 selected all 10 | Held-out 8 | Tuning 2 | change [95% CI] |
|---|---:|---:|---:|---:|---|
| Delivery/off | 19.8 | 19.6 | 19.5 | 20.0 | -.2 [-1.01,.61] |
| Delivery/on | 19.2 | 19.6 | 19.5 | 20.0 | +.4 [-.37,1.17] |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] |

Delivery and Zenotravel are preserved. TPP/on is nearly preserved. TPP/off is
not a broad mild loss: nine seeds retain 20/20 and seed 1972442430 collapses to
9/20. That lineage starts S2 at 10/20, peaks at test 13/20 at epoch 5, and the
validation-selected epoch 12 scores 9/20. Its anchor KL is .1059 versus peer
mean .0516. The permanent audit records seed-specific first-update instability,
not timeout/OOM/invalid plans, as the primary finding.

## Stage-1 policy versus MCTS — completed

Every row is the validation-selected Stage-1 checkpoint. MCTS columns are the
same recorded run under 30-minute, 2-hour and 6-hour per-instance cutoffs.

| Domain/VH | Search | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | raw/Holm p |
|---|---|---:|---:|---|---:|
| Block Grouping/off | fixed narrow 5/20 | 16.3 | 11.6 / 14.8 / 15.4 | -.9 [-1.88,.08] | .109/.547 |
| Block Grouping/on | fixed narrow 5/20 | 15.9 | 12.0 / 14.0 / 16.2 | +.3 [-.29,.89] | .453/.750 |
| Drone/off | fixed normal 20/70 | 5.9 | 6.9 / 6.9 / 6.9 | +1.0 [.05,1.95] | .074/.445 |
| Drone/on | fixed normal 20/70 | 5.1 | 10.0 / 10.4 / 10.4 | +5.3 [3.42,7.18] | .002/.020 |
| FO Counters/off | fixed normal 20/70 | 4.2 | 7.5 / 7.8 / 7.8 | +3.6 [2.20,5.00] | .004/.035 |
| FO Counters/on | fixed normal 20/70 | 3.7 | 5.3 / 5.7 / 5.7 | +2.0 [1.05,2.95] | .004/.035 |
| Rover/off | fixed normal 20/70 | 4.0 | 4.8 / 5.0 / 5.0 | +1.0 [.25,1.75] | .031/.219 |
| Rover/on | fixed normal 20/70 | 3.8 | 4.4 / 4.4 / 4.4 | +.6 [-.00,1.20] | .125/.547 |
| Counters/off | fixed narrow 5/20 | 32.5 | 24.9 / 25.6 / 25.7 | -6.8 [-17.73,4.13] | .250/.750 |
| Counters/on | fixed narrow 5/20 | 18.6 | 20.3 / 22.1 / 22.5 | +3.9 [-4.33,12.13] | .328/.750 |

## Stage-2 policy versus MCTS — branch-aware availability

| Domain/VH | Validation-led terminal/live/missing | Terminal-led terminal/live/missing |
|---|---:|---:|
| Block Grouping/off | 0/0/10 | 10/0/0 |
| Block Grouping/on | 8/0/2 | 9 exact logs/0/1 provenance gap |
| Drone/off | 10/0/0 | 10/0/0 |
| Drone/on | 10/0/0 | 10/0/0 |
| FO Counters/off | 6/0/4 | 4/6/0 |
| FO Counters/on | 9/0/1 | 0/5/5 |
| Rover/off | 1/0/9 | 10/0/0 |
| Rover/on | 0/0/10 | 10/0/0 |
| Counters/off | 10/0/0 | 0/0/10 |
| Counters/on | 10/0/0 | 0/0/10 |

Once the live FO branch terminates, 57 comparison jobs remain genuinely
unsubmitted: Block Grouping 13, FO Counters 5, Rover 19, Counters 20.

| Missing scope | Jobs | Per job | All-at-once request | Comparable whole-job median |
|---|---:|---:|---:|---:|
| Block Grouping | 13 | 6 CPU / 120 GiB | 78 CPU / 1,560 GiB | 12.5 h (10.2–19.2 h) |
| FO Counters | 5 | 6 CPU / 120 GiB | 30 CPU / 600 GiB | 48.3 h (30.1–72 h) |
| Rover | 19 | 6 CPU / 120 GiB | 114 CPU / 2,280 GiB | 30.1 h (26.8–42.6 h) |
| Counters narrow 5/20 | 20 | 6 CPU / 120 GiB | 120 CPU / 2,400 GiB | 49.9 h (3.6–72 h) |
| Total | 57 | — | 342 CPU / 6,840 GiB | — |

The known complete Stage-2 comparisons are:

| Domain/VH/branch | Search | n | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | raw p |
|---|---|---:|---:|---:|---|---:|
| BG/off terminal | narrow 5/20 | 10 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22,-.38] | .031 |
| BG/on terminal | narrow 5/20 | 10 aggregate; 9 exact logs | 11.6 | 4.4 / 8.7 / 9.4 | -2.2 [-4.60,.20] | .031 |
| Drone/off validation | normal 20/70 | 10 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [-.43,2.43] | .195 |
| Drone/on validation | normal 20/70 | 10 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33,8.07] | .002 |
| Drone/off terminal | normal 20/70 | 10 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [1.00,3.20] | .0078 |
| Drone/on terminal | normal 20/70 | 10 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [4.54,8.66] | .0020 |
| FO/off terminal, current terminal subset | normal 20/70 | 4 | 3.25 | 6.25 / 6.25 / 6.25 | provisional only | — |
| Rover/off terminal | normal 20/70 | 10 | 3.8 | 4.2 / 4.2 / 4.2 | +.4 [.03,.77] | .125 |
| Rover/on terminal | normal 20/70 | 10 | 4.0 | 4.5 / 4.5 / 4.5 | +.5 [.12,.88] | .0625 |
| Counters/off validation | narrow 5/20 | 10 | 36.9 | 34.9 / 36.7 / 36.7 | -.2 [-2.87,2.47] | .969 |
| Counters/on validation | narrow 5/20 | 10 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-.70,11.30] | .082 |

FO terminal remains live; its four terminal VH-off jobs score 7, 5, 8 and 5,
with all successes achieved by 30 minutes. Two VH-off and nine VH-on jobs are
running; four VH-off and one VH-on are ordinary memory-pending. The eleven
running jobs currently contain 38 classified successful instances and 69
explicit timeout records across retries;
attempts must be deduplicated by instance before a final mean is computed.

## Progressive widening

PW20 and PW70 are distinct. The completed two-seed screen used PW20 only where
the declared comparator was fixed narrow 5/20; FO Counters and Rover used PW70
against fixed normal 20/70.

| Cell | Policy | Fixed comparator | PW simulations | PW 30m / 2h / 6h | Status |
|---|---:|---:|---:|---:|---|
| BG S1/off | 16.5/20 | narrow 5/20: 11.0 / 13.5 / 15.0 | 20 | 11.5 / 15.0 / 15.0 | complete screen |
| BG S1/on | 17.0/20 | narrow 5/20: 11.5 / 13.5 / 17.0 | 20 | 12.5 / 17.0 / 18.0 | complete screen |
| Counters S1/off | 18.0/59 | narrow 5/20: 21.5 / 21.5 / 21.5 | 20 | 20.5 / 20.5 / 20.5 | complete screen |
| Counters S1/on | 5.0/59 | narrow 5/20: 13.5 / 13.5 / 13.5 | 20 | 12.5 / 12.5 / 12.5 | complete screen |
| Counters S2/off | 49.0/59 | narrow 5/20: >=37.0 / >=41.5 / 44.5 | 20 | 46.0 / 49.5 / 49.5 | complete screen |
| Counters S2/on | 5.0/59 | narrow 5/20: 17.5 / 17.5 / 17.5 | 20 | 15.0 / 15.0 / 15.0 | complete screen |
| FO Counters S1/off | 3.5/20 | normal 20/70: 9.5 / 9.5 / 9.5 | 70 | 9.5 / 9.5 / 9.5 | complete screen |
| FO Counters S1/on | 3.5/20 | normal 20/70: 7.5 / 7.5 / 7.5 | 70 | 8.0 / 8.0 / 8.0 | complete screen |
| Rover S1/off | 4.0/20 | normal 20/70: 4.5 / 4.5 / 4.5 | 70 | 5.0 / 5.0 / 5.0 | complete screen |
| Rover S1/on | 4.0/20 | normal 20/70: 4.5 / 4.5 / 4.5 | 70 | 5.5 / 5.5 / 5.5 | complete screen |

The true PW70 correction has six terminal and six running jobs:

| Cell | Policy | Fixed comparator | PW70 30m / 2h / 6h | State |
|---|---:|---:|---:|---|
| BG S1/off | 16.5/20 | narrow 5/20: 11.0 / 13.5 / 15.0 | 10.5 / 11.5 / 13.0 | 2/2 terminal |
| BG S1/on | 17.0/20 | narrow 5/20: 11.5 / 13.5 / 17.0 | 9.0 / 11.5 / 13.5 | 2/2 terminal |
| Counters S1/off | 18.0/59 | narrow 5/20: 21.5 / 21.5 / 21.5 | >=21.0 / >=21.0 / >=21.0 | 2 running; 21/21 classified |
| Counters S1/on | 5.0/59 | narrow 5/20: 13.5 / 13.5 / 13.5 | >=12.0 / >=12.5 / >=12.5 | 2 running; 17/22 classified |
| Counters S2/off | 49.0/59 | narrow 5/20: >=37.0 / >=41.5 / 44.5 | 34.0 / 39.5 / 43.5 | 2/2 terminal; both OOM-labelled but final scores retained |
| Counters S2/on | 5.0/59 | narrow 5/20: 17.5 / 17.5 / 17.5 | >=13.5 / >=14.0 / >=14.0 | 2 running; 15/22 classified |

Running correction jobs have used about 42–55 h and have about 17–30 h to the
hard limit. The 18-job confirmatory expansion is fully running. Across 359
classified instances it currently has lower bounds 230/271/284 successes at
30m/2h/6h; elapsed ranges from minutes to about 23.5 h, with at most 48.5–72 h
to the hard bounds.

The six exact-snapshot Counters divergence jobs are now all running. Current
lower bounds across 136 classified instances are 122/125/125 at 30m/2h/6h.
They compare PW20 and PW70 against the same three policy snapshots whose fixed
narrow search regressed severely; no result is excluded from the broader screen.

## Binding Horizon

Eight aware/unaware Counters jobs are running. Across 138 classified instances,
the current lower bounds are 131/135/138 successes at 30m/2h/6h. No explicit
instance timeout has yet been recorded and the longest completed plan is 1,105
actions. All emitted horizon summaries still show zero cutoffs. With 43–53 h
elapsed, hard remaining bounds are about 19–29 h. The jobs have not yet reached
the late-trajectory regime needed to test whether cutoffs are counted correctly
near 10,000 external actions, so no Horizon efficacy claim is available.

## Held/design experiments in actual priority order

| Priority | Experiment | State | Release gate / next scope |
|---:|---|---|---|
| 1 | MPRIME-VAL-ADEQUACY | Phase A active; Phase B held | freeze two independent harder validation replicates, then validation-only rescore |
| 2 | MCTS-PW-30M | held design | finish PW70 screen and run fresh hard-30m confirmation only on qualifying cells |
| 3 | ANCHOR-KL-CONTROL | held ready | implement/smoke-test constant vs adaptive-target-KL vs linear-decay 12-lineage screen |
| 4 | MCTS-RESOURCE | held design | after current FO work, test two workers/160 GiB on censored FO/Rover cells |
| 5 | STOP-ORIG | held design | small matched original-stopping pilot |
| 6 | LONG-DRONE-SELECTED-MCTS | held design | three selected-checkpoint MCTS jobs complete the selected-vs-final side comparison |
| 7 | ACT-HISTORY-ABLATION | held by user | fresh Drone/Counters/TPP networks; compare policy, fixed MCTS and PW |
| 8 | MCTS-PW-PATHBATCH | held design | implement single-backup semantics and run a small matched pilot |
| 9 | PUCT-EST | deliberately deferred | one-factor PUCT/estimator sensitivity after primary evidence |
| 10 | MCTS-SAFE2 | held low priority | require audited nonzero horizon cutoffs and a changed decision first |

ANCHOR-KL-CONTROL has three arms, not four. The earlier “strong-decaying anchor”
was a vague proposal and was replaced by the formal linear-decay arm. Constant
KL is the baseline; PPO-style adaptive target-KL changes the coefficient in
response to measured KL; Kickstarting-style linear decay decreases a prescribed
teacher/KL weight to zero with training progress. No ad-hoc high-to-nonzero
fourth arm is registered.

## Complete experiment inventory

| State | Experiment IDs |
|---|---|
| Live Slurm | PRESERVE-3-TERM; MCTS-HORIZON-COUNTERS; MCTS-LEGACY-FO; MCTS-PW70-CROSS-DOMAIN; MCTS-PW70-CONFIRMATORY; MCTS-PW-COUNTERS-DIVERGENCE |
| Active local analysis | MPRIME-VAL-ADEQUACY |
| Completed primary/policy | MAIN-VAL; MAIN-TERM; PRESERVE-3-VAL; MAIN-EXT6-MPRIME; MAIN-TERM-EXT6-MPRIME; ANCHOR-4 |
| Completed search/sensitivity | MCTS-WIDTH; MCTS-PW; MCTS-PW-SAFE; MCTS-PW-CROSS-DOMAIN; MCTS-SAFE; MCTS-SAFE-CONTEXT; MCTS-HORIZON; MAIN-VAL-S2-MCTS; MCTS-LEGACY-ROVER; MCTS-DETERMINISM-AUDIT; ENHSP-LEAF; BG-HIST |
| Completed policy with held side endpoint | LONG-DRONE; three continuation-selected MCTS jobs are priority-6 held design |
| Held/design | MCTS-PW-30M; ANCHOR-KL-CONTROL; MCTS-RESOURCE; STOP-ORIG; ACT-HISTORY-ABLATION; MCTS-PW-PATHBATCH; PUCT-EST; MCTS-SAFE2; MPrime validation Phase B |

The determinism audit is intentionally not re-expanded in routine status prose:
same CPU-family repeats produced identical decisions; different CPU families
changed internal prediction/statistics checksums but not selected actions in the
audited instances; a fresh same-node aware/unaware replay was identical with
zero cutoffs. It is complete diagnostic evidence, not an active campaign.

## Provenance and reproducibility

- MPrime validation-led seed results point directly to Stage-1 evaluation,
  Stage-2 training and Stage-2 evaluation logs in
  `mprime_validation_ipc_scale_v1/validation_led_stage2_selected_results_20260903.csv`.
- Their aggregate statistics use that seed ledger as the named companion.
- `result_provenance_index_20260902.csv` maps every other presentation summary
  to direct or companion evidence.
- `result_provenance_audit_20260902.csv` is regenerated after this update; no
  score-bearing CSV may be authoritative with a `needs_mapping` result.
- The three `.tmp_*` directories are pre-existing scratch data and are not part
  of this status update.

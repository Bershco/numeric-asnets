# Experiment status — 3 September 2026, 17:14 IDT

This snapshot joins the live Slurm queue, current training/MCTS logs, and frozen
static ledgers. Live sources are `cluster_workload_latest.csv`,
`live_training_progress_latest.csv`, and `live_mcts_progress_latest.csv`.
Every static result cited here either has absolute row-level cluster-log paths or
is mapped to its row-level evidence in `result_provenance_index_20260902.csv`.

## Cluster workload

| State | Jobs | Requested CPU | Requested RAM |
|---|---:|---:|---:|
| Running | 70 | 508 | 6,128 GiB |
| Pending | 203 | 1,946 | 5,941 GiB |
| Held | 0 | 0 | 0 GiB |

| Experiment | Running | Pending | CPU running/pending | RAM running/pending |
|---|---:|---:|---:|---:|
| Counters Binding Horizon | 8 | 0 | 48/0 | 960/0 GiB |
| Counters PW divergence recovery | 6 | 0 | 36/0 | 720/0 GiB |
| FO Counters terminal-led S2 MCTS | 12 | 1 | 72/6 | 1,440/120 GiB |
| PRESERVE-3 terminal-led S2 training | 1 | 0 | 6/0 | 48/0 GiB |
| PRESERVE-3 policy controller | 0 | 1 | 0/2 | 0/1 GiB |
| PRESERVE-3 terminal TPP/off policy | 22 | 170 | 220/1,700 | 440/3,400 GiB |
| PRESERVE-3 terminal Delivery policy retries | 0 | 12 | 0/120 | 0/240 GiB |
| MPrime missing terminal policy point | 0 | 1 | 0/10 | 0/20 GiB |
| PW70 confirmatory expansion | 15 | 0 | 90/0 | 1,800/0 GiB |
| PW70 two-seed correction | 6 | 0 | 36/0 | 720/0 GiB |
| Stage-2 MCTS branch completion — Block Grouping | 0 | 13 | 0/78 | 0/1,560 GiB |
| Stage-2 MCTS branch completion — FO Counters | 0 | 5 | 0/30 | 0/600 GiB |

The TPP/off policy scope includes exact replacements 20891761 and 20892200 for
two jobs that failed on `ise-cpu128-03` semaphore ENOSPC. Job 20890696 suffered
a node failure on `ise-cpu256-08` and was automatically requeued; it has not
been duplicated. Failed originals are not usable evidence. No job is Slurm-held.

## Live training and policy pipeline

| Experiment/cell | Terminal | Running | Current point | Realistic estimate |
|---|---:|---:|---|---|
| PRESERVE-3 terminal-led TPP/off | 10/10 | 0 | 213 every-five/selected/final policy identities submitted; 23 completed, 22 running, 168 pending at 17:15; the two failed originals have exact replacements | policy jobs normally finish within four hours after starting; queue start times depend on memory availability |
| PRESERVE-3 terminal-led TPP/on | 9/10 | 1 | final lineage at epoch 81 after about 69h17m | about 2h43m hard bound; dependent controller 20768210 waits on this cell only and will submit its policy curve automatically |
| MPrime validation-led S2 | 20/20 | 0 | all policy endpoints complete and reconciled | complete, but validation adequacy is under audit |
| MPrime terminal-led S2 | 20/20 | 0 | 839/840 Phase-A points complete; exact epoch-55 replacement job 20890973 is pending | four-hour allocation after scheduling; validation adequacy remains under audit |
| PRESERVE-3 Delivery | 20/20 | 0 | original curve campaigns terminal, but 12 rows lacked scores; exact checkpoint retries 20892672–20892683 are pending, including one selected endpoint per VH mode | four-hour allocation after scheduling; no retraining |
| PRESERVE-3 Zenotravel | 20/20 | 0 | all policy curves/endpoints complete | complete |

TPP/off continuation job 20834985 resumed from cumulative S2 epoch 84 and
finished at epoch 100. This continuation was caused by the cluster-outage
requeue/restart defect, not an ordinary scheduler timeout. Its final checkpoint
is continuation snapshot 15. Restored validation state identifies parent job
20755752 snapshot 75 (cumulative S2 epoch 75) as the inherited validation-best
checkpoint. The repaired materializer combines the original directory at
offset 0 with the continuation directory at offset 85. Two local regression
tests pass. Both absolute checkpoints and both logs are frozen in
`four_domain_preservation/tpp_off_continuation_provenance_20260903.csv`.
The repaired controller 20890658 completed successfully and submitted all 213
TPP/off policy identities. Their manifest retains the source checkpoint, source
training job, epoch and training-log pointer for every row. The epoch-90 and
epoch-15 replacements have their own two-row manifest and submission ledger so
the ENOSPC failures are never mistaken for scores.

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
already 30/30. Phase A now contains all 290 Stage-1 checkpoint scores and
839/840 Stage-2 checkpoint scores. The sole missing Stage-2 score is terminal-led
VH-off seed 923500475, epoch 55: job 20862221 failed and produced no score.

| Phase-A scope | VH | Lineages/checkpoints | Unique validation scores per lineage | Fraction at maximum | Positive validation/test Spearman | Mean selected-test regret |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 | off | 10/153 | 7.3 mean | .124 mean | 5/10 | 3.1 plans |
| Stage 1 | on | 10/137 | 7.7 mean | .105 mean | 7/10 | 2.5 plans |
| S2 validation-led | off | 10/210 | exactly 1 | 1.000 | 0/10; undefined within-lineage correlation | 1.9 plans |
| S2 validation-led | on | 10/210 | exactly 1 | 1.000 | 0/10; undefined within-lineage correlation | 2.6 plans |
| S2 terminal-led | off | 10/209 scored | exactly 1 | 1.000 | 0/10; undefined within-lineage correlation | 3.0 plans |
| S2 terminal-led | on | 10/210 | exactly 1 | 1.000 | 0/10; undefined within-lineage correlation | 2.5 plans |

Stage 1 has weak checkpoint-ranking resolution; Stage 2 is completely
saturated across every recorded checkpoint. That makes the two independently
frozen harder validation replicates the highest-priority next design task.

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

| Domain/VH | S1 selected | S2 selected all 10 | Held-out 8 | Tuning 2 | change [95% CI] | raw/Holm p |
|---|---:|---:|---:|---:|---|---:|
| Delivery/off | 19.8 | 19.6 | 19.5 | 20.0 | -.2 [-1.01,.61] | 1/1 |
| Delivery/on | 19.2 | 19.6 | 19.5 | 20.0 | +.4 [-.37,1.17] | .5/1 |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | 1/1 |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | 1/1 |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | 1/1 |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | 1/1 |

Delivery and Zenotravel are preserved. TPP/on is nearly preserved. TPP/off is
not a broad mild loss: nine seeds retain 20/20 and seed 1972442430 collapses to
9/20. That lineage starts S2 at 10/20, peaks at test 13/20 at epoch 5, and the
validation-selected epoch 12 scores 9/20. Its anchor KL is .1059 versus peer
mean .0516. The permanent audit records seed-specific first-update instability,
not timeout/OOM/invalid plans, as the primary finding.

## PRESERVE-3 terminal-led — current result state

Both terminal-led branches were trained for Delivery, TPP and Zenotravel. The
terminal-led experiment is not yet fully result-complete: Zenotravel is complete;
Delivery has two selected-endpoint retries pending; TPP/off policy evaluations
are live; TPP/on still has one training lineage plus its dependent policy
controller. The currently defensible selected-endpoint results are:

| Domain/VH | matched n | S1 final | S2 selected | held-out / tuning S2 | change [95% CI] | raw p | status |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 9/10 | 14.67 | 15.22 | 14.14 / 19.00 | +.56 [-3.35,4.46] | .801 | provisional; selected endpoint retry 20892673 pending |
| Delivery/on | 9/10 | 17.22 | 18.78 | 19.14 / 17.50 | +1.56 [-1.40,4.51] | .344 | provisional; selected endpoint retry 20892676 pending |
| TPP/off | 0/10 | — | — | — | — | — | 213-point policy campaign live |
| TPP/on | 0/10 | — | — | — | — | — | final training lineage and cell controller live |
| Zenotravel/off | 10/10 | 20.00 | 19.80 | 19.875 / 19.50 | -.20 [-.50,.10] | .500 | complete; preserved |
| Zenotravel/on | 10/10 | 19.80 | 20.00 | 20.00 / 20.00 | +.20 [-.10,.50] | .500 | complete; preserved |

The seed-level file includes direct Stage-1 and Stage-2 training/evaluation-log
pointers and both checkpoint paths. The Delivery table excludes rather than
zero-fills the two failed selected endpoints. All twelve scoreless Delivery
curve rows were resubmitted so the learning curves are also made whole.

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
| Block Grouping/off | 0/10/0 | 10/0/0 |
| Block Grouping/on | 8/2/0 | 9 exact logs/1/0 |
| Drone/off | 10/0/0 | 10/0/0 |
| Drone/on | 10/0/0 | 10/0/0 |
| FO Counters/off | 6/4/0 | 4/6/0 |
| FO Counters/on | 9/1/0 | 3/7/0 |
| Rover/off | 1/0/9 | 10/0/0 |
| Rover/on | 0/0/10 | 10/0/0 |
| Counters/off | 10/0/0 | 0/0/10 |
| Counters/on | 10/0/0 | 0/0/10 |

The complete 57-job branch-completion gap has now been acted on in the requested
order. The first 18 jobs are submitted and ordinary resource-pending: BG off 10
and on 3 at narrow 5/20, plus FO off 4 and on 1 at normal 20/70. The remaining
39 are still unsubmitted by design: Rover 19 and terminal-led Counters 20. The
18-row immutable submission ledger contains each job ID, source checkpoint,
source training job, seed, VH mode, width and simulation budget.

| Missing scope | Jobs | Per job | All-at-once request | Comparable whole-job median |
|---|---:|---:|---:|---:|
| Block Grouping | 13 | 6 CPU / 120 GiB | 78 CPU / 1,560 GiB | 12.5 h (10.2–19.2 h) |
| FO Counters | 5 | 6 CPU / 120 GiB | 30 CPU / 600 GiB | 48.3 h (30.1–72 h) |
| Rover | 19 | 6 CPU / 120 GiB | 114 CPU / 2,280 GiB | 30.1 h (26.8–42.6 h) |
| Counters narrow 5/20 | 20 | 6 CPU / 120 GiB | 120 CPU / 2,400 GiB | 49.9 h (3.6–72 h) |
| Submitted now: BG + FO | 18 | — | 108 CPU / 2,160 GiB | pending |
| Still unsubmitted: Rover + Counters | 39 | — | 234 CPU / 4,680 GiB | awaiting the next explicit release decision |
| Full completion scope | 57 | — | 342 CPU / 6,840 GiB | — |

The known complete Stage-2 comparisons are:

| Domain/VH/branch | Search | n | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | raw p |
|---|---|---:|---:|---:|---|---:|
| BG/off terminal | narrow 5/20 | 10 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22,-.38] | .031 |
| BG/on terminal | narrow 5/20 | 10 aggregate; 9 exact logs | 11.6 | 4.4 / 8.7 / 9.4 | -2.2 [-4.60,.20] | .031 |
| Drone/off validation | normal 20/70 | 10 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [-.43,2.43] | .195 |
| Drone/on validation | normal 20/70 | 10 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33,8.07] | .002 |
| Drone/off terminal | normal 20/70 | 10 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [1.00,3.20] | .0078 |
| Drone/on terminal | normal 20/70 | 10 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [4.54,8.66] | .0020 |
| FO/off terminal, current terminal subset | normal 20/70 | 4 | 3.25 | 6.25 / 6.25 / 6.25 | provisional +3.00 | — |
| FO/on terminal, current terminal subset | normal 20/70 | 3 | 4.33 | 5.00 / 5.00 / 5.00 | provisional +.67 | — |
| Rover/off terminal | normal 20/70 | 10 | 3.8 | 4.2 / 4.2 / 4.2 | +.4 [.03,.77] | .125 |
| Rover/on terminal | normal 20/70 | 10 | 4.0 | 4.5 / 4.5 / 4.5 | +.5 [.12,.88] | .0625 |
| Counters/off validation | narrow 5/20 | 10 | 36.9 | 34.9 / 36.7 / 36.7 | -.2 [-2.87,2.47] | .969 |
| Counters/on validation | narrow 5/20 | 10 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-.70,11.30] | .082 |

FO terminal remains live. Four VH-off terminal jobs score 7, 5, 8 and 5; three
VH-on terminal jobs score 5, 5 and 5. Every terminal success occurred within
30 minutes, so their 30m/2h/6h columns are identical. Six off jobs and six on
jobs are running; one on job is ordinary memory-pending. The running logs
currently contain nine successful classified instances. Retry timeout messages
are diagnostic only and will be deduplicated by instance before the final
ten-seed statistics. The seven-row authoritative subset links each policy log,
MCTS log, training log and checkpoint.

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
| Counters S1/off | 18.0/59 | narrow 5/20: 21.5 / 21.5 / 21.5 | >=21.0 / >=21.0 / >=21.0 | 2 running; retained/live evidence remains a lower bound |
| Counters S1/on | 5.0/59 | narrow 5/20: 13.5 / 13.5 / 13.5 | >=12.0 / >=12.5 / >=12.5 | 2 running; 17/22 classified |
| Counters S2/off | 49.0/59 | narrow 5/20: >=37.0 / >=41.5 / 44.5 | 34.0 / 39.5 / 43.5 | 2/2 terminal; both OOM-labelled but final scores retained |
| Counters S2/on | 5.0/59 | narrow 5/20: 17.5 / 17.5 / 17.5 | >=13.5 / >=14.0 / >=14.0 | 2 running; live/requeue logs must be joined with retained ledgers |

The six running correction jobs currently contain 118 classified instances and
lower bounds of >=93/>=95/>=95 successes at 30m/2h/6h. Their elapsed times range
from 1h40m to 61h28m; the oldest hard bound is about 10h32m and younger requeues
have substantially longer bounds.

Three of the 18 confirmatory jobs are terminal. FO Counters S1/on seeds
534933607 and 923500475 score 7/20 and 6/20 versus policy 2/20 and 3/20 and
fixed 20/70 scores 4/20 and 4/20. Rover S1/off seed 1073581256 scores 4/20
versus policy 4/20 and fixed 20/70 6/20. All 17 successes occurred within
30 minutes and all are VAL-valid; the second FO allocation ended OOM only after
its complete 6/20 result was printed. Fifteen jobs remain running. Their live
322 classified instances have lower bounds >=213/>=254/>=273; including the
three terminal jobs gives current campaign lower bounds >=230/>=271/>=291 over
383 classified or terminal instance records.

All six exact-snapshot Counters divergence jobs are running. Current live lower
bounds across 176 classified instances are >=125/>=139/>=143 at 30m/2h/6h.
They compare PW20 and PW70 against the same three policy snapshots whose fixed
narrow search regressed severely; no result is excluded from the broader screen.

## Binding Horizon

Eight aware/unaware Counters jobs are running. Across 138 classified instances,
the current lower bounds are 131/135/138 successes at 30m/2h/6h. No explicit
instance timeout has yet been recorded and the longest completed plan is 1,105
actions. All emitted horizon summaries still show zero cutoffs. Seven jobs have
run about 59–60 h and have 12–13 h hard bounds; one outage-requeued job has run
about 50 h and has a 22 h hard bound. The jobs have not yet reached the
late-trajectory regime needed to test whether cutoffs are counted correctly near
10,000 external actions, so no Horizon efficacy claim is available.

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
| Live Slurm | PRESERVE-3-TERM; MPRIME-VAL-ADEQUACY one-point repair; MCTS-HORIZON-COUNTERS; MCTS-LEGACY-FO; MCTS-STAGE2-BRANCH-COMPLETION; MCTS-PW70-CROSS-DOMAIN; MCTS-PW70-CONFIRMATORY; MCTS-PW-COUNTERS-DIVERGENCE |
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

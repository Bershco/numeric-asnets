# Current experiment status — 30 August 2026, 20:45 IDT

Exact jobs are in `../live_jobs.csv`; grouped resources are in
`cluster_workload.csv`; machine-readable experiment states are in
`experiment_status.csv`.

## Cluster workload

| Queue class | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 67 | 438 | 6,132 GiB |
| Ordinary/resource/dependency pending | 271 | 2,538 | 9,063 GiB |
| Deliberately held | 20 | 120 | 2,400 GiB |
| Total pending shown by Slurm | 291 | 2,658 | 11,463 GiB |

The running allocation is exactly 6,132 GiB, twelve GiB below the 6-TiB
per-user memory ceiling. New heavy jobs may be submitted and wait safely, but
cannot start until running jobs release memory. The three one-GiB controllers
are dependency-pending by design.

## Automatic policy chain and its boundary

- MPrime controller `20743983` waits for all nine remaining anchor-training
  jobs.
- Delivery controller `20743984` waits for its final two held-out Stage-2 jobs.
- TPP controller `20743985` waits for its final three held-out Stage-2 jobs.

These idempotent controllers submit every-five and selected/final policy
evaluations. They do **not** choose MPrime anchors or launch MPrime Stage 2.
MPrime coefficient selection reads validation AUC/peak/final values directly
from the training logs; external test-policy curves are evidence, not tuning
inputs. A reviewed finalizer is still needed before the first forty MPrime
Stage-2 lineages (twenty validation-led and twenty terminal-led) are released.

## Live experiment table

| Experiment | Queue/resources now | Scientific position | Best current evidence | Timing |
|---|---|---|---|---|
| MPrime anchor tuning | 9 running; 54 CPU, 432 GiB | 19/28 terminal; validation-led tuning only | One coefficient per VH will be selected by two-seed validation AUC, then peak/final validation tie-breaks | Most remaining jobs were at epochs 64–96; roughly 1–5 h from the 19:55 audit |
| MPrime tuning policy | 7 running + 207 pending; running 70 CPU/140 GiB | Analysis curves/endpoints, not the coefficient selector | Final controller attached to all remaining training | Starts/finishes as memory frees |
| Delivery Stage 2 | 2 running; 12 CPU/96 GiB | 14/16 held-out lineages terminal; four tuning winners give 20 total eventual validation-led lineages | Frozen anchor 30 for both VH modes | Both at epoch 97; about 1–2 h |
| Delivery policy | 2 running + 24 pending; running 20 CPU/40 GiB | Every-five and selected/final endpoints | Final dependency controller attached | Memory-dependent |
| TPP Stage 2 | 3 running; 18 CPU/144 GiB | 13/16 held-out terminal; four tuning winners reused | Frozen anchors off=3 and on=10 | Epochs 52/74/86: about 27 h hard-bound for the slowest, approximately 16 h and 7 h for the others |
| Counters Stage-2 narrow | 6 running; 36 CPU/720 GiB | 14/20 terminal | Off n=8: policy 43.13 vs narrow 42.50; on n=6: 30.83 vs 32.83 | At most about 16 h to the 72-h allocation cap |
| Binding Horizon | 1 running; 6 CPU/120 GiB | 19/20 terminal; nine matched pairs | Seven ties, one aware +1, one aware -3; paired mean -0.22, 95% CI [-1.06, 0.62], raw sign-flip p=1.0 | Final aware arm has run about 9.5 h; hours expected, 72-h cap |
| SAFE-CONTEXT | 18 running; 108 CPU/2,160 GiB | Physical-only versus action-history-contextual nodes | Diagnostic different-context revisits were frequent; node multiplier ranged 1.00–5.44x without tracker overflow | Many instances finish in 2–12 h; timeout tails may extend to 12–30 h |
| Budget-matched cross-domain PW | 10 running + 9 pending; running 60 CPU/1,200 GiB | Two seeds per domain/stage/VH screen | First Counters S2 row: policy 59, fixed narrow 49, PW 59; PW 3:47 vs fixed 18:05 | Running arms: hours to about one day; pending start unknown under memory cap |
| MAIN-VAL Drone S2 MCTS | 16 pending; 96 CPU/1,920 GiB requested | Four historical rows plus sixteen confirmed gaps | Completes Stage-2 policy-vs-MCTS evidence | Start unknown; memory-blocked |
| Rover endpoint MCTS | 9 running + 12 pending; running 54 CPU/1,080 GiB | Released legacy endpoint work; fixed-budget OOM risk is part of outcome | Early running lower bounds are not final aggregate evidence | Historical median allocation about 30 h; 72-h cap |

## Counters Stage-2 narrow interim statistics

All unclassified instances after OOM/allocation termination remain failures;
all printed successful plans in the terminal subset are VAL-confirmed.

| VH | Matched terminal n | Policy | Narrow 5/20 | Change | 95% CI | Raw exact p |
|---|---:|---:|---:|---:|---:|---:|
| Off | 8 | 43.13/59 | 42.50/59 | -0.63 | [-4.00, 2.75] | .922 |
| On | 6 | 30.83/59 | 32.83/59 | +2.00 | [-5.02, 9.02] | .563 |

These rows are still interim because six matched seeds run.

## Progressive widening: budget correction

The completed Drone PW pilot and Kmin=3 extension used 70 simulations. The new
cross-domain screen is deliberately comparator-matched rather than uniform:

- Block Grouping and Counters use 20 simulations to compare PW against fixed
  narrow 5/20.
- FO Counters and Rover use 70 simulations to compare PW against fixed normal
  20/70.

The first Counters result therefore shows a strong equal-small-budget
improvement over fixed narrow search. It does not yet tell us how Counters PW70
performs. Promotion occurs by domain-stage-VH cell, not by favourable seed: PW
must preserve policy, stay within five percentage points of fixed search, and
materially improve efficiency/safety. If either VH mode qualifies, both modes
are expanded to five seeds. A promoted Block Grouping/Counters cell then needs
a separately labelled PW70 confirmation for a standard-budget claim.

## Stable-domain Stage-2 correction and Zenotravel result

The existing Delivery, TPP and Zenotravel Stage-2 lineages were initialized
from validation-selected Stage-1 checkpoints. Comparing their Stage-2 scores
to Stage-1 final scores is descriptive; it is not a terminal-led intervention.
A true stable-domain terminal-led branch therefore remains a sixty-lineage
held design (three domains x two VH x ten seeds), with exact duplicates reusable
only when selected and final checkpoints truly coincide.

Zenotravel validation-led Stage 2 is complete:

| VH | S1 validation-selected | S2 validation-selected | Paired change (95% CI) | Raw p |
|---|---:|---:|---:|---:|
| Off | 20.0/20 | 20.0/20 | 0.0 [0.0, 0.0] | 1.0 |
| On | 20.0/20 | 19.9/20 | -0.1 [-0.33, 0.13] | 1.0 |

There is no statistically supported degradation. Stage-1 final means were
20.0/off and 19.8/on, but comparisons against them remain descriptive until a
real final-led Stage-2 branch is run.

## Horizon next step

Drone-750 has produced only one recorded cutoff and the newest aware arm fell
from 9/20 to 6/20 despite zero cutoffs in that run. The campaign is therefore a
non-regression/variation study, not evidence that horizon enforcement helps.
The preferred efficacy follow-up is Counters, where executions genuinely reach
the 10,000-action budget. A held eight-job pilot is registered: two seeds, two
VH modes, fresh aware/unaware arms from one commit, normal width20/70, no PW.
It expands only if cutoffs actually occur.

## Held and deferred

- FO Counters endpoint MCTS: 20 Slurm-held jobs, 120 CPU/2,400 GiB requested.
- Stable-domain terminal-led Stage 2: 60-lineage design hole, not submitted.
- Counters Horizon efficacy: eight-job held design; wait for Drone closure and
  memory capacity.
- MCTS-PW-30M: held-ready until cross-domain cells are selected.
- MCTS-RESOURCE: two-worker/160-GiB FO Counters/Rover sensitivity.
- STOP-ORIG: original training-success stopping replication.
- SAFE-2: fully horizon-indexed statistics, explicitly held by the user.
- Path-batched PW and PUCT/estimator sensitivity remain design-held.

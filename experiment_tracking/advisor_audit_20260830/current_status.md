# Current experiment status — 31 August 2026, 00:13 IDT

Exact queue rows and log pointers are in `../live_jobs.csv`. This file is the
human-readable interpretation of that immutable snapshot; completed mainstream
policy results remain in the authoritative endpoint/statistics ledgers and are
not repeated here by default.

## Cluster workload

| Queue class | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 86 | 672 | 6,132 GiB |
| Ordinary/resource pending | 150 | 1,104 | 8,580 GiB |
| Dependency pending | 2 | 4 | 3 GiB |
| Deliberately Slurm-held | 20 | 120 | 2,400 GiB |
| Total pending shown by Slurm | 172 | 1,228 | 10,983 GiB |

Running memory is twelve GiB below the 6-TiB per-user ceiling. This is why the
heavy newly submitted work is waiting rather than starting immediately. Held
jobs count toward Slurm's pending-job count but request no allocated memory.

## Live, eligible, and dependency-gated experiments

| Experiment | Running / pending | Running resources | Scientific progress | Timing |
|---|---:|---:|---|---|
| MPrime anchor tuning | 1 / 0 | 6 CPU, 48 GiB | 27/28 terminal; final lineage at epoch 87/100 | Trainer ETA about 2h36m |
| MPrime tuning-policy evidence | 30 / 0 | 300 CPU, 600 GiB | Running work retained; 73 not-started redundant all-coefficient evaluations cancelled | Individual jobs should clear over hours; no coefficient decision waits for them |
| MPrime Stage-2 finalizer | 0 / 1 dependency | 0 | Job 20755820 freezes anchors and submits 16 held-out validation-led plus 20 terminal-led lineages | Begins immediately after the last tuning job terminates |
| Delivery validation-led Stage-2 policy | 9 / 51 | 90 CPU, 180 GiB | Training is complete; remaining every-five/endpoints are queued | Memory-dependent; expected hours after slots open |
| TPP validation-led Stage 2 | 3 / 0 | 18 CPU, 144 GiB | 13/16 held-out lineages terminal; four tuning winners are reused | Epochs 92/79/56: roughly 4–7h, 12–17h, and at most 23h to scheduler limit |
| TPP policy finalizer | 0 / 1 dependency | 0 | Idempotent every-five/endpoints refresh | Starts after all three training jobs terminate |
| PRESERVE-3-TERM | 0 / 60 | 0 | True final-Stage1-led Delivery/TPP/Zenotravel Stage 2, 60 unique lineages | Eligible; start follows memory availability; each has a 72h cap |
| Counters Stage-2 narrow 5/20 | 5 / 0 | 30 CPU, 600 GiB | 15/20 terminal, 14 reconciled; five tails remain | About 10h42m to their 72h caps at snapshot |
| Cross-domain PW20 screen | 14 / 3 | 84 CPU, 1,680 GiB | 3/20 terminal; Block Grouping/on n=2 and Counters-S2/off n=1 | Current jobs range under 1h to about 11h; tails may run to 72h |
| PW70 correction | 0 / 12 | 0 | Same Block Grouping/Counters cells rerun with intended 70 simulations | Eligible; start unknown under memory ceiling |
| Counters Horizon efficacy | 0 / 8 | 0 | Fresh same-commit aware/unaware pairs at the real 10,000-action cap | Eligible; start unknown under memory ceiling |
| SAFE-CONTEXT | 3 / 0 | 18 CPU, 360 GiB | 17/20 terminal; final three have 11–16 classified instances and 3–6 timeouts | About 14–15h elapsed; timeout tails can extend to 72h |
| MAIN-VAL Drone Stage-2 MCTS gap | 0 / 16 | 0 | Completes the 20-row validation-led Stage-2 policy/MCTS comparison | Eligible; memory-blocked |
| Rover endpoint MCTS | 21 / 0 | 126 CPU, 2,520 GiB | Twenty terminal-led Stage-2 endpoints plus one older validation-led endpoint | 1–10.6h elapsed; historical median allocation about 30h; 72h cap |

## Deliberately held or design-held

| Experiment | Held scope | Reason / release condition |
|---|---:|---|
| FO Counters endpoint MCTS | 20 Slurm-held jobs; 120 CPU and 2,400 GiB requested if released | Preserve queue capacity; release only after higher-priority PW/Horizon/mainstream work |
| MCTS-PW-30M | Manifest-ready design | Wait for cross-domain cells to qualify, then run fresh hard-30-minute comparisons |
| MCTS-RESOURCE | Two-worker/160-GiB FO Counters/Rover resumptions | Deploy/use lifecycle-safe code and report separately as resource sensitivity |
| SAFE-2 | Design-held | User explicitly deferred horizon-indexed `(state,h)` statistics |
| PW path-batched widening | Design-held | Nonstandard multi-expansion update semantics require a separate pilot |
| STOP-ORIG and PUCT/estimator sensitivities | Design-held | Not current resource priorities |

## Current comparative evidence

### Counters validation-led Stage-2 policy versus narrow MCTS

The reconciled rows are final fixed-budget outcomes; incomplete instances after
OOM/allocation termination count as failures and every printed plan is VAL
checked.

| VH | Reconciled n | Policy | Narrow 5/20 | Change | 95% CI | Raw exact p |
|---|---:|---:|---:|---:|---:|---:|
| Off | 8 | 43.13/59 | 42.50/59 | -0.63 | [-4.00, 2.75] | .922 |
| On | 6 | 30.83/59 | 32.83/59 | +2.00 | [-5.02, 9.02] | .563 |

The fifteenth terminal job still needs static reconciliation; five jobs remain
live. These are final scores for the terminal subset, not optimistic lower
bounds.

### Cross-domain PW screening

The earlier Block Grouping/Counters rows accidentally used 20 simulations. We
retain them as a valid narrow-budget experiment and have submitted a distinct
PW70 correction. They must be displayed separately.

| Cell | n | Policy | Fixed 5/20 | PW20 | Fixed runtime | PW20 runtime | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| Block Grouping Stage1/on | 2 | 17.0/20 | 17.0/20 | 18.0/20 | 8h38m | 9h22m | Coverage-positive candidate; no demonstrated runtime gain |
| Counters Stage2/off | 1 | 59/59 | 49/59 | 59/59 | 18h05m | 3h47m | Strong provisional candidate; needs second seed |

No cell is yet formally promoted or rejected. Formal promotion requires the
complete two-seed cell, no systematic policy-success regression, and an
efficiency or retained-node benefit. The twelve PW70 jobs are a correction,
not a replacement; final reporting will contain separate PW20 and PW70 columns.

### Drone binding-Horizon experiment — completed

All twenty aware/unaware arms completed inference and are VAL-confirmed. Across
ten matched pairs: seven ties, one aware +1, one aware -3, and one aware +2.

| Statistic | Result |
|---|---:|
| Mean paired coverage change | 0.00/20 |
| 95% paired t interval | [-0.89, +0.89] |
| Exact sign-flip p | 1.000 |

For the surprising -3 pair, the unaware-only successes used 170, 253, and 237
external actions; the aware failures stopped unsolved after 206, 510, and 323.
All are far below the 750-action cap and that aware run logged zero horizon
cutoffs. The losses therefore were not rejected as horizon-infeasible paths.
They are run/process variation in a non-bitwise-deterministic multiprocessing
TensorFlow/MDPSim search stack. The zero mean over all ten pairs supports the
conclusion that Drone-750 did not demonstrate a causal Horizon effect.

### Zenotravel validation-led Stage 2 — completed comparison

| VH | Stage1 selected | Stage2 selected | Change (95% CI) | Raw p |
|---|---:|---:|---:|---:|
| Off | 20.0 | 20.0 | 0.0 [0.0, 0.0] | 1.0 |
| On | 20.0 | 19.9 | -0.1 [-0.33, 0.13] | 1.0 |

The newly submitted PRESERVE-3-TERM branch is the separate causal test starting
from Stage1 final checkpoints; numerical comparison against those finals alone
would not substitute for that intervention.

## Immediate automatic sequence

1. The final MPrime anchor lineage terminates.
2. Finalizer 20755820 freezes validation-selected coefficients and submits both
   Stage-2 branches idempotently.
3. Existing TPP dependency controller releases all newly available policy
   curves/endpoints.
4. Ready heavy jobs (PRESERVE-3-TERM, PW70, Counters Horizon, Drone Stage-2
   MCTS) enter freed 120/48-GiB slots according to Slurm priority.
5. The PW20 and Counters-narrow terminal ledgers are reconciled as each live job
   ends; no completed inference is rerun merely for extraction.

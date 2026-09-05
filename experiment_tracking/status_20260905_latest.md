# Thesis experiment status — refreshed 2026-09-05 13:36 IDT

This snapshot is based on a successful live Slurm query through the documented
normal Windows profile and `uni-cluster` alias. Counters scores are out of 59;
all other domain scores are out of 20. Every MCTS table reports deterministic
30-minute, 2-hour and 6-hour per-instance cutoffs where elapsed records exist.

## Current cluster workload

| State | Jobs | Requested CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 51 | 306 | 6,120 GiB |
| Pending | 0 | 0 | 0 GiB |
| Held | 0 | 0 | 0 GiB |

| Live experiment | Jobs | CPUs | RAM |
|---|---:|---:|---:|
| Counters terminal-led Stage-2 MCTS, narrow 5/20 | 20 | 120 | 2,400 GiB |
| Rover validation-led Stage-2 MCTS, normal 20/70 | 19 | 114 | 2,280 GiB |
| FO Counters validation-led Stage-2 MCTS, normal 20/70 | 5 | 30 | 600 GiB |
| FO Counters terminal-led Stage-2 MCTS, normal 20/70 | 2 | 12 | 240 GiB |
| PW70 two-seed correction | 1 | 6 | 120 GiB |
| PW70 five-seed confirmation | 1 | 6 | 120 GiB |
| Counters exact-snapshot PW70 divergence recovery | 3 | 18 | 360 GiB |

The queue is at the effective user memory ceiling. No additional job should be
submitted now. The twenty Counters jobs were the already-approved next workload:
all nineteen Rover replacements were running and the pre-submission request was
below 6 TiB. Nine Counters originals failed before inference on incompatible
nodes; only those nine exact identities were recreated with the full known
bad-node exclusion list. All twenty identities are now running.

Row-level scheduler, stdout-log and compact-completion-record provenance is in
`live_mcts_job_provenance_20260905_1314.csv`.

## Running and live experiments

| Experiment | Terminal / target | Live | Current evidence | Realistic timing |
|---|---:|---:|---|---|
| Stage-2 branch completion: BG validation | 20/20 | 0 | Both VH modes complete | Complete |
| Stage-2 branch completion: FO validation | 15/20 | 5 | off 6+4 live; on 9+1 live | Median analogue 48.3h; current jobs 23–26h; roughly 20–46h remaining |
| Stage-2 branch completion: Rover validation | 1/20 | 19 | Most jobs have processed 13–19/20 instances including timeouts | Most likely 0–15h; hard bounds 46–51h |
| Stage-2 branch completion: Counters terminal | 0/20 | 20 | Narrow 5/20; easy-instance lower bounds already 6–30 | Median analogue 49.9h; fast seeds may finish in hours; tail 2–3 days |
| FO terminal Stage-2 MCTS | 18/20 | 2 | off complete; on current all-seed lower bound 4.4 | 25h and 37h hard bounds; timeout-heavy tail likely |
| PW70 correction | 11/12 | 1 | only Counters S1/off seed 2011206605 remains | 25.6h hard bound; likely near-limit |
| PW70 confirmation | 17/18 | 1 | only Counters S2/on seed 923500475 remains | 12.5h hard bound; likely near-limit |
| Counters divergence recovery | PW20 3/3; PW70 0/3 | 3 | PW70 lower bounds 21, 21 and 19 | 19–27h hard bounds; likely near-limit |
| MPRIME-VAL-ADEQUACY Phase B | preparation | 0 | readiness 35%; no frozen harder replicate yet | no Slurm estimate |

### Stage-2 MCTS branch availability

| Domain/VH | Validation terminal / live / missing | Terminal terminal / live / missing | Search |
|---|---:|---:|---|
| BG/off | 10 / 0 / 0 | 10 / 0 / 0 | narrow 5/20 |
| BG/on | 10 / 0 / 0 | 10 / 0 / 0 | narrow 5/20 |
| Drone/off | 10 / 0 / 0 | 10 / 0 / 0 | normal 20/70 |
| Drone/on | 10 / 0 / 0 | 10 / 0 / 0 | normal 20/70 |
| FO/off | 6 / 4 / 0 | 10 / 0 / 0 | normal 20/70 |
| FO/on | 9 / 1 / 0 | 8 / 2 / 0 | normal 20/70 |
| Rover/off | 1 / 9 / 0 | 10 / 0 / 0 | normal 20/70 |
| Rover/on | 0 / 10 / 0 | 10 / 0 / 0 | normal 20/70 |
| Counters/off | 10 / 0 / 0 | 0 / 10 / 0 | narrow 5/20 |
| Counters/on | 10 / 0 / 0 | 0 / 10 / 0 | narrow 5/20 |

All 57 previously absent branch-completion identities are therefore either
terminal (13) or running (44); none remains unsubmitted.

### Stage-2 policy versus MCTS: completed and newly closed rows

| Domain/VH | Branch | Search | n | Policy | MCTS 30m / 2h / 6h | 6h change, 95% CI | Raw / Holm p | Conclusion |
|---|---|---|---:|---:|---:|---|---:|---|
| BG/off | validation | narrow 5/20 | 10 | 16.0 | 11.4 / 15.0 / 15.7 | -0.3 [-0.89, 0.29] | .500 / pending-family | Neutral at 6h; 30m is too short |
| BG/on | validation | narrow 5/20 | 10 | 12.8 | 10.1 / 10.6 / 12.6 | -0.2 [-1.08, 0.68] | .813 / pending-family | Neutral at 6h; 30m is too short |
| BG/off | terminal | narrow 5/20 | 10 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22, -0.38] | .031 / .219 | MCTS loses coverage |
| BG/on | terminal | narrow 5/20 | 10 | 11.6 | 4.4 / 8.7 / 9.4 | -2.2 [-4.60, 0.20] | .031 / .219 | MCTS loses mean coverage |
| Drone/off | validation | normal 20/70 | 10 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [-0.43, 2.43] | .195 / .391 | Positive but not significant |
| Drone/on | validation | normal 20/70 | 10 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33, 8.07] | .002 / .020 | Large significant gain |
| Drone/off | terminal | normal 20/70 | 10 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [0.65, 3.55] | .020 / .156 | Positive; not Holm-significant |
| Drone/on | terminal | normal 20/70 | 10 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [5.47, 7.73] | .002 / .020 | Large significant gain |
| FO/off | terminal | normal 20/70 | 10 | 2.8 | 5.3 / 5.3 / 5.3 | +2.5 [1.73, 3.27] | .002 / <=.010 | Strong significant gain |
| FO/on | terminal | normal 20/70 | 8 terminal + 2 live | 3.8 | >=4.4 / >=4.4 / >=4.4 | live | — | Current lower bound is positive |
| Rover/off | terminal | normal 20/70 | 10 | 3.8 | 4.2 / 4.2 / 4.2 | +0.4 [0.03, 0.77] | .125 / .375 | Small non-significant gain |
| Rover/on | terminal | normal 20/70 | 10 | 4.0 | 4.5 / 4.5 / 4.5 | +0.5 [0.12, 0.88] | .063 / .313 | Small non-significant gain |
| Counters/off | validation | narrow 5/20 | 10 | 36.9 | 34.9 / 36.7 / 36.7 | -0.2 [-2.87, 2.47] | .969 / .969 | No meaningful change |
| Counters/on | validation | narrow 5/20 | 10 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [-0.70, 11.30] | .082 / .328 | Positive mean; high variance |

FO/off's Holm value is bounded above by 5×.001953=.009766 in its eventual
five-domain RQ2 family, so it remains significant regardless of the missing
Counters terminal result. Job 20974692 deduplicated job 20559567's eight retry
records to four unique successful instances; all four passed VAL with zero
invalid plans. The FO/off row is now fully certified without repeated inference.

### Progressive widening

| Domain/stage/VH | n | Policy | Fixed comparator 30m / 2h / 6h | PW70 30m / 2h / 6h | Conclusion |
|---|---:|---:|---:|---:|---|
| BG S1/off | 2 | 16.5 | narrow 11.5 / 15.0 / 15.0 | 10.5 / 11.5 / 13.0 | Unpromising |
| BG S1/on | 2 | 17.0 | narrow 12.5 / 17.0 / 18.0 | 9.0 / 11.5 / 13.5 | Unpromising |
| Counters S1/off | 2 | 18.0 | narrow 20.5 / 20.5 / 20.5 | >=21 / >=21 / >=21 | Live; do not freeze |
| Counters S1/on | 2 | 5.0 | narrow 12.5 / 12.5 / 12.5 | 12.0 / 12.5 / 12.5 | Matches fixed at 2h/6h |
| Counters S2/off | 5 | 37.8 | narrow 34.6 / 36.4 / 36.4 | 29.4 / 33.6 / 36.4 | Matches fixed at 6h; below policy |
| Counters S2/on | 4 terminal + 1 live | 32.4 | narrow 28.2 / 34.2 / 35.6 | >=22.0 / >=28.4 / >=31.4 | Live lower bound near policy |
| FO S1/off | 5 | 3.6 | normal 8.6 / 8.6 / 8.6 | 8.0 / 8.0 / 8.0 | Promising; near fixed and above policy |
| FO S1/on | 5 | 3.2 | normal 5.4 / 5.4 / 5.4 | 7.2 / 7.2 / 7.2 | Promising; above fixed mean |
| Rover S1/off | 5 | 4.0 | normal 4.8 / 4.8 / 4.8 | 4.8 / 4.8 / 4.8 | Promising; matches fixed |
| Rover S1/on | 5 | 3.6 | normal 4.6 / 4.6 / 4.6 | 4.8 / 4.8 / 4.8 | Promising; slightly above fixed mean |

Five-seed PW exact tests cannot reach a two-sided p below .0625. FO and Rover
are strong directional screening evidence, not final conventional-significance
claims. The fresh hard-30m experiment remains held-ready and should be the next
PW job family after current memory clears and the Counters tail is known.

## Research questions

### RQ1 — does MCTS-guided continued training improve VH-off policy?

| Domain | Validation-led change [95% CI]; Holm p | Terminal-led change [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -0.3 [-1.20, 0.60]; 1.000 | 0.0 [-0.95, 0.95]; 1.000 | No improvement |
| Drone | +0.8 [-1.27, 2.87]; 1.000 | +0.4 [-1.33, 2.13]; 1.000 | No reliable improvement |
| FO Counters | -1.3 [-2.37, -0.23]; .234 | -0.8 [-1.46, -0.14]; .219 | Negative raw effect; not Holm-significant |
| Rover | 0.0 [0, 0]; 1.000 | 0.0 [-0.34, 0.34]; 1.000 | No change |
| Counters | +4.4 [-17.44, 26.24]; 1.000 | +16.4 [3.09, 29.71]; .137 | Large noisy mean; not Holm-significant |

Conclusion: no original five-domain RQ1 cell survives within-RQ Holm correction.

### RQ3 — does the value head improve refinement?

The estimand is a paired difference-in-differences: `(VH-on S2−S1) −
(VH-off S2−S1)`.

| Domain | Validation-led DiD [95% CI]; Holm p | Terminal-led DiD [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -2.8 [-4.61, -0.99]; .088 | -3.4 [-5.13, -1.67]; .020 | Terminal-led VH significantly worsens refinement |
| Drone | -0.9 [-2.88, 1.08]; 1.000 | -1.1 [-3.57, 1.37]; .836 | No reliable effect |
| FO Counters | +0.7 [-0.26, 1.66]; .813 | +1.3 [0.08, 2.52]; .188 | Positive raw tendency only |
| Rover | +0.1 [-0.31, 0.51]; 1.000 | -0.1 [-0.51, 0.31]; 1.000 | No effect |
| Counters | -1.2 [-25.87, 23.47]; 1.000 | -16.8 [-29.64, -3.96]; .078 | Large negative tendency; high variance |

Conclusion: only terminal-led Block Grouping is significant after correction.

### RQ2/RQ4 — does inference-time MCTS improve the Stage-2 policy?

Primary evidence is the Stage-2 table above. The clearest complete findings are:

- VH-off/RQ2: FO Counters is strongly positive; Drone terminal is positive but
  not Holm-significant; BG terminal is negative; Rover is small; Counters
  terminal and FO/Rover validation branches remain live.
- VH-on/RQ4: both Drone branches are large and Holm-significant; BG terminal is
  negative; Rover is small; FO terminal/on and Counters terminal/on remain live.
- MPrime cannot join RQ2/RQ4 until Phase B freezes defensible selected
  checkpoints and 40 Stage-2 MCTS jobs are materialized (two branches × two VH ×
  ten seeds).

## Completed experiments and static conclusions

### Stage-1 selected policy versus preferred MCTS

| Domain/VH | Search | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | Raw / Holm p | Conclusion |
|---|---|---:|---:|---|---:|---|
| BG/off | narrow 5/20 | 16.3 | 11.6 / 14.8 / 15.4 | -0.9 [-1.88, 0.08] | .109 / .547 | No reliable benefit |
| BG/on | narrow 5/20 | 15.9 | 12.0 / 14.0 / 16.2 | +0.3 [-0.29, 0.89] | .453 / .750 | No reliable benefit |
| Drone/off | normal 20/70 | 5.9 | 6.9 / 6.9 / 6.9 | +1.0 [0.05, 1.95] | .074 / .445 | Positive but not corrected-significant |
| Drone/on | normal 20/70 | 5.1 | 10.0 / 10.4 / 10.4 | +5.3 [3.42, 7.18] | .002 / .020 | Significant gain |
| FO/off | normal 20/70 | 4.2 | 7.5 / 7.8 / 7.8 | +3.6 [2.20, 5.00] | .004 / .035 | Significant gain |
| FO/on | normal 20/70 | 3.7 | 5.3 / 5.7 / 5.7 | +2.0 [1.05, 2.95] | .004 / .035 | Significant gain |
| Rover/off | normal 20/70 | 4.0 | 4.8 / 5.0 / 5.0 | +1.0 [0.25, 1.75] | .031 / .219 | Positive; not corrected-significant |
| Rover/on | normal 20/70 | 3.8 | 4.4 / 4.4 / 4.4 | +0.6 [-0.00, 1.20] | .125 / .547 | No reliable effect |
| Counters/off | narrow 5/20 | 32.5 | 24.9 / 25.6 / 25.7 | -6.8 [-17.73, 4.13] | .250 / .750 | Search regression on average |
| Counters/on | narrow 5/20 | 18.6 | 20.3 / 22.1 / 22.5 | +3.9 [-4.33, 12.13] | .328 / .750 | Positive noisy mean |

### PRESERVE-3 validation-led

| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -0.2 [-1.01, .61] | 1.0 | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +0.4 [-.37, 1.17] | .5 | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59, 1.39] | 1.0 | Not uniformly preserved |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -0.5 [-1.63, .63] | 1.0 | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0, 0] | 1.0 | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -0.1 [-.33, .13] | 1.0 | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 scores 9/20 after a
seed-specific Stage-2 collapse. It is not a broad mild decline.

### PRESERVE-3 terminal-led

| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52, 5.12] | .496 | Preserved relative to final source |
| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91, 4.31] | .188 | Preserved/improved mean |
| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +0.4 [-2.13, 2.93] | 1.0 | Preserved on average |
| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -0.6 [-3.77, 2.57] | 1.0 | No reliable change |
| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -0.2 [-.50, .10] | .5 | Essentially preserved |
| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +0.2 [-.10, .50] | .5 | Preserved |

All 60 terminal-led lineages, all selected endpoints and the nineteen final TPP
curve-repair jobs are now Slurm-complete. No PRESERVE-3 compute remains live.

### MPrime policy extension and validation adequacy

| Branch/VH | Stage-1 | Stage-2 selected | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---|---:|---|
| Validation/off | 15.0 | 15.2 | +0.2 [-1.18, 1.58] | .880 | Provisional neutral |
| Validation/on | 14.6 | 15.2 | +0.6 [-1.46, 2.66] | .580 | Provisional neutral |
| Terminal/off | 13.5 | 14.0 | +0.5 [-1.29, 2.29] | .672 | Provisional neutral |
| Terminal/on | 13.2 | 14.5 | +1.3 [-.49, 3.09] | .172 | Positive tendency only |

These policy results are computationally complete but selector-provisional:
all 840 Stage-2 checkpoints scored 30/30 on the current validation set. Phase B
was not submitted because readiness is only 35%, below the approved 85% gate.
Two harder, independent planner-difficulty-stratified validation replicates,
checksums and a one-checkpoint preflight are still missing.

### Other completed side experiments

| Experiment | Conclusion | Provenance |
|---|---|---|
| Counters binding Horizon | Both VH modes have aware−unaware 6h change 0.0 [0,0], p=1.0; no benefit and zero logged cutoffs | `mcts_horizon_binding/horizon_completion_records_20260902.csv` plus cluster completion records |
| Drone Horizon | Same-commit aware/unaware change 0.0 [-.89,.89], p=1.0 | `mcts_horizon_determinism_causal/` |
| MCTS-SAFE-1 | Repaired 2/4 targeted Drone dead-end outcomes; useful safety guard but not a complete search-quality fix | `mcts_safe_preflight/` |
| MCTS-SAFE-CONTEXT | Negative: off -0.4 [-1.08,.28], p=.5; on -3.4 [-6.64,-.16], p=.0625 | `mcts_safe_context/live_reconciliation_20260831.csv` |
| Determinism audits | CPU-family numerical checksums differ, but audited actions and outcomes did not; timing randomness is unsupported | `mcts_determinism_audit/README.md` |

## Held designs in priority order

1. MPRIME-VAL-ADEQUACY Phase B — active preparation, not submission-ready.
2. MCTS-PW-30M — held-ready; activate after current PW tails and memory pressure clear.
3. ANCHOR-KL-CONTROL — constant versus target-KL adaptive versus scheduled KL.
4. MCTS-RESOURCE — two workers / 160 GiB resource sensitivity.
5. STOP-ORIG — original stopping-rule replication.
6. LONG-DRONE-SELECTED-MCTS — three side-experiment endpoints.
7. ACT-HISTORY-ABLATION — fresh Drone/Counters/TPP networks; user-held.
8. MCTS-PW-PATHBATCH — nonstandard multiple expansion.
9. PUCT-EST — deferred causal grid.
10. MCTS-SAFE2 — do not activate without observed horizon contamination.

These are design holds only; none occupies the Slurm pending queue.

## Provenance contract

Every aggregate score table points to a seed/job-level companion. The current
live companion contains Slurm job IDs, states, nodes, stdout paths and compact
completion-record paths. Historical policy/MCTS tables retain training and
evaluation log paths in `policy_paired_seed_results.csv`,
`mcts_paired_seed_results.csv`, `stage2_mcts_historical_log_audit_20260902.csv`
and the experiment-specific seed-result files. No aggregate in this snapshot is
intended to be an orphaned score table.


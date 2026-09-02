# Experiment status — 2 September 2026, 16:59 IDT

This is the authoritative current snapshot. Live scheduler state comes from
`cluster_workload_latest.csv`; per-job training and MCTS evidence comes from
`live_training_progress_latest.csv` and `live_mcts_progress_latest.csv`.
Static scores come only from the result ledgers named below. Every result file
either contains absolute source-log paths or is mapped to a row-level companion
in `result_provenance_index_20260902.csv`.

## Cluster workload

| State | Jobs | Requested CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 97 | 770 | 6,004 GiB |
| Pending | 503 | 4,918 | 12,203 GiB |
| Held | 0 | 0 | 0 GiB |

Pending reasons are 499 Priority, three Dependency, and one Resources. The
low-Nice PW jobs are ordinary eligible pending jobs, not held jobs.

| Experiment | Running | Pending | CPUs R/P | RAM R/P |
|---|---:|---:|---:|---:|
| MPrime terminal-led policy | 47 | 268 | 470/2,680 | 940/5,360 GiB |
| MPrime validation-led policy | 0 | 210 | 0/2,100 | 0/4,200 GiB |
| MPrime validation-led Stage 2 | 2 | 0 | 12/0 | 96/0 GiB |
| PRESERVE-3 TPP Stage 2 | 11 | 0 | 66/0 | 528/0 GiB |
| PRESERVE-3 cell controllers | 0 | 2 | 0/4 | 0/2 GiB |
| MPrime VH-on controller | 0 | 1 | 0/2 | 0/1 GiB |
| FO Counters terminal-led S2 MCTS | 20 | 0 | 120/0 | 2,400/0 GiB |
| Counters binding Horizon | 8 | 0 | 48/0 | 960/0 GiB |
| PW70 two-seed correction | 7 | 0 | 42/0 | 840/0 GiB |
| PW70 five-seed confirmation | 2 | 16 | 12/96 | 240/1,920 GiB |
| Counters PW divergence recovery | 0 | 6 | 0/36 | 0/720 GiB |

## Live training and policy pipeline

| Experiment/cell | Terminal | Running | Policy state | Estimate |
|---|---:|---:|---|---|
| MPrime validation-led S2 off | 10/10 | 0 | 210 every-five/endpoints pending | evaluations minutes to hours; starts depend on priority |
| MPrime validation-led S2 on | 8/10 | 2 | controller 20863176 depends only on this cell | epochs97/98; about0.4--0.7h by current rate |
| MPrime terminal-led S2 | 20/20 | 0 | 105/420 left queue,47 running,268 pending | evaluations minutes to hours; no defensible queue-start time |
| PRESERVE-3 Delivery | 20/20 | 0 | curve/endpoint compute finished | static reconciliation only |
| PRESERVE-3 TPP off | 7/10 | 3 | cell controller dependency-pending | epochs65/86/91; roughly5--18h |
| PRESERVE-3 TPP on | 2/10 | 8 | cell controller dependency-pending | epochs57--98; most1--18h, slowest likely reaches72h in about27h |
| PRESERVE-3 Zenotravel | 20/20 | 0 | curve/endpoint compute finished | done |

The MPrime and TPP controllers are VH-cell-specific: one VH mode cannot block
the other. There is no newly available training/policy work absent from its
manifest or dependency chain.

## Completed mainstream policy experiments

All cells have ten paired seeds. These compare the Stage-1 source used by each
branch with that branch's validation-selected Stage-2 endpoint.

| Domain/VH | MAIN-VAL S1 selected -> S2 selected | Change [95% CI] | raw/Holm p | MAIN-TERM S1 final -> S2 selected | Change [95% CI] | raw/Holm p |
|---|---:|---|---:|---:|---|---:|
| Block Grouping/off | 16.3 -> 16.0 | -.3 [-1.20,.60] | .625/1 | 16.3 -> 16.3 | 0 [-.95,.95] | 1/1 |
| Block Grouping/on | 15.9 -> 12.8 | -3.1 [-5.16,-1.04] | .0156/.156 | 15.0 -> 11.6 | -3.4 [-5.25,-1.55] | .0078/.078 |
| Drone/off | 5.9 -> 6.7 | +.8 [-1.27,2.87] | .504/1 | 7.4 -> 7.8 | +.4 [-1.33,2.13] | .688/1 |
| Drone/on | 5.1 -> 5.0 | -.1 [-1.34,1.14] | 1/1 | 7.2 -> 6.5 | -.7 [-1.77,.37] | .250/1 |
| FO Counters/off | 4.2 -> 2.9 | -1.3 [-2.37,-.23] | .0469/.422 | 3.6 -> 2.8 | -.8 [-1.46,-.14] | .0547/.438 |
| FO Counters/on | 3.7 -> 3.1 | -.6 [-1.50,.30] | .250/1 | 3.3 -> 3.8 | +.5 [-.41,1.41] | .344/1 |
| Rover/off | 4.0 -> 4.0 | 0 [0,0] | 1/1 | 3.8 -> 3.8 | 0 [-.34,.34] | 1/1 |
| Rover/on | 3.8 -> 3.9 | +.1 [-.31,.51] | 1/1 | 4.1 -> 4.0 | -.1 [-.33,.13] | 1/1 |
| Counters/off | 32.5 -> 36.9 | +4.4 [-17.44,26.24] | .660/1 | 21.5 -> 37.9 | +16.4 [3.09,29.71] | .0273/.246 |
| Counters/on | 18.6 -> 21.8 | +3.2 [-6.15,12.55] | .563/1 | 17.2 -> 16.8 | -.4 [-11.44,10.64] | .961/1 |

No Stage-1-to-Stage-2 policy change survives Holm correction across ten cells.
Source: `experiment_statistics.csv` and `experiment_results.csv`.

## Early Stage-2 update audit

Epoch0 is after one full Stage-2 explore/replay/optimizer update. Complete
terminal-led ten-seed evidence:

| Domain/VH | Stage-1 final -> S2 epoch0 | Change [95% CI] | raw/Holm p | Seeds losing >=5 |
|---|---:|---|---:|---:|
| Block Grouping/off | 16.3 -> 15.7 | -.6 [-2.36,1.16] | .539/1 | 1 |
| Block Grouping/on | 15.0 -> 10.3 | -4.7 [-6.91,-2.49] | .0059/.0586 | 7 |
| Drone/off | 7.4 -> 6.3 | -1.1 [-3.19,.99] | .328/1 | 1 |
| Drone/on | 7.2 -> 6.0 | -1.2 [-2.01,-.39] | .0313/.281 | 0 |
| FO Counters/off | 3.6 -> 2.4 | -1.2 [-2.31,-.09] | .0625/.500 | 0 |
| FO Counters/on | 3.3 -> 3.3 | 0 [-1.01,1.01] | 1/1 | 0 |
| Rover/off | 3.8 -> 3.7 | -.1 [-.51,.31] | 1/1 | 0 |
| Rover/on | 4.1 -> 4.0 | -.1 [-.33,.13] | 1/1 | 0 |
| Counters/off | 21.5 -> 22.4 | +.9 [-10.71,12.51] | .836/1 | 3 |
| Counters/on | 17.2 -> 10.3 | -6.9 [-16.69,2.89] | .164/1 | 5 |

This establishes broader early-update instability, especially Block Grouping
VH-on. Exact seed/log provenance: `stage2_epoch0_seed_audit_20260902.csv`.
Interpretation and held ANCHOR-SCHEDULE design:
`stage2_early_update_audit_20260902.md`.

## PRESERVE-4 validation-led policy — complete

| Domain/VH | S1 selected | S2 selected all10 | held-out8 | tuning2 | Change [95% CI] | raw/Holm p |
|---|---:|---:|---:|---:|---|---:|
| Delivery/off | 19.8 | 19.6 | 19.5 | 20.0 | -.2 [-1.01,.61] | 1/1 |
| Delivery/on | 19.2 | 19.6 | 19.5 | 20.0 | +.4 [-.37,1.17] | .5/1 |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | 1/1 |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | 1/1 |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | 1/1 |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | 1/1 |

`*` Nine TPP/off seeds score20/20; seed1972442430 scores9/20. Its Stage-1
source was20/20 and Stage-2 epoch0 was10/20. All nine plans are VAL-valid; the
eleven failures exhaust10,000 actions. See
`tpp_stage2_vh_off_regression_audit_20260902.md`.

## Stage-1 validation-selected policy versus MCTS — complete

Every MCTS score shows counterfactual per-instance limits 30m/2h/6h.

| Domain/VH | Search | n | Policy | MCTS 30m/2h/6h | 6h change [95% CI] | raw/Holm p |
|---|---|---:|---:|---:|---|---:|
| Block Grouping/off | narrow5/20 | 10 | 16.3 | 11.6/14.8/15.4 | -.9 [-1.88,.08] | .109/.547 |
| Block Grouping/on | narrow5/20 | 10 | 15.9 | 12.0/14.0/16.2 | +.3 [-.29,.89] | .453/.750 |
| Drone/off | normal20/70 | 10 | 5.9 | 6.9/6.9/6.9 | +1.0 [.05,1.95] | .074/.445 |
| Drone/on | normal20/70 | 10 | 5.1 | 10.0/10.4/10.4 | +5.3 [3.42,7.18] | .002/.020 |
| FO Counters/off | normal20/70 | 10 | 4.2 | 7.5/7.8/7.8 | +3.6 [2.20,5.00] | .004/.035 |
| FO Counters/on | normal20/70 | 10 | 3.7 | 5.3/5.7/5.7 | +2.0 [1.05,2.95] | .004/.035 |
| Rover/off | normal20/70 | 10 | 4.0 | 4.8/5.0/5.0 | +1.0 [.25,1.75] | .031/.219 |
| Rover/on | normal20/70 | 10 | 3.8 | 4.4/4.4/4.4 | +.6 [-.00,1.20] | .125/.547 |
| Counters/off | narrow5/20 | 10 | 32.5 | 24.9/25.6/25.7 | -6.8 [-17.73,4.13] | .250/.750 |
| Counters/on | narrow5/20 | 10 | 18.6 | 20.3/22.1/22.5 | +3.9 [-4.33,12.13] | .328/.750 |

## Stage-2 MCTS availability and results

| Domain/VH | Validation terminal/live/missing | Terminal terminal/live/missing | Search |
|---|---:|---:|---|
| Block Grouping/off | 0/0/10 | 10/0/0 | narrow5/20 |
| Block Grouping/on | 8/0/2 | 9/0/1 | narrow5/20 |
| Drone/off | 10/0/0 | 10/0/0 | normal20/70 |
| Drone/on | 10/0/0 | 10/0/0 | normal20/70 |
| FO Counters/off | 6/0/4 | 0/10/0 | normal20/70 |
| FO Counters/on | 9/0/1 | 0/10/0 | normal20/70 |
| Rover/off | 1/0/9 | 10/0/0 | normal20/70 |
| Rover/on | 0/0/10 | 10/0/0 | normal20/70 |
| Counters/off | 10/0/0 | 0/0/10 | narrow5/20 |
| Counters/on | 10/0/0 | 0/0/10 | narrow5/20 |

The unresolved gap is exactly13 Block Grouping +5 FO Counters +19 Rover +20
Counters =57 jobs. Causal history: `stage2_mcts_gap_explanation_20260902.md`.

| Domain/VH/branch | n | Policy | MCTS 30m/2h/6h | 6h change [95% CI] | raw p |
|---|---:|---:|---:|---|---:|
| BG/off terminal | 10 | 16.3 | 4.4/12.3/14.5 | -1.8 [-3.22,-.38] | .031 |
| BG/on terminal | 10 aggregate;9 exact logs | 11.6 | 4.4/8.7/9.4 | -2.2 [-4.60,.20] | .031 |
| Drone/off validation | 10 | 6.7 | 7.5/7.7/7.7 | +1.0 [-.43,2.43] | .195 |
| Drone/on validation | 10 | 5.0 | 10.9/11.2/11.2 | +6.2 [4.33,8.07] | .002 |
| Drone/off terminal | 10 | 7.8 | 9.7/9.8/9.9 | +2.1 [.65,3.55] | .020 |
| Drone/on terminal | 10 | 6.5 | 12.9/13.1/13.1 | +6.6 [5.47,7.73] | .002 |
| FO/off terminal | 0 terminal;10 live | 2.8 | n/a/>=6.3/>=6.3 | retained lower bound | n/a |
| FO/on terminal | 0 terminal;10 live | 3.8 | >=4.4/>=4.4/>=4.4 | live lower bound | n/a |
| Rover/off terminal | 10 | 3.8 | 4.2/4.2/4.2 | +.4 [.03,.77] | .125 |
| Rover/on terminal | 10 | 4.0 | 4.5/4.5/4.5 | +.5 [.12,.88] | .063 |
| Counters/off validation | 10 | 36.9 | 34.9/36.7/36.7 | -.2 [-2.87,2.47] | .969 |
| Counters/on validation | 10 | 21.8 | 22.6/26.4/27.1 | +5.3 [-.70,11.30] | .082 |

FO live stdout currently contains97 successes after the outage restart (off53,
on44) and107 timeout messages. Pre-outage records retain at least63 off-mode
successes; these are lower bounds, not terminal aggregates. Comparable jobs
historically take median48.3h (range30.1--72h);
current allocations have run6--18h, so roughly12--54h remains.

## Progressive widening

PW20 and PW70 are never pooled. Fixed is narrow5/20 for Block Grouping/Counters
and normal20/70 for FO Counters/Rover.

| Cell | Policy | Fixed 30m/2h/6h | PW20 30m/2h/6h | PW70 30m/2h/6h | PW70 state |
|---|---:|---:|---:|---:|---|
| BG S1/off | 16.5/20 | 11.0/13.5/15.0 | 11.5/15.0/15.0 | 10.5/11.5/13.0 | terminal2/2 |
| BG S1/on | 17.0/20 | 11.5/13.5/17.0 | 12.5/17.0/18.0 | 9.0/11.5/13.5 | terminal2/2 |
| Counters S1/off | 18.0/59 | 21.5/21.5/21.5 | 20.5/20.5/20.5 | >=21/>=21/>=21 | 2 live |
| Counters S1/on | 5.0/59 | 13.5/13.5/13.5 | 12.5/12.5/12.5 | >=12/>=12.5/>=12.5 | 2 live |
| Counters S2/off | 49.0/59 | >=37/>=41.5/44.5 | 46.0/49.5/49.5 | >=34/>=39.5/>=43.5 | 1 terminal,1 live |
| Counters S2/on | 5.0/59 | 16.5/17.5/17.5 | 15.0/15.0/15.0 | >=13.5/>=14/>=14 | 2 live |
| FO S1/off | 3.5/20 | 9.0/9.5/9.5 | not run | 9.5/9.5/9.5 | screen terminal |
| FO S1/on | 3.5/20 | 6.0/7.5/7.5 | not run | 8.0/8.0/8.0 | screen terminal |
| Rover S1/off | 4.0/20 | 4.0/4.5/4.5 | not run | 5.0/5.0/5.0 | screen terminal |
| Rover S1/on | 4.0/20 | 4.5/4.5/4.5 | not run | 5.5/5.5/5.5 | screen terminal |

Five-seed confirmation adds three seeds to six cells (18 jobs): two Counters
S2/off run and16 are low-priority pending. Current two-job lower bounds total
46/56/57 successes at30m/2h/6h across61 classified records. The approved
six-job exact-snapshot Counters divergence recovery is eligible low-priority
pending. Running correction jobs have34--48h hard bounds remaining; a precise
completion time is not defensible.

## Counters binding Horizon

Eight aware/unaware jobs run. Across138 completed instance records, all138 are
successes:131 by30m,135 by2h,138 by6h. Longest completed plan:1,105 actions.
No completed long failure has tested the boundary, and all cutoff summaries are
zero. Jobs have run26--36h; hard bounds leave roughly36--45h.

External action count is the evaluator's outer step loop. Search depth is
`len(path)-1` inside one root search. A long trajectory can consist of many
shallow searches; if an aware run reaches action9,998 and a search reaches
depth2, the tested counter must become nonzero. Exact records and original
paths: `mcts_horizon_binding/horizon_completion_records_20260902.csv`.

## Held experiment designs

| Experiment | Why held | Activation criterion |
|---|---|---|
| MCTS-PW-30M | wait for PW70 screen | fresh hard30m confirmation on qualifying cells |
| ANCHOR-SCHEDULE | schedule/protocol not frozen | small first-update diagnostic then held-out confirmation |
| MCTS-PW-PATHBATCH | nonstandard multi-expansion design | implement after standard PW |
| MCTS-SAFE2 | no audited horizon contamination; memory risk | nonzero cutoffs + horizon-dependent reuse + changed decision |
| ACT-HISTORY-ABLATION | new network shape/full training | held Drone/Counters/TPP pilot |
| MCTS-RESOURCE | resource sensitivity | lifecycle deployment and explicit priority |
| STOP-ORIG | separate stopping-rule replication | after live mainstream work |
| PUCT-EST | separate causal grid | explicitly deferred |

No held design currently has Slurm jobs. Completed/negative diagnostic
experiments remain registered in `experiments.csv`; they are not live work.

## Provenance contract

- `experiment_results.csv`: direct training, evaluation and VAL log paths.
- `stage1_mcts_results.csv`: source log, completion ledger and validation log.
- `stage2_mcts_historical_log_audit_20260902.csv`: all200 intended Stage-2
  identities, including explicit absence.
- `stage2_epoch0_seed_audit_20260902.csv`: first-update scores and four absolute
  source paths per row.
- `mcts_horizon_binding/horizon_completion_records_20260902.csv`: job, instance,
  signature, stdout and completion-ledger paths.
- `result_provenance_index_20260902.csv`: companion mapping for aggregate files.

`result_provenance_audit_20260902.csv` is regenerated after every authoritative
snapshot. It currently finds90 score-bearing CSVs:73 carry direct provenance,
17 use an explicit companion mapping, and zero need mapping. Any future unmapped
result file is a release blocker.

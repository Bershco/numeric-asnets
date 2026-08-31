# Overnight pipeline status — 31 August 2026, 03:50 IDT

This is the authoritative interpretation of the replaceable Slurm snapshot in
`../live_jobs.csv`. Static per-job evidence remains in the experiment-specific
CSV ledgers; this report does not infer scores from job names.

## What progressed automatically

- MPrime anchor tuning reached 28/28 terminal lineages. Finalizer `20755820`
  verified the evidence and selected anchor coefficient **0 for both VH modes**.
- The MPrime validation-led Stage-2 extension now contains four reused tuning
  winners plus sixteen new held-out lineages. New jobs `20760292--20760307` are
  submitted and resource-pending.
- The MPrime terminal-led Stage-2 extension contains twenty distinct lineages
  starting from corrected Stage-1 final checkpoints. Jobs
  `20760308--20760327` are submitted and resource-pending.
- PRESERVE-3-TERM started: seventeen Delivery jobs are running and forty-three
  Delivery/TPP/Zenotravel jobs are ordinarily resource-pending.
- Five small dependency controllers were attached. They submit **policy only**:
  every-five learning curves plus validation-selected and final Stage-2 policy
  endpoints. They do not submit MCTS.
- No new MCTS experiment was submitted or released during this heartbeat. One
  already-submitted MAIN-VAL Drone Stage-2 MCTS job started when memory freed.

## Cluster workload

| State | Jobs | Requested CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 63 | 378 | 6,120 GiB |
| Pending, including dependencies and holds | 140 | 816 | 10,398 GiB |

The running allocation is 24 GiB below the 6-TiB per-user memory ceiling.
Consequently, almost every eligible heavy job waits on `QOSMaxMemoryPerUser`.
The twenty FO Counters jobs are deliberately held and consume no allocated RAM.
The six dependency controllers request only 2 CPUs and 1 GiB each when they run.

## Live and eligible experiments

| Experiment | Running | Pending | Running resources | Current position | Timing |
|---|---:|---:|---:|---|---|
| PRESERVE-3-TERM | 17 | 43 | 102 CPU / 816 GiB | Delivery final-Stage1-led Stage 2 has begun; TPP and Zenotravel wait for memory | Most Delivery jobs are at epochs 8--21 after about 3h. A rough 15--35h completion window is more realistic than the 72h hard cap, but early stopping can shorten it |
| PRESERVE-3-TERM policy controllers | 0 | 3 dependency | 0 | One controller per domain; IDs `20760689--20760691` | Each releases after all twenty training jobs for its domain become terminal |
| MPrime validation-led Stage 2 | 0 | 16 | 0 | Four tuning winners are reused; sixteen held-out jobs are queued | Start time depends on memory; 72h hard limit after start |
| MPrime terminal-led Stage 2 | 0 | 20 | 0 | Twenty final-checkpoint lineages are queued | Same scheduling constraint and 72h hard limit |
| MPrime policy controllers | 0 | 2 dependency | 0 | `20760692` covers the sixteen new validation-led jobs; `20760693` covers all twenty terminal-led jobs | Automatic after the corresponding training branch terminates |
| TPP validation-led Stage 2 | 3 | 0 | 18 CPU / 144 GiB | Thirteen held-out lineages are terminal; three remain at epochs about 98, 83, and 60 | Approximately 1--2h, about 14h, and scheduler-limited in about 19h respectively |
| TPP policy controller | 0 | 1 dependency | 0 | Existing controller `20743985` is still valid | Releases after the last three lineages terminate |
| Counters Stage-2 narrow 5/20 | 5 | 0 | 30 CPU / 600 GiB | Ten completed, five OOM terminal, five running; fourteen terminal rows currently reconciled | Running jobs have about 6h53m before their 72h allocation limits |
| Cross-domain PW20/PW70 screen | 15 | 12 | 90 CPU / 1,800 GiB | PW20 is 5/20 terminal; PW70 correction is queued | Running tails range from about 1.4h to 14.6h; 72h hard cap, no defensible precise finish time yet |
| Counters binding-Horizon | 0 | 8 | 0 | Fresh same-commit aware/unaware pairs at the real 10,000-action cap | Eligible but memory-pending; 72h hard limit after start |
| SAFE-CONTEXT | 1 | 0 | 6 CPU / 120 GiB | Nineteen of twenty arms terminal; final contextual arm has run about 18h | Likely hours, but timeout tails can extend to 72h |
| MAIN-VAL Drone Stage-2 MCTS gap | 1 | 15 | 6 CPU / 120 GiB | One of the previously submitted sixteen missing comparisons has started | Drone usually terminates well before 72h; exact finish remains instance-dependent |
| Rover endpoint MCTS | 21 | 0 | 126 CPU / 2,520 GiB | All declared Rover endpoints are now running | 4.6--14.2h elapsed; historical allocations often ran roughly 30h; 72h cap |

## New comparative evidence

### SAFE-CONTEXT — provisional negative result

Nine pairs are terminal. Contextual nodes use physical state plus action-history
digest for node statistics and cached predictions; the physical arm uses the
original transposition identity.

| VH | Matched n | Physical mean | Contextual mean | Paired change | 95% paired t CI | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| Off | 4 | 9.25/20 | 9.00/20 | -0.25 | [-1.05, 0.55] | 1.0000 |
| On | 5 | 10.80/20 | 7.40/20 | -3.40 | [-6.64, -0.16] | .0625 |
| Combined | 9 | 10.11/20 | 8.11/20 | -2.00 | [-3.92, -0.08] | .03125 |

The combined exact test is significant before any multiple-comparison
correction, but the result remains provisional until the tenth pair terminates.
The effect is concentrated in VH-on. The current contextual-node design reduces
transposition sharing enough to hurt coverage; this argues against promoting it
as the default even though it removes action-history aliasing.

### Cross-domain PW20 — five terminal jobs

The Block Grouping/Counters rows below use **20 simulations** and must remain
separate from the queued PW70 correction.

| Cell / seed set | Policy | Fixed narrow 5/20 | PW20 | Fixed runtime | PW20 runtime | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Block Grouping S1/off, two seeds | 16.5/20 | 15.0/20 | 15.0/20 | 7h39m | 12h26m | Unpromising: worse than policy, no coverage gain over fixed narrow, and slower |
| Block Grouping S1/on, two seeds | 17.0/20 | 17.0/20 | 18.0/20 | 8h38m | 9h22m | Coverage-positive but slower; PW70/retained-node evidence decides whether it is worth expanding |
| Counters S2/off, one seed | 59/59 | 49/59 | 59/59 | 18h05m | 3h47m | Strong screening result, but one seed is not confirmatory |

### Delivery validation-led Stage-2 policy evidence

The authoritative held-out ledger now contains all 346 materialized policy
evaluations and all are terminal/scored. For the sixteen held-out confirmation
lineages, validation-selected means are 19.5/20 in both VH modes; final means
are 18.25/off and 17.5/on. The four reused tuning-winner lineages are tracked
separately and must be integrated before claiming the final twenty-seed
confirmation mean. This separation prevents tuning evidence from being
mistaken for held-out evidence.

## Deliberately held or design-held

| Experiment | Scope | Why held / release condition |
|---|---:|---|
| FO Counters endpoint MCTS | 20 Slurm-held jobs | Preserve memory for mainstream/PW/Horizon work; release only when reprioritized |
| MCTS-PW-30M | Manifest-ready design | Wait for qualifying cross-domain cells; then run a fresh hard-30-minute comparison |
| MCTS-RESOURCE | Two workers / 160 GiB FO Counters and Rover resumptions | Separate resource-sensitivity result after current prioritized work |
| SAFE-2 | Design-held | User explicitly deferred horizon-indexed `(state,h)` statistics |
| PW path-batched widening | Design-held | Needs separately frozen multi-expansion/backpropagation semantics |
| STOP-ORIG and PUCT/estimator sensitivity | Design-held | Not current resource priorities |

## Policy coverage contract

Every currently available Stage-1/Stage-2 lineage retains an every-five policy
curve plus selected/final policy endpoints. Newly submitted controllers use ten
workers, ten CPUs, 20 GiB, and four hours per policy evaluation. The new
controllers are dependency-gated by complete domain/branch training ledgers,
therefore they cannot evaluate incomplete checkpoints. Stage-2-final MCTS
remains intentionally excluded; validation-selected Stage-2 MCTS remains the
declared comparison endpoint.

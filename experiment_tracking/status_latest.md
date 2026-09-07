# Complete experiment snapshot — 2026-09-07T16:53:24+03:00

This is the updated version of the 6 September report. It incorporates a fresh
Slurm query and bounded reads of every live MCTS log. The MPrime Phase-B repair
and rescore gate and the four-arm TPP catastrophic-seed diagnostic were
submitted during this refresh.

## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 17 | 100 | 1,940 GiB |
| Pending | 60 | 180 requested; 36 concurrency-capped | 1,200 GiB requested; 240 GiB concurrency-capped |
| Held in Slurm | 0 | 0 | 0 GiB |

| Live experiment | Running | Pending | Running CPU / RAM | Current position | Timing |
|---|---:|---:|---:|---|---|
| MPrime Phase-B validation adequacy | 1 | 60 | 4 CPUs / 20 GiB | Corrected one-checkpoint preflight running; 60 lineages dependency-gated, max 12 concurrent | Preflight ≤1h02m; full duration becomes estimable after first wave |
| TPP catastrophic-seed MCTS | 4 | 0 | 24 CPUs / 480 GiB | Fixed normal, fixed narrow, PW20 and PW70 all started on the exact 9/20 checkpoint | Healthy startup; likely first results within hours, hard cap 72h |
| Stage-2 MCTS: Counters terminal-led | 6 | 0 | 36 CPUs / 720 GiB | 14/20 scheduler-terminal; six live | Current allocations have 20h36m–20h49m hard bounds |
| Stage-2 MCTS: FO terminal-led tail | 1 | 0 | 6 CPUs / 120 GiB | 19/20 scheduler-terminal; final seed at 5/20 | ≤44h12m |
| PW70 two-seed correction | 1 | 0 | 6 CPUs / 120 GiB | 11/12 terminal; Counters S1/off tail at 21/59 | ≤45h08m |
| PW70 ten-seed FO/Rover expansion | 3 | 0 | 18 CPUs / 360 GiB | 37/40 scheduler-terminal; FO/off final seed OOMed at 7/20 after ten classified, leaving n=9 complete | ≤43h30m; no reliable earlier straggler estimate |
| Counters exact-snapshot PW70 recovery | 1 | 0 | 6 CPUs / 120 GiB | Two PW70 arms terminal; last arm at 22/59 | ≤37h42m |

The queue is no longer resource-saturated: requested live RAM fell from 5,048
GiB yesterday to 1,800 GiB. There is room to release more work, but this pass
deliberately makes no submission so that the next choice can be reviewed first.

## Changes since 6 September

- One Counters Stage-2/on job ended OOM after 31/59 successful, VAL-valid
  plans and 49 classified instances. Its evidence remains a lower bound.
- FO validation-led/on is no longer live. Its final job timed out after 6/20;
  five successes were within 30 minutes and all six within two hours.
- MPrime's unsupported validation-tier mapping was repaired without changing
  the frozen manifests or checksums. Preflight 21084274 is running and the
  60-lineage array 21084275 will release only if it succeeds.
- The TPP/off 9/20 outlier now has four nonredundant diagnostic arms running:
  fixed 20/70, fixed 5/20, PW20 and PW70.
- The 20-job PW70 FO/Rover expansion is now 37/40 scheduler-terminal with three
  jobs live. FO/off seed 2082152039 ended OOM at 7/20 after ten classified
  instances, so its completed inferential subset remains n=9 rather than
  treating unknown instances as ordinary failures.
- The 57-job Stage-2 branch-completion campaign advanced to 49
  scheduler-terminal identities and eight live identities. Scheduler-terminal
  does not automatically mean a complete scientific run: several FO/Counters
  jobs ended with timeout, OOM or application failure after producing partial
  valid evidence.
- The older failed MPrime preflight and dependency-cancelled array remain in
  provenance only; no expensive scoring was lost.

## Live experiment results

### Stage-1 validation-selected policy versus MCTS — complete

All rows use ten matched seeds. The inferential family is the declared six-hour
result; 30-minute and two-hour columns are deterministic reclassifications.

| Domain/VH | Search | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | Raw / Holm p | Conclusion |
|---|---|---:|---:|---|---|---|
| BG/off | Narrow 5/20 | 16.3 | 11.6 / 14.8 / 15.4 | −0.9 [−1.88, 0.08] | .109 / .547 | No reliable gain |
| BG/on | Narrow 5/20 | 15.9 | 12.0 / 14.0 / 16.2 | +0.3 [−0.29, 0.89] | .453 / .750 | Neutral |
| Drone/off | Normal 20/70 | 5.9 | 6.9 / 6.9 / 6.9 | +1.0 [0.05, 1.95] | .074 / .445 | Positive, not Holm-significant |
| Drone/on | Normal 20/70 | 5.1 | 10.0 / 10.4 / 10.4 | +5.3 [3.42, 7.18] | .002 / .020 | **Large significant gain** |
| FO/off | Normal 20/70 | 4.2 | 7.5 / 7.8 / 7.8 | +3.6 [2.20, 5.00] | .004 / .035 | **Significant gain** |
| FO/on | Normal 20/70 | 3.7 | 5.3 / 5.7 / 5.7 | +2.0 [1.05, 2.95] | .004 / .035 | **Significant gain** |
| Rover/off | Normal 20/70 | 4.0 | 4.8 / 5.0 / 5.0 | +1.0 [0.25, 1.75] | .031 / .219 | Small non-significant gain |
| Rover/on | Normal 20/70 | 3.8 | 4.4 / 4.4 / 4.4 | +0.6 [−0.00, 1.20] | .125 / .547 | Neutral-positive |
| Counters/off | Narrow 5/20 | 32.5 | 24.9 / 25.6 / 25.7 | −6.8 [−17.73, 4.13] | .250 / .750 | Negative mean; severe seed regressions |
| Counters/on | Narrow 5/20 | 18.6 | 20.3 / 22.1 / 22.5 | +3.9 [−4.33, 12.13] | .328 / .750 | Positive but highly variable |

Visible conclusion: Stage-1 inference-time MCTS is convincingly beneficial for
Drone/on and both FO Counters modes, neutral for Rover, and unsafe as a blanket
replacement in Block Grouping/Counters.

### Stage-2 policy versus MCTS — every domain and both branches

The MCTS columns are recorded 30-minute, 2-hour and 6-hour per-instance
cutoffs. `≥` marks a live or partially interrupted lower bound. Confidence
intervals and p-values are shown only when all ten matched fixed-budget scores
are scientifically usable.

| Domain/VH | Branch | Search | Terminal/live | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | Raw / Holm p | Conclusion |
|---|---|---|---:|---:|---:|---|---|---|
| BG/off | Validation | Narrow 5/20 | 10/0 | 16.0 | 11.4 / 15.0 / 15.7 | −0.3 [−0.89, 0.29] | .500 / family pending | Neutral at 6h; 30m too short |
| BG/on | Validation | Narrow 5/20 | 10/0 | 12.8 | 10.1 / 10.6 / 12.6 | −0.2 [−1.08, 0.68] | .813 / family pending | Neutral at 6h |
| BG/off | Terminal | Narrow 5/20 | 10/0 | 16.3 | 4.4 / 12.3 / 14.5 | −1.8 [−3.22, −0.38] | .031 / .219 | Negative; not Holm-significant |
| BG/on | Terminal | Narrow 5/20 | 10/0 | 11.6 | 4.4 / 8.7 / 9.4 | −2.2 [−4.60, 0.20] | .031 / .219 | Negative mean |
| Drone/off | Validation | Normal 20/70 | 10/0 | 6.7 | 7.5 / 7.7 / 7.7 | +1.0 [−0.43, 2.43] | .195 / .391 | Positive; not significant |
| Drone/on | Validation | Normal 20/70 | 10/0 | 5.0 | 10.9 / 11.2 / 11.2 | +6.2 [4.33, 8.07] | .002 / .020 | **Large significant gain** |
| Drone/off | Terminal | Normal 20/70 | 10/0 | 7.8 | 9.7 / 9.8 / 9.9 | +2.1 [0.65, 3.55] | .020 / .156 | Positive; not Holm-significant |
| Drone/on | Terminal | Normal 20/70 | 10/0 | 6.5 | 12.9 / 13.1 / 13.1 | +6.6 [5.47, 7.73] | .002 / .020 | **Large significant gain** |
| FO/off | Validation | Normal 20/70 | 10/0 | 2.9 | ≥6.0 / ≥6.0 / ≥6.0 | ≥+3.1 | Withheld | Positive lower bound; two failed partial runs prevent final inference |
| FO/on | Validation | Normal 20/70 | 10/0, one timeout-partial | 3.1 | ≥5.3 / ≥5.4 / ≥5.4 | ≥+2.3 | Withheld | Positive lower bound; final seed ended 6/20 |
| FO/off | Terminal | Normal 20/70 | 10/0 | 2.8 | 5.3 / 5.3 / 5.3 | +2.5 [1.73, 3.27] | .002 / .010 | **Strong significant gain** |
| FO/on | Terminal | Normal 20/70 | 9/1 | 3.8 | ≥4.4 / ≥4.4 / ≥4.4 | ≥+0.6 | Withheld | Positive lower bound; one live seed |
| Rover/off | Validation | Normal 20/70 | 10/0 | 4.0 | 4.5 / 4.5 / 4.5 | +0.5 [−0.01, 1.01] | .125 / family pending | Small non-significant gain |
| Rover/on | Validation | Normal 20/70 | 10/0 | 3.9 | 4.4 / 4.5 / 4.5 | +0.6 [−0.09, 1.29] | .156 / family pending | Small non-significant gain |
| Rover/off | Terminal | Normal 20/70 | 10/0 | 3.8 | 4.2 / 4.2 / 4.2 | +0.4 [0.03, 0.77] | .125 / .375 | Small non-significant gain |
| Rover/on | Terminal | Normal 20/70 | 10/0 | 4.0 | 4.5 / 4.5 / 4.5 | +0.5 [0.12, 0.88] | .063 / .313 | Small non-significant gain |
| Counters/off | Validation | Narrow 5/20 | 10/0 | 36.9 | 34.9 / 36.7 / 36.7 | −0.2 [−2.87, 2.47] | .969 / .969 | No meaningful change |
| Counters/on | Validation | Narrow 5/20 | 10/0 | 21.8 | 22.6 / 26.4 / 27.1 | +5.3 [−0.70, 11.30] | .082 / .328 | Positive but variable |
| Counters/off | Terminal | Narrow 5/20 | 6/4 | 37.9 | ≥34.9 / ≥37.8 / ≥38.4 | ≥+0.5 | Withheld | Current lower bound slightly exceeds policy |
| Counters/on | Terminal | Narrow 5/20 | 8/2 | 16.8 | ≥19.9 / ≥22.0 / ≥22.4 | ≥+5.6 | Withheld | Encouraging lower bound; newest OOM seed contributed 31/59 |

Visible conclusion: inference-time MCTS is already convincingly helpful for
Drone/VH-on and FO/off, while Block Grouping terminal-led search is harmful at
the declared narrow budget. Counters terminal-led looks promising but remains
live. FO validation-led and terminal/on need interruption-aware completion
before final inferential claims.

### PW70 ten-seed FO/Rover extension — provisional n=9 per cell

These rows use only the nine terminal matched seeds in each cell. They are not
final ten-seed claims and can be affected by the identity of the remaining
straggler. The p-values are unadjusted exact paired sign-flip tests.

| Cell | Policy | Fixed normal 20/70 | PW70 30m / 2h / 6h | PW−policy [95% CI]; p | PW−fixed [95% CI]; p | Current conclusion |
|---|---:|---:|---:|---|---|---|
| FO/off | 4.11 | 7.78 | 8.56 / 8.56 / 8.56 | +4.44 [3.58, 5.31]; .0039 | +0.78 [−0.65, 2.20]; .328 | Strong policy gain; indistinguishable from fixed search |
| FO/on | 3.56 | 5.56 | 7.44 / 7.44 / 7.44 | +3.89 [3.18, 4.60]; .0039 | +1.89 [0.91, 2.86]; .0156 | Strong provisional gain over both comparators |
| Rover/off | 4.00 | 5.11 | 4.78 / 4.78 / 4.78 | +0.78 [0.03, 1.52]; .125 | −0.33 [−1.55, 0.88]; .688 | Improves mean over policy; no fixed-search advantage |
| Rover/on | 3.78 | 4.44 | 4.56 / 4.67 / 4.67 | +0.89 [−0.09, 1.86]; .125 | +0.22 [−0.92, 1.36]; .828 | Rough fixed-search parity; modest policy gain |

Visible conclusion: FO Counters is the clearly promising PW70 domain. Rover is
at best a parity/efficiency story. Three final seeds are live; the fourth,
FO/off seed 2082152039, ended OOM with only ten instances classified. Therefore
the final ten-seed table and Holm family still cannot be frozen.

### PW20 and PW70 interpretation

- Drone's completed Kmin-3 extension used eight matched seeds across both VH
  modes. Mean policy coverage was 7.0/20, fixed top-20 MCTS was 10.5/20, and
  PW was 9.5/20. Conditional on success, fixed-search runtime had median 130 s,
  p90 893 s and maximum 6,127 s; PW had median 94 s, p90 318 s and maximum
  594 s. At a post-hoc 30-minute cutoff fixed scored 10.25 and PW 9.5.
  Therefore PW delivered a substantial tail-runtime and memory reduction and
  improved over policy, but failed the primary non-inferiority target against
  fixed search.
- The original mixed cross-domain screen used PW20 for Block Grouping/Counters
  and PW70 for FO/Rover. Those arms remain separate.
- The corrected two-seed PW70 screen is complete except Counters S1/off seed
  2011206605, currently at ≥21/59.
- In the five-seed Counters Stage-2 screen, PW70 matched fixed narrow at 6h for
  VH-off (36.4/59) but remained below the policy mean (37.8/59). VH-on PW70 was
  below both policy and fixed narrow means. Counters was therefore not expanded
  to ten as part of the FO/Rover confirmation.
- Post-hoc 30m/2h cutoffs are defensible for coverage achievable within those
  budgets. A fresh hard-cap campaign is only needed to validate whole-job
  runtime/resource behavior or where interrupted logs make the post-hoc record
  incomplete.

### Counters exact-snapshot divergence recovery

| Seed | Policy | Fixed narrow 5/20 | PW20 30m / 2h / 6h | PW70 30m / 2h / 6h | Conclusion |
|---|---:|---:|---:|---:|---|
| 534933607 | 59 | 23 | 19 / 21 / 21 | ≥19 / ≥21 / ≥22 | Neither arm has recovered policy; PW70 live |
| 923500475 | 59 | 29 | 32 / 41 / 48 | 18 / 20 / 21 | PW20 strongly recovers fixed-search losses; PW70 does not |
| 2082152039 | 35 | 18 | 18 / 18 / 18 | 19 / 19 / 19 | Neither arm recovers policy |

Visible conclusion: widening is not a general guarantee against the severe
Counters policy-to-MCTS regressions. PW20 produced one strong recovery, whereas
PW70 did not; the last PW70 arm remains live.

### TPP catastrophic-seed MCTS — newly live

This diagnostic is defensible and nonredundant. It asks whether search can
recover failures from the exact validation-selected Stage-2 checkpoint that
scored 9/20 while the other nine TPP/off seeds scored 20/20.

| Arm | Job | Configuration | State | Comparator |
|---|---:|---|---|---:|
| Fixed normal | 21084616 | width 20, 70 simulations | Running | Policy 9/20 |
| Fixed narrow | 21084617 | width 5, 20 simulations | Running | Policy 9/20 |
| PW20 | 21084618 | Kmin 3, c 0.6, alpha 0.5, 20 simulations | Running | Policy 9/20 |
| PW70 | 21084619 | Kmin 3, c 0.6, alpha 0.5, 70 simulations | Running | Policy 9/20 |

All four use SAFE-1 and the same checkpoint, seed, VH-off mode, evaluator and
six-hour per-instance cap. They have passed application startup. This is a
mechanism diagnostic, not a ten-seed TPP MCTS mean.

### MPrime validation adequacy Phase B

Completed successfully:

- 240 candidates across two independent pools and three structural tiers.
- Planner screening and VAL validation completed in jobs 21039224_0–5.
- Sixty instances were frozen with paths and SHA-256 checksums in
  `/home/hersco/training_new_domains/2026-09-06/mprime_phase_b/frozen_validation_manifest.csv`.

The obsolete attempt was blocked before scoring:

- Preflight 21040245 failed after 51 seconds.
- The generated domain module used `VALIDATION_PDDLS = {'phase_b': [...]}`.
- The wrapper converted that key to `--validation-pddls-phase_b`, but the parser
  supports only `easy`, `medium`, and `hard` tier names.
- No checkpoint score was produced. Array 21040246 was dependency-cancelled and
  no expensive rescore work began.

That repair is now complete. The frozen lists are exposed through the supported
`hard` validation tier without changing their contents or hashes. Corrected
preflight 21084274 is running. Array 21084275 contains 60 tasks and is held by an
`afterok` dependency, with at most 12 simultaneous tasks. If the preflight
fails, the array will not run; if it succeeds, the rescore starts automatically.
MPrime MCTS remains intentionally unsubmitted until this audit freezes
defensible checkpoints.

## Best currently demonstrated result by domain

Paper values are the reported reference means used in the advisor audit. A
live lower bound or a nine-seed subset is labelled explicitly; it is not mixed
with completed ten-seed evidence.

| Domain | Best current result | 30m / 2h / 6h | Matched Stage-1 policy | Paper | Difference vs Stage 1 / paper | Defensible conclusion |
|---|---|---:|---:|---:|---:|---|
| Delivery | S1 validation-selected policy, VH-off | 19.8 / 19.8 / 19.8 | 19.8 | 20 | 0.0 / −0.2 | Stage 2 is preserved but does not beat the best S1 policy |
| TPP | Stage-1 policy | 20 / 20 / 20 | 20 | 20 | 0 / 0 | Perfect S1; S2 is not uniform because one off seed collapsed |
| Zenotravel | S1 policy and multiple S2 policy cells | 20 / 20 / 20 | 20 | 17 | 0 / +3 | Solved and preserved; exceeds paper reference |
| MPrime | Validation-led S2 policy | 15.2 / 15.2 / 15.2 | 15.0 | 19 | +0.2 / −3.8 | Provisional; Phase B must repair checkpoint-selection validity |
| Block Grouping | S1 validation-selected policy, off | 16.3 / 16.3 / 16.3 | 16.3 | 17 | 0 / −0.7 | Neither S2 nor narrow MCTS improves the best policy |
| Drone | Terminal-led S2 normal MCTS, on | 12.9 / 13.1 / 13.1 | 7.2 | 9 | +5.9 / +4.1 | Best established search success; Holm-significant |
| FO Counters | S1 PW70/off, provisional n=9 | 8.56 / 8.56 / 8.56 | 4.11 | 6 | +4.45 / +2.56 | Promising current maximum; completed fixed MCTS remains 7.8 |
| Rover | S1 fixed normal MCTS/off | 4.8 / 5.0 / 5.0 | 4.0 | 7 | +1.0 / −2.0 | Search improves policy modestly but remains below paper |
| Counters | Terminal-led S2 narrow MCTS/off, live | ≥34.9 / ≥37.8 / ≥38.4 | 21.5 | 17 | ≥+16.9 / ≥+21.4 | Largest current gain, but still a live lower bound |

The machine-readable version, including row-level log provenance, is
`best_configuration_by_domain_20260907.csv`.

## Held and ready designs

These are design holds only; none occupies Slurm.

| Priority | Experiment | Reason for hold | Activation condition |
|---:|---|---|---|
| 1 | MCTS-PW-30M | Fresh execution is optional for coverage; useful for whole-job efficiency validation | Select final promising cells after PW70 tails |
| 2 | ANCHOR-KL-CONTROL | Requires generic schedule/controller implementation | Use TPP collapse and matched stable seeds as diagnostic screen |
| 3 | MCTS-PW-PATHBATCH | Nonstandard multiple-expansion semantics | Freeze update semantics and smoke-test |
| 4 | MCTS-RESOURCE | Resource/lifecycle sensitivity | Use only for unresolved interruption/OOM endpoints |
| 5 | ACT-HISTORY-ABLATION | Requires entirely new Stage-1 and Stage-2 networks | Launch only as a separate long campaign |
| 6 | LONG-DRONE selected-checkpoint MCTS | Three side-experiment endpoints remain | Release only if the long-training side question is reprioritized |
| 7 | STOP-ORIG | Separate stopping-rule replication | Finalize compatibility manifest |
| 8 | PUCT-EST | Large causal hyperparameter grid | Release only after higher-priority evidence closes |
| 9 | MCTS-SAFE2 | No observed horizon-statistics contamination and memory risk | Require nonzero audited cutoffs plus cross-horizon reuse |

## Completed RQs and experiments — unchanged static evidence

### RQ1 — does MCTS-guided Stage-2 training improve VH-off policy?

| Domain | Validation-led change [95% CI]; Holm p | Terminal-led change [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | −0.3 [−1.20, 0.60]; 1.000 | 0.0 [−0.95, 0.95]; 1.000 | No improvement |
| Drone | +0.8 [−1.27, 2.87]; 1.000 | +0.4 [−1.33, 2.13]; 1.000 | No reliable improvement |
| FO Counters | −1.3 [−2.37, −0.23]; .234 | −0.8 [−1.46, −0.14]; .219 | Negative raw effect; not Holm-significant |
| Rover | 0.0 [0, 0]; 1.000 | 0.0 [−0.34, 0.34]; 1.000 | No change |
| Counters | +4.4 [−17.44, 26.24]; 1.000 | +16.4 [3.09, 29.71]; .137 | Large noisy mean; not Holm-significant |

Conclusion: no original five-domain RQ1 result survives Holm correction.

### RQ3 — does the value head improve refinement?

| Domain | Validation-led DiD [95% CI]; Holm p | Terminal-led DiD [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | −2.8 [−4.61, −0.99]; .088 | −3.4 [−5.13, −1.67]; .020 | **Terminal-led VH significantly worsens refinement** |
| Drone | −0.9 [−2.88, 1.08]; 1.000 | −1.1 [−3.57, 1.37]; .836 | No reliable effect |
| FO Counters | +0.7 [−0.26, 1.66]; .813 | +1.3 [0.08, 2.52]; .188 | Positive raw tendency only |
| Rover | +0.1 [−0.31, 0.51]; 1.000 | −0.1 [−0.51, 0.31]; 1.000 | No effect |
| Counters | −1.2 [−25.87, 23.47]; 1.000 | −16.8 [−29.64, −3.96]; .078 | Large negative tendency; high variance |

Conclusion: only terminal-led Block Grouping is significant after correction.

### RQ2/RQ4 — inference-time MCTS

The complete current table is above. The advisor-ready conclusion is already
strong for Drone/VH-on and FO/off, negative for terminal-led Block Grouping,
and unresolved for live FO/Counters cells.

### PRESERVE-3 validation-led

| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Conclusion |
|---|---:|---:|---:|---:|---|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | −0.2 [−1.01, .61] | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +0.4 [−.37, 1.17] | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | −1.1 [−3.59, 1.39] | Not uniformly preserved |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | −0.5 [−1.63, .63] | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0, 0] | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | −0.1 [−.33, .13] | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 collapses to 9/20.

### PRESERVE-3 terminal-led

| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Conclusion |
|---|---:|---:|---:|---:|---|---|
| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [−2.52, 5.12] | Preserved relative to final source |
| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [−.91, 4.31] | Preserved/improved mean |
| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +0.4 [−2.13, 2.93] | Preserved on average |
| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | −0.6 [−3.77, 2.57] | No reliable change |
| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | −0.2 [−.50, .10] | Essentially preserved |
| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +0.2 [−.10, .50] | Preserved |

### MPrime policy extension — provisional pending adequacy audit

| Branch/VH | Stage 1 | Stage 2 selected | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---|---:|---|
| Validation/off | 15.0 | 15.2 | +0.2 [−1.18, 1.58] | .880 | Neutral |
| Validation/on | 14.6 | 15.2 | +0.6 [−1.46, 2.66] | .580 | Neutral |
| Terminal/off | 13.5 | 14.0 | +0.5 [−1.29, 2.29] | .672 | Neutral |
| Terminal/on | 13.2 | 14.5 | +1.3 [−.49, 3.09] | .172 | Positive tendency only |

These results are not promoted into the primary RQ family until Phase B
establishes defensible checkpoint selection.

### Other completed side experiments

| Experiment | Final conclusion |
|---|---|
| MCTS-PW Drone | Faster and dramatically fewer retained nodes, but significant coverage loss |
| MCTS-SAFE-1 | Repaired 2/4 targeted Drone dead-end outcomes; useful guard, incomplete quality fix |
| MCTS-SAFE-CONTEXT | Negative; contextual node splitting harmed coverage, especially VH-on |
| MCTS-HORIZON Drone and Counters | Non-results with zero effective cutoffs; no efficacy evidence |
| Determinism audit | CPU-family numerical checksums differ, but audited selected actions/outcomes did not |
| MCTS-WIDTH | BG and Counters primary configurations corrected to narrow 5/20; normal Counters is not the confirmatory comparator |
| LONG-DRONE | Policy and final-checkpoint MCTS complete; three selected-checkpoint MCTS endpoints remain design-held |

### Complete lifecycle inventory

| Lifecycle | Experiments |
|---|---|
| Live | MPRIME-VAL-ADEQUACY Phase B; TPP-CATASTROPHIC-MCTS; Counters terminal-led Stage-2 MCTS; FO terminal-led Stage-2 MCTS tail; PW70 narrow-domain correction; PW70 FO/Rover ten-seed expansion; Counters PW70 divergence tail |
| Completed mainstream policy | MAIN-VAL; MAIN-TERM; PRESERVE-3-VAL; PRESERVE-3-TERM |
| Completed/provisional MPrime policy | MPRIME-VAL corrected Stage 1; MAIN-EXT6-MPRIME; MAIN-TERM-EXT6-MPRIME; ANCHOR-4 |
| Completed search | Stage-1 fixed MCTS; completed Stage-2 fixed cells; MCTS-PW Drone; MCTS-PW-SAFE; mixed-budget cross-domain PW screen; five-seed PW70 screen; MCTS-SAFE-1; MCTS-SAFE-CONTEXT; MCTS-HORIZON Drone; MCTS-HORIZON-COUNTERS; MCTS-WIDTH; determinism audit |
| Completed operational | Storage audit and large-log compaction; policy-curve coverage materialization; Stage-2 MCTS branch-provenance recovery |
| Design-held | MCTS-PW-30M; ANCHOR-KL-CONTROL; MCTS-PW-PATHBATCH; MCTS-RESOURCE; ACT-HISTORY-ABLATION; LONG-DRONE selected MCTS; STOP-ORIG; PUCT-EST; MCTS-SAFE2 |

MPrime is absent from the inference-time MCTS comparison by design. Phase B
must first select defensible Stage-1/Stage-2 checkpoints. Only then can the
40-job MPrime MCTS matrix (two branches × two VH modes × ten seeds) be built;
those MCTS jobs have not been submitted.

## Provenance and authoritative files

- Every live job: `cluster_workload_20260907.csv` and
  `cluster_workload_latest.csv`.
- Every Stage-2 MCTS branch: `stage2_mcts_branch_coverage_20260907.csv` and
  `stage2_policy_mcts_comparison_by_branch_20260907.csv`.
- PW70 expansion job-level logs: `mcts_progressive_widening_cross_domain/pw70_ten_seed_jobs_20260907.csv`.
- PW70 interim statistics: `mcts_progressive_widening_cross_domain/pw70_ten_seed_interim_20260907.csv`.
- MPrime Phase B jobs and logs: `mprime_validation_phase_b_status_20260907.csv`.
- TPP outlier diagnostic jobs, checkpoint, training and policy logs:
  `tpp_catastrophic_mcts/submissions_20260907.csv`.
- Best per-domain result and paper comparison:
  `best_configuration_by_domain_20260907.csv`.
- Full experiment registry: `experiments.csv` and `experiment_registry.csv`.

Every new result row points either directly to the original cluster log or to a
row-level provenance companion containing the checkpoint and original logs.

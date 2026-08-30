# Numeric ASNets advisor meeting brief — 30 August 2026

## Executive position

- The two mainstream **policy** experiments are complete at ten matched seeds per domain/VH cell. Continued Stage-2 training is not uniformly beneficial. Positive effects in Counters are large but variable; several Block Grouping and FO Counters cells regress.
- RQ-aligned reanalysis changes the statistical story: no RQ1 Stage-2 effect survives the **original five-domain confirmatory** Holm correction. MPrime is not silently inserted into that family: after its Stage-2 confirmation, a separate six-domain extension will be reported. For RQ3, all ten domain/lineage cells are reported below; only terminal-led Block Grouping survives Holm correction.
- Stage-1 MCTS is a strong secondary result: VH-off FO Counters improves by +3.6 plans (Holm p=.0195); VH-on Drone improves by +5.3 (p=.0098) and FO Counters by +2.0 (p=.0156). Block Grouping and normal-search Counters do not improve.
- Progressive widening creates a real efficiency/coverage frontier. Kmin=3 retains 9.5/20 versus fixed top-20’s 10.5/20 while cutting successful-instance median runtime from 130 s to 94 s and eliminating recorded successes above 30 minutes (0/76 versus 2/84).
- SAFE-1 repairs two of four observed known-terminal selections. SAFE-CONTEXT is the more consequential correctness test: 47.1% of physical-state revisits in the diagnostic carried a different action-history context, with up to 5.44× node multiplication. Its full matched Drone experiment is live and must report memory as well as coverage.
- The original “four perfect domains” story did not hold. Corrected MPrime selection averages 15.0/20 (VH-off) and 14.6/20 (VH-on), so the defensible domain split is **three stable domains** (Delivery, TPP, Zenotravel) and **six imperfect domains**.

## Research questions and registered estimands

| RQ | Question | Primary estimand | Current state |
|---|---|---|---|
| RQ1 | Can MCTS-guided continued training improve a ν-ASNet policy? | VH-off Stage-2 policy − Stage-1 policy | Complete for MAIN-VAL and MAIN-TERM policy; no Holm-significant domain |
| RQ2 | Does inference-time MCTS improve policy-only inference? | VH-off Stage-2 MCTS − same-checkpoint policy | Primary Stage-2 evidence incomplete; Stage-1 secondary evidence complete |
| RQ3 | Does the value head improve MCTS-guided refinement? | (VH-on S2−S1) − (VH-off S2−S1) | Complete; terminal-led Block Grouping is significantly negative |
| RQ4 | Does the value head improve inference-time MCTS? | VH-on Stage-2 MCTS − same-checkpoint policy | Primary Stage-2 evidence incomplete; Stage-1 secondary evidence complete |

Exact paired sign-flip tests use network seed as the independent unit. Holm correction is applied across the original five domains **within each RQ**, not across ten VH/domain rows. MPrime entered the imperfect-domain group after the confirmatory family was frozen, so its future Stage-2 result will be shown both alone and in a separately labelled six-domain extension. Existing domain CIs and raw p-values will not change; the six-domain Holm p-values may stay equal or become more conservative. Paper scores have no seed-level variance, so they receive descriptive—not inferential—comparisons.

## Mainstream Stage-2 policy results

Mean coverage; Counters is out of 59, other domains out of 20. `raw/Holm` are exact sign-flip p-values for the paired change.

| Domain | MAIN-VAL VH-off S1→S2 | Δ [95% CI]; raw/Holm | MAIN-TERM VH-off S1→S2 | Δ [95% CI]; raw/Holm |
|---|---:|---:|---:|---:|
| Block Grouping | 16.3→16.0 | −0.3 [−1.20, .60]; .625/1.000 | 16.3→16.3 | 0.0 [−.95, .95]; 1.000/1.000 |
| Drone | 5.9→6.7 | +0.8 [−1.27, 2.87]; .504/1.000 | 7.4→7.8 | +0.4 [−1.33, 2.13]; .688/1.000 |
| FO Counters | 4.2→2.9 | −1.3 [−2.37, −.23]; .047/.234 | 3.6→2.8 | −0.8 [−1.46, −.14]; .055/.219 |
| Rover | 4.0→4.0 | 0.0; 1.000/1.000 | 3.8→3.8 | 0.0 [−.34, .34]; 1.000/1.000 |
| Counters | 32.5→36.9 | +4.4 [−17.44, 26.24]; .660/1.000 | 21.5→37.9 | +16.4 [3.09, 29.71]; .027/.137 |

### RQ3 — value-head difference-in-differences

The estimand is `(VH-on S2−S1) − (VH-off S2−S1)`. Negative values mean the value head made Stage-2 refinement worse relative to VH-off.

| Domain | MAIN-VAL ΔΔ [95% CI] | raw / Holm p | MAIN-TERM ΔΔ [95% CI] | raw / Holm p |
|---|---:|---:|---:|---:|
| Block Grouping | −2.8 [−4.61, −.99] | .0176 / .0879 | **−3.4 [−5.13, −1.67]** | **.0039 / .0195** |
| Drone | −0.9 [−2.88, 1.08] | .377 / 1.000 | −1.1 [−3.57, 1.37] | .418 / .836 |
| FO Counters | +0.7 [−.26, 1.66] | .203 / .813 | +1.3 [.08, 2.52] | .0625 / .188 |
| Rover | +0.1 [−.31, .51] | 1.000 / 1.000 | −0.1 [−.51, .31] | 1.000 / 1.000 |
| Counters | −1.2 [−25.87, 23.47] | .938 / 1.000 | −16.8 [−29.64, −3.96] | .0195 / .0781 |

Only terminal-led Block Grouping is significant after five-domain Holm correction. Counters terminal-led is large and raw-significant, but its high seed variance leaves the corrected result above .05.

## Stage-1 policy versus MCTS — secondary RQ2/RQ4 evidence

| Domain | Search | VH-off policy→MCTS | Δ [95% CI]; Holm | VH-on policy→MCTS | Δ [95% CI]; Holm |
|---|---|---:|---:|---:|---:|
| Block Grouping | width 5 / 20 simulations | 16.3→15.4 | −0.9 [−1.88, .08]; .223 | 15.9→16.2 | +0.3 [−.29, .89]; .906 |
| Drone | width 20 / 70 | 5.9→6.9 | +1.0 [.05, 1.95]; .223 | 5.1→10.4 | +5.3 [3.42, 7.18]; **.0098** |
| FO Counters | width 20 / 70 | 4.2→7.8 | +3.6 [2.20, 5.00]; **.0195** | 3.7→5.7 | +2.0 [1.05, 2.95]; **.0156** |
| Rover | width 20 / 70 | 4.0→5.0 | +1.0 [.25, 1.75]; .125 | 3.8→4.4 | +0.6 [−.00, 1.20]; .375 |
| Counters | width 20 / 70 | 32.5→21.9 | −10.6 [−23.64, 2.44]; .223 | 18.6→16.7 | −1.9 [−13.95, 10.15]; .906 |

The domain-specific search configurations are intentional. A pooled cross-domain “MCTS effect” is not valid. Counters’ width-5/20 campaign is live because normal 20/70 creates too many expensive successors.

## Three stable domains and six imperfect domains

Published single-run baselines were: Delivery 20/20, TPP 20/20, Zenotravel 17/20, MPrime 19/20, Block Grouping 17/20, Drone 9/20, FO Counters 6/20, Rover 7/20, Counters 17/59.

- **Stable three:** Delivery Stage-1 selected 19.8/off and 19.2/on; TPP 20.0 in both modes; Zenotravel 20.0 in both modes. Zenotravel Stage 2 preserves 20.0/off and reaches 19.9/on. Delivery and TPP confirmation training is live.
- **Reclassified MPrime:** corrected validation-selected means are 15.0/off and 14.6/on; finals 13.5/off and 13.2/on. Corrected validation is better than the epoch-1-biased set but remains a weak ranking signal (pooled Spearman ≈.25). MPrime now belongs with the six imperfect domains.
- **Six imperfect:** Block Grouping, Drone, FO Counters, Rover, Counters, MPrime. MCTS helps some cells strongly, does nothing in others, and can degrade coverage under inappropriate width or long divergence.

## Live experiment status and estimates

Snapshot: 2026-08-30 17:09 IDT. Requested resources are exact Slurm requests, not measured consumption.

| Experiment | Running | Ordinary pending | Held | Running resources | Position and estimate |
|---|---:|---:|---:|---:|---|
| SAFE-CONTEXT matched Drone | 19 | 0 | 0 | 114 CPU / 2,280 GiB | One contextual arm terminal; running arms have 5–17/20 classified after 6–8.5 h; timeout-heavy arms can take 12–30 h, 72 h hard cap |
| Cross-domain PW Kmin=3 | 10 | 9 | 0 | 60 CPU / 1,200 GiB | One Counters S2 arm terminal; Block Grouping 11–15 successes already, Counters 12–40; FO/Rover await memory |
| Binding Horizon, remaining pairs | 2 | 0 | 0 | 12 CPU / 240 GiB | Aware arms have 6/17 and 13/16 successes; likely hours, 72 h hard cap |
| Counters Stage-2 narrow | 6 | 0 | 0 | 36 CPU / 720 GiB | 54.4 h elapsed; ≤17.6 h to scheduler cap; conservative lower bounds retained |
| MPrime anchor tuning | 17 | 0 | 0 | 102 CPU / 816 GiB | 11/28 terminal; most remaining jobs at epochs 75–99, slowest epoch 48; roughly minutes to 6.5 h for most |
| Delivery Stage-2 | 3 | 0 | 0 | 18 CPU / 144 GiB | Epochs 91–94; roughly 2.7–4.2 h to epoch 100 |
| TPP Stage-2 | 3 | 0 | 0 | 18 CPU / 144 GiB | Epochs 82, 70, 49; roughly 9 h, 18 h, and scheduler-limited in ≤30 h |
| Rover endpoint MCTS | 5 | 16 | 0 | 30 CPU / 600 GiB | Running jobs have 4–5 successes after 9 min–3.5 h; historical ~30 h median and substantial OOM risk |
| Drone MAIN-VAL Stage-2 MCTS gap | 0 | 16 | 0 | pending 96 CPU / 1,920 GiB | All blocked by per-user memory; required for primary RQ2/RQ4 evidence |
| FO Counters legacy MCTS | 0 | 0 | 20 | held 120 CPU / 2,400 GiB | Deliberately held pending lifecycle/resource-safe release |

Totals: **65 running jobs, 390 CPUs, 6,144 GiB requested (exactly 6 TiB)**. Pending: **41 ordinary jobs, 246 CPUs, 4,920 GiB**, all memory-blocked; plus **20 deliberately held jobs, 120 CPUs, 2,400 GiB**.

## Search-method findings

### Progressive widening

- Fixed top-5 and fixed top-20 are statistically indistinguishable on the original Drone pilot.
- PW c=.6: 6.7/20, but average whole-job runtime 8.12 h versus fixed top-20 11.83 h.
- PW c=1.0: 6.2/20, runtime 5.03 h, but clear coverage loss.
- SAFE Kmin=3 extension: policy 7.0/20, fixed top-20 10.5/20, PW 9.5/20 across eight matched Drone cells. Successful-instance median 94 s versus 130 s; p90 318 s versus 893 s; maximum 594 s versus 6,127 s.
- Under a deterministic 30-minute post-hoc cap: policy 7.0, fixed top-20 10.25, PW Kmin=3 9.5. This is already valid descriptive recensoring of recorded runs. A fresh hard-cap campaign is now registered as `MCTS-PW-30M` because it is inexpensive and would turn the result into direct operational evidence.

### Counters narrow search

- Stage-1 conservative terminal lower bounds: VH-off policy 32.5, normal 21.9, narrow ≥25.7; VH-on policy 18.6, normal 16.7, narrow ≥22.5. Interrupted rows count unfinished instances as failures; all printed plans were post-hoc VAL-valid.
- Stage-2 reconciled subset: VH-off policy 44.83→narrow 43.67 (n=6); VH-on 29.4→33.8 (n=5). Eleven of twelve VH-off policy-only successes diverged to exactly 10,000 actions rather than timing out.

### Safety and contextual identity

- SAFE-1’s terminal mask is successful but incomplete: two of four targeted Drone failures repaired.
- SAFE-CONTEXT gives `(physical state, action-history digest)` an independent MCTS node/statistics cache while cycle detection remains physical-state based. The diagnostic found 171,659 different-context revisits among 364,576 physical revisits (47.1%).
- This is not free: observed contextual node multiplication reaches 5.44× in Counters. Full matched Drone results must jointly report coverage, runtime, nodes and peak RSS.

### Horizon

- The original 10,000-action Drone horizon run was nonbinding: zero cutoffs.
- The live 750-action fresh aware/unaware experiment has eight complete pairs: seven ties and one aware +1; only one cutoff has been observed. Two aware arms remain live. If cutoffs remain rare, Counters is the defensible next domain.

## Story holes that materially affect the thesis

The full register is `story_holes.csv`. Highest-priority gaps are:

1. Finish Stage-2 selected-checkpoint MCTS before claiming primary RQ2/RQ4.
2. Keep MAIN-TERM framed as terminal-checkpoint sensitivity; use STOP-ORIG for strict paper-style stopping fidelity.
3. Do not treat partial/OOM MCTS scores as ordinary completed jobs without explicitly calling them conservative fixed-budget lower bounds.
4. Separate tuning seeds from held-out seeds in the stable-domain preservation claim.
5. Treat two-seed cross-domain PW as screening, not confirmation.
6. Report memory multiplication in SAFE-CONTEXT; otherwise a coverage gain could hide an infeasible resource cost.
7. Do not infer “no effect” from wide CIs or small n.

## Recommended decision order

1. Let current policy/training and the 16 submitted Drone Stage-2 MCTS gaps run; they answer registered questions.
2. Freeze the corrected RQ-aligned statistics and use them in the thesis/presentation.
3. Complete Delivery/TPP and MPrime anchor analysis, with tuning/held-out separation.
4. Use the current two-seed PW cross-domain run as a screen; expand only promising domains to five seeds.
5. Deploy the lifecycle/resource-safe continuation before broad FO Counters/Rover MCTS release.
6. Run STOP-ORIG as a three-seed pilot before committing to a full strict-replication campaign.
7. Keep SAFE-2, path-batched PW, PUCT/estimator sweeps and Stage-2-final MCTS in the appendix/held register until primary RQs are closed.

## Source files

- RQ statistics: `rq_results.csv`
- Exact live resources: `cluster_workload.csv`
- Story holes: `story_holes.csv`
- Per-experiment improvements: `improvement_priorities.csv`
- Seed-level policy pairs: `../policy_paired_seed_results.csv`
- Seed-level MCTS pairs: `../mcts_paired_seed_results.csv`
- Raw endpoint ledger and log pointers: `../experiment_results.csv`
- Learning-curve/MCTS coverage contract: `../evaluation_coverage_plan.csv`
- Live Slurm jobs: `../live_jobs.csv`

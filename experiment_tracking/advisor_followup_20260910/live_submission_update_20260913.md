# Live submission update — 13 September 2026, 23:45 IDT

Only validation-led work is treated as thesis-primary. Terminal-led campaigns remain archived context and are not continued, plotted, or included in RQ families.

## Counters tie-break confirmation

Compute smoke `21233924` completed in 1m15s and passed 23 tests, CLI checks, the complete 20-row manifest, and all checkpoint existence checks. Four scientific tasks are terminal and 16 are running. The two completed matched pairs are exact ties: seed `1073581256` scores 59/59 under both rules, and seed `2011206605` scores 20/59 under both rules. This is neutral two-pair evidence, not a domain-wide effect estimate. Across all live ledgers, action-ID has classified 412/590 seed-instances with at least 254 successes; policy-prior has classified 403/590 with at least 249 successes. Those aggregate live values are lower bounds and must not be paired until every matching seed is terminal.

The array is a strict same-build Stage-1 VH-off comparison over all ten validation-selected checkpoints and all 59 test instances:

- ten current action-index tie-break evaluations;
- ten policy-prior tie-break evaluations;
- narrow search: five retained children and 20 simulations;
- six-hour per-instance and 10,000-external-action limits;
- six CPUs and 120 GiB per task, 72-hour task allocation;
- at most 120 CPUs and 2.4 TiB if all tasks run concurrently.

Historical comparable Counters jobs have a median whole-job runtime near 49.9 hours. At this snapshot the 16 live tasks had run about 10.5 hours, so a domain-based midpoint estimate is roughly 39 hours more, while their hard remaining bounds are about 61.3-61.5 hours. The array directly tests whether the candidate recovers the main RQ2 VH-off downgrade. Q-only final tie-breaking was not promoted because it recovered none of the three causal diagnostic instances.

The causal screen motivating this confirmation is final: action-ID and Q tie-breaking solved 0/3 targeted VH-off failures, while policy-prior tie-breaking solved all 3/3 by two hours with VAL-valid plans. The VH-on arm also scored 0/3 under all three rules, but its exact policy solved 0/3 of those targets and only 3/59 overall. It is therefore only a matched-mode behavior check, not a policy-preservation control and specifically not a failure of policy-prior tie-breaking. This is strong targeted VH-off mechanism evidence, not yet a domain-wide effect estimate.

The action-ID half is intentionally rerun. It makes the domain-wide comparison same-build and same-scheduler-era, so any net difference can be attributed to tie-breaking rather than comparing the new policy-prior rule against historical binaries and environments. Historical and current baselines will not be pooled.

## MPrime

At this snapshot, 466/588 checkpoint validations are complete. All original attempts have left the queue. Six resumable replacement tasks are live, requesting 24 CPUs and 120 GiB in total, with about 13 hours to their 24-hour hard bounds. Recheck controller `21233927` remains dependency-pending and will skip every completed point.

Controller `21233927` audits paired done/summary markers, submits only incomplete lineage indices, and can repeat this bounded recovery up to four times before submitting the analysis-only coefficient finalizer. It cannot launch Stage-2 training itself: the frozen coefficient still requires manual curve review. Once frozen, the approved path is validation-led Stage-2 training, policy curves/endpoints, then matched Stage-2 fixed MCTS. An exact epoch join already proves 16/20 old Stage-2 lineages have the wrong Stage-1 source. Four remain candidates for reuse and require exact checkpoint-hash, selected-coefficient, code and full-configuration matches. Therefore the current defensible fresh-training range is 16-20. The separate 0/20 Stage-1 MCTS reuse result does not prove that all 20 Stage-2 training identities differ.

The Phase-B-A Stage-1 checkpoints are already final. An exact identity audit found 0/20 reusable Stage-1 fixed-MCTS evaluations because historical MCTS checkpoints did not match the final Phase-B-A selector. Compute smoke `21237283` solved its real MPrime instance and passed VAL. The two ten-task arrays `21237328` (VH-off) and `21237329` (VH-on) are running, requesting 120 CPUs and 2,400 GiB total. Every task uses normal fixed 20/70 search, terminal-safe action selection disabled, 6 CPUs, 120 GiB, a six-hour per-instance limit and a 72-hour hard allocation. At about 4h55m elapsed, their durable success lower bounds are at least 6.8/20 VH-off and 7.2/20 VH-on; no seed is terminal, so these are not final scores and no CI or p-value is valid. Their hard remaining bound is about 67.1 hours. A Stage-1 PW screen is gated on having an interpretable fixed comparator only, not on Stage-2 training.

MPrime PW has not been run. Stage-1 PW can begin with a two-seed screen after the Stage-1 fixed comparator; Stage-2 PW remains gated on canonical Stage-2 endpoints and its fixed comparator.

## Block Grouping tie-break diagnostic

A two-job same-build screen is running as array `21237401` on the four exact Stage-1 VH-off seed `1963100312` instances that policy solved and historical narrow MCTS drove to 10,000 actions. It compares action-ID against policy-prior final tie-breaking with one worker, 2 CPUs, 120 GiB and 30 hours per task. At this snapshot both rules have durably classified the same first three targets as ordinary unsolved outcomes (0/3); the fourth is still active. The hard remaining bound is about 25.1 hours. The real compute smoke passed 23 tie-break tests; only the exact node `ise-cpu-intl-13` is excluded after native exit `-4`.

This screen covers four of the 16 VH-off policy-success/MCTS failures across ten seeds. The full decomposition is 46 MCTS failures = 16 policy-loss cases + 30 shared failures, plus seven policy-failure/MCTS-success gains. A preservation tie-break is not expected to solve the 30 shared failures. Block Grouping is the only immediate cross-domain transfer because it shares narrow 5/20 coarse-visit search. Drone, FO Counters and Rover use 70 simulations and have no traced tied-visit mechanism, so full reruns there would currently be speculative.

Phase C itself is complete. It removed the original validation saturation, but Phase-B replicate A had the best available rank/stability trade-off and is frozen as the final MPrime validator. The current rescore is the required bridge from that validator decision to defensible Stage-2 training.

## Current allocation

The live scientific workload is 244 CPUs and 4,680 GiB: Counters strict confirmation 96/1,920; MPrime Stage-1 MCTS 120/2,400; MPrime anchor recovery 24/120; Block Grouping screen 4/240. One 1-CPU/2-GiB controller is dependency-pending. Every unambiguous primary prerequisite is already live or controller-covered. MPrime Stage-2 remains gated on the anchor review; wider Block Grouping confirmation remains gated on this selected-failure screen.

## FO Counters Stage-2 fixed MCTS

The last exact recovery `21219947` completed after 6h01m. Its requested instance consumed the full six-hour per-instance budget and produced no plan. Therefore the VH-off result is final, not partial: policy 2.9/20 versus MCTS 6.1/20 at 30m, 2h and 6h; paired change +3.2, 95% CI [1.74, 4.66], raw p=.00195, Holm p=.00977.

Row-level source training jobs, policy jobs, MCTS logs and durable ledger paths are in `fo_stage2_validation_vh_off_exact_seed_results_20260913.csv` and the canonical RQ CSVs.

## Adaptive-KL screen — recently completed

Jobs `21144388` and `21144389` both completed 100 epochs. The coefficient remained at its minimum value 3 for every update because post-update KL never crossed the predeclared target 0.1143; the controller therefore made zero adjustments. The catastrophic seed was still 10/20 at epoch 0 versus 20/20 for the stable control. This parameterization did not activate and did not repair the first-update collapse. It is a negative/inconclusive activation screen, not live work and not evidence against adaptive control in general.

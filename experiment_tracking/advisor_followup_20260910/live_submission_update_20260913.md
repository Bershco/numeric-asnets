# Live submission update — 13 September 2026, 19:01 IDT

Only validation-led work is treated as thesis-primary. Terminal-led campaigns remain archived context and are not continued, plotted, or included in RQ families.

## Counters tie-break confirmation

Compute smoke `21233924` completed in 1m15s and passed 23 tests, CLI checks, the complete 20-row manifest, and all checkpoint existence checks. All 20 low-priority scientific tasks have now received compute: two are terminal and 18 are running. The first completed matched seed (`1073581256`) scores 59/59 with both action-ID and policy-prior tie-breaking, with every plan VAL-valid. This is one neutral pair, not an interim domain-wide effect estimate.

The array is a strict same-build Stage-1 VH-off comparison over all ten validation-selected checkpoints and all 59 test instances:

- ten current action-index tie-break evaluations;
- ten policy-prior tie-break evaluations;
- narrow search: five retained children and 20 simulations;
- six-hour per-instance and 10,000-external-action limits;
- six CPUs and 120 GiB per task, 72-hour task allocation;
- at most 120 CPUs and 2.4 TiB if all tasks run concurrently.

Historical comparable Counters jobs have a median whole-job runtime near 49.9 hours; each task's hard bound is 72 hours after it starts. The array directly tests whether the candidate recovers the main RQ2 VH-off downgrade. Q-only final tie-breaking was not promoted because it recovered none of the three causal diagnostic instances.

The causal screen motivating this confirmation is final: action-ID and Q tie-breaking solved 0/3 targeted VH-off failures, while policy-prior tie-breaking solved all 3/3 by two hours with VAL-valid plans. The VH-on arm also scored 0/3 under all three rules, but its policy solved none of those three instances and only 3/59 overall. It is therefore only a matched-mode behavior check, not a policy-preservation control or evidence against VH-on recovery. This is strong targeted VH-off mechanism evidence, not yet a domain-wide effect estimate.

The action-ID half is intentionally rerun. It makes the domain-wide comparison same-build and same-scheduler-era, so any net difference can be attributed to tie-breaking rather than comparing the new policy-prior rule against historical binaries and environments. Historical and current baselines will not be pooled.

## MPrime

At this snapshot, 422/588 checkpoint validations are complete. Fourteen long-running original tasks and six resumable replacement tasks are live, requesting 80 CPUs and 400 GiB in total. The original tasks have about 2h15m to their 24-hour hard bounds; the replacement tasks have about 17h55m. Recheck controller `21233927` remains dependency-pending and will skip every completed point.

Controller `21233927` audits paired done/summary markers, submits only incomplete lineage indices, and can repeat this bounded recovery up to four times before submitting the analysis-only coefficient finalizer. It cannot launch Stage-2 training itself: the frozen coefficient still requires manual curve review. Once frozen, the approved path is up to twenty validation-led Stage-2 lineages, policy curves/endpoints, then matched Stage-2 fixed MCTS. An existing lineage is reused only if its Stage-1 source checkpoint hash, newly selected coefficient, code and complete configuration match exactly; the final fresh-training count will be decided by that identity audit rather than assumed.

The Phase-B-A Stage-1 checkpoints are already final. An exact identity audit found 0/20 reusable Stage-1 fixed-MCTS evaluations because historical MCTS checkpoints did not match the final Phase-B-A selector. Compute smoke `21237283` solved its real MPrime instance and passed VAL. The two ten-task arrays `21237328` (VH-off) and `21237329` (VH-on) are now running, requesting 120 CPUs and 2,400 GiB total. Every task uses normal fixed 20/70 search, terminal-safe action selection disabled, 6 CPUs, 120 GiB, a six-hour per-instance limit and a 72-hour hard allocation. A Stage-1 PW screen is gated on having that fixed comparator only, not on Stage-2 training.

MPrime PW has not been run. Stage-1 PW can begin with a two-seed screen after the Stage-1 fixed comparator; Stage-2 PW remains gated on canonical Stage-2 endpoints and its fixed comparator.

## Block Grouping tie-break diagnostic

A two-job same-build screen is running as array `21237401` on the four exact Stage-1 VH-off seed `1963100312` instances that policy solved and historical narrow MCTS drove to 10,000 actions. It compares action-ID against policy-prior final tie-breaking with one worker, 2 CPUs, 120 GiB and 30 hours per task. It also records action-level evidence. The real compute smoke passed 23 tie-break tests and completed a ten-minute target-instance run on `ise-cpu-intl-20`; an earlier attempt exposed native exit `-4` on `ise-cpu-intl-13`, so only that exact node is excluded. This selected-failure diagnostic is eligible because Block Grouping has 16 VH-off policy-success/MCTS-failure cases across eight seeds, but it is deliberately not a premature 20-job domain confirmation and does not test PW70. Historical elapsed times put the action-ID arm near 8.2 hours for these four targets; the hard bound is 30 hours per task.

Phase C itself is complete. It removed the original validation saturation, but Phase-B replicate A had the best available rank/stability trade-off and is frozen as the final MPrime validator. The current rescore is the required bridge from that validator decision to defensible Stage-2 training.

## FO Counters Stage-2 fixed MCTS

The last exact recovery `21219947` completed after 6h01m. Its requested instance consumed the full six-hour per-instance budget and produced no plan. Therefore the VH-off result is final, not partial: policy 2.9/20 versus MCTS 6.1/20 at 30m, 2h and 6h; paired change +3.2, 95% CI [1.74, 4.66], raw p=.00195, Holm p=.00977.

Row-level source training jobs, policy jobs, MCTS logs and durable ledger paths are in `fo_stage2_validation_vh_off_exact_seed_results_20260913.csv` and the canonical RQ CSVs.

## Adaptive-KL screen — recently completed

Jobs `21144388` and `21144389` both completed 100 epochs. The coefficient remained at its minimum value 3 for every update because post-update KL never crossed the predeclared target 0.1143; the controller therefore made zero adjustments. The catastrophic seed was still 10/20 at epoch 0 versus 20/20 for the stable control. This parameterization did not activate and did not repair the first-update collapse. It is a negative/inconclusive activation screen, not live work and not evidence against adaptive control in general.

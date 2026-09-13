# Live submission update — 13 September 2026, 13:00 IDT

Only validation-led work is treated as thesis-primary. Terminal-led campaigns remain archived context and are not continued, plotted, or included in RQ families.

## Counters tie-break confirmation

Compute smoke `21233924` completed in 1m15s and passed 23 tests, CLI checks, the complete 20-row manifest, and all checkpoint existence checks. Scientific array `21233925[0-19]` is submitted at low priority. Task 0 is running and tasks 1–19 are resource-pending.

The array is a strict same-build Stage-1 VH-off comparison over all ten validation-selected checkpoints and all 59 test instances:

- ten current action-index tie-break evaluations;
- ten policy-prior tie-break evaluations;
- narrow search: five retained children and 20 simulations;
- six-hour per-instance and 10,000-external-action limits;
- six CPUs and 120 GiB per task, 72-hour task allocation;
- at most 120 CPUs and 2.4 TiB if all tasks run concurrently.

Historical comparable Counters jobs have a median whole-job runtime near 49.9 hours; each task's hard bound is 72 hours after it starts. The array directly tests whether the candidate recovers the main RQ2 VH-off downgrade. Q-only final tie-breaking was not promoted because it recovered none of the three causal diagnostic instances.

The causal screen motivating this confirmation is final: action-ID and Q tie-breaking solved 0/3 targeted VH-off failures, while policy-prior tie-breaking solved all 3/3 by two hours with VAL-valid plans. The matched VH-on control remained 0/3 under all three rules. This is strong targeted mechanism evidence, not yet a domain-wide effect estimate.

## MPrime

The original anchor-rescore array has 18 live tasks. Six exact failed/held-index replacements are live. At the snapshot, 317/588 checkpoint validations are complete. The original tasks have at most 8h07m left in their 24-hour allocations; the replacements have nearly their full 24-hour bounds.

Controller `21233927` is dependency-pending. It audits paired done/summary markers, submits only incomplete lineage indices, and can repeat this bounded recovery up to four times before submitting the analysis-only coefficient finalizer. It cannot launch Stage-2 training itself: the frozen coefficient still requires manual curve review. Once frozen, the approved path is twenty fresh validation-led Stage-2 lineages, policy curves/endpoints, then matched fixed MCTS. Existing mismatched Stage-2 lineages are not duplicated or reused as canonical evidence.

MPrime PW has not been run. It is gated behind canonical Stage-2 endpoints and the fixed-MCTS baseline; the efficient next step is a two-seed PW70 screen, not immediate ten-seed confirmation.

Phase C itself is complete. It removed the original validation saturation, but Phase-B replicate A had the best available rank/stability trade-off and is frozen as the final MPrime validator. The current rescore is the required bridge from that validator decision to defensible Stage-2 training.

## FO Counters Stage-2 fixed MCTS

The last exact recovery `21219947` completed after 6h01m. Its requested instance consumed the full six-hour per-instance budget and produced no plan. Therefore the VH-off result is final, not partial: policy 2.9/20 versus MCTS 6.1/20 at 30m, 2h and 6h; paired change +3.2, 95% CI [1.74, 4.66], raw p=.00195, Holm p=.00977.

Row-level source training jobs, policy jobs, MCTS logs and durable ledger paths are in `fo_stage2_validation_vh_off_exact_seed_results_20260913.csv` and the canonical RQ CSVs.

## Adaptive-KL screen — recently completed

Jobs `21144388` and `21144389` both completed 100 epochs. The coefficient remained at its minimum value 3 for every update because post-update KL never crossed the predeclared target 0.1143; the controller therefore made zero adjustments. The catastrophic seed was still 10/20 at epoch 0 versus 20/20 for the stable control. This parameterization did not activate and did not repair the first-update collapse. It is a negative/inconclusive activation screen, not live work and not evidence against adaptive control in general.

# Reporting corrections — 6 September 2026

The 5 September PW70 table incorrectly copied PW20 scores into several fixed
comparator columns. The corrected mixed table is
`mcts_progressive_widening_cross_domain/mixed_pw20_pw70_comparison_20260906.csv`.
For example, BG/on fixed 6h is 17.0, PW20 is 18.0, and PW70 is 13.5.
Counters S1/off fixed is 21.5 while PW20 is 20.5. Scores were not re-evaluated;
the labeled column mapping was repaired from the existing evidence.

The read-only completion summarizer also had two parsing defects: its integer
regex captured the trailing zero in decimal EVAL FINAL scores, and it did not
recognize the separate `VAL-valid plans` / `VAL-invalid plans` lines. It now
parses decimal scores and the actual validator format. Coverage is independently
cross-checked against the compact per-instance records and VAL counts. None of
these parser repairs submits inference.

Five seeds was a staged screen and was insufficient for a two-sided exact
sign-flip p below .05 (the minimum is .0625). It should not have been described
as sufficient for conventional confirmatory significance. The 20-job extension
completes ten original seeds in FO Counters/Rover S1, both VH modes. Ten seeds
improve resolution and uncertainty estimation but do not guarantee significance.

Post-hoc per-instance cutoffs are a defensible description of the number of
recorded solutions found within a specified time. They do not directly measure
fresh whole-job runtime, changed scheduling, memory contention, or outcomes on
instances censored by allocation failure. Those limitations motivate a fresh
capped run only when that additional execution-level claim is needed.

Job 21039344 post-hoc validated the three new partial Counters records:
20974363:22 unique plans;20863187:21;20838267:29. All passed VAL with zero
invalid plans. Their OOM/failure/timeout statuses remain visible and are not
relabelled normal-completed. Two live jobs were automatically requeued at06:03;
their current Slurm elapsed values understate total historical compute.

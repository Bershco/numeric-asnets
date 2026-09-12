# Counters MCTS root-visit tie-break comparison

This six-task matched diagnostic changes only how equal maximum root visit
counts are resolved after an otherwise identical narrow MCTS search.

| Mode | Rule after maximum visits | Final exact tie |
|---|---|---|
| `action_id` | Historical behavior | Lowest global action index |
| `q` | Best sign-correct child Q | Lowest global action index when Q differs by at most `1e-8` |
| `policy` | Largest root network prior | Lowest global action index |

Each mode ran both the VH-off failure checkpoint and matched VH-on checkpoint
on Counters instances 51, 55 and 59. Every other declared choice is frozen:
seed 1963100312, validation-led Stage-2 checkpoints, five retained children,
20 simulations, PUCT 0.1, estimator mixture 0.5, one worker, 10,000 external
actions and six hours per instance. Goal chasing and safety/eligibility masks
retain precedence over all three root tie-break rules.

## Result

All six tasks completed. Historical action-index tie-breaking and Q
tie-breaking solved 0/3 in both value-head modes. Policy-prior tie-breaking
solved all three targeted VH-off failures by two hours (0/3 at 30 minutes,
3/3 at two and six hours); all three plans are VAL-valid. The matched VH-on
policy-tie arm remained 0/3.

This is causal evidence for the selected failure mechanism: when narrow MCTS
produces equal maximum visit counts and effectively equal Q values, arbitrary
global action order can discard useful policy ordering. It is not yet a
domain-wide coverage estimate because it covers one seed and three deliberately
selected failure instances. The next confirmatory experiment should apply the
policy tie-break to a predeclared multi-seed Counters evaluation set.

Each task requested 2 CPUs, 120 GiB and 24 hours. Actual elapsed times were
4:33--18:02. The exact result/provenance ledger is
`counters_tie_break_3way_20260911/results_latest.csv`; compact milestone files
are cached under `counters_tie_break_3way_20260911/compact_results/`.

The manifest is
`advisor_followup_20260910/counters_tie_break_3way_manifest.csv`. Every row now
carries its final job ID and output-log path for direct provenance.

Implementation commit `de1d29b83c6900e15de772074f8dd5595803dbbd` is
deployed in the isolated `numeric-asnets-tie-break` checkout. Compute smoke
`21185508` passed all 23 tests on a compute node and released all six running
tasks in array `21185509[0-5]`. Initial smoke `21185504` failed only
because it addressed the test through an installed-package module path; its
dependent array `21185505` was cancelled before allocating or running any
scientific task. The replacement discovers the same 23-test file directly.

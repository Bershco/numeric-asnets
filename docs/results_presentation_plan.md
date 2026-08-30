# Thesis results presentation plan

This file defines the advisor-ready results package. Static plots must be built
from authoritative experiment CSVs, never by reparsing completed raw logs.
Network seed is the independent unit; problem instances are descriptive
within-seed outcomes, not independent replicates.

## Main figures

1. **Original RQs: Stage 1 to Stage 2.** A paired forest plot for MAIN-VAL and
   MAIN-TERM showing mean seed-level coverage change and 95% paired confidence
   intervals for every domain/VH cell. A compact companion table reports raw
   exact sign-flip p-values and Holm-adjusted p-values.
2. **Learning dynamics.** Small-multiple Stage-1 and Stage-2 policy learning
   curves per domain/VH using mean, median and best curves plus the full-run
   min-max envelope. Curves remain separate by lineage and never forward-fill
   missing epochs. Show `n_runs` and the selected/final checkpoints.
3. **Policy versus MCTS.** A paired forest plot of seed-level MCTS-policy
   coverage changes, accompanied by a stacked outcome bar for completed,
   per-instance-timeout, scheduler-timeout and OOM lineages.
4. **Search efficiency/coverage frontier.** A coverage-versus-runtime Pareto
   plot comparing policy, fixed top-20, fixed top-5, progressive widening and
   relevant narrow-search arms. Encode retained nodes or peak memory as point
   size only when the same measurement exists across arms.

## Focused experimental figures

5. **Progressive widening.** Matched coverage lines by seed; successful-instance
   runtime ECDF/boxplots; compact child-count, visit-count and depth-band
   histograms. Report the coverage denominator beside the conditional runtime
   distribution so faster successful cases cannot hide fewer successes.
6. **Counters width sensitivity.** Per-seed policy versus normal 20/70 versus
   narrow 5/20 coverage, plus runtime and timeout/OOM outcomes. Stage 2 is
   policy versus narrow only because normal Stage-2 MCTS is excluded by design.
7. **MCTS-SAFE.** A four-row targeted repair table for SAFE-1, followed by
   SAFE-CONTEXT collision multipliers/policy-disagreement distributions and,
   when terminal, the paired physical/contextual Drone coverage-runtime-memory
   comparison.
8. **Finite horizon.** Paired aware/unaware coverage and runtime, plus cutoff
   count and cutoff-depth histograms. Separate nonbinding instances from the
   jobs that actually exercised the remaining-horizon boundary.
9. **MPrime validation redesign.** Corrected validation/test scatter and rank
   agreement, validation and test learning curves, selected-versus-final
   checkpoints, and the old epoch-1-biased distribution only as provenance.
10. **Preservation domains.** For Delivery, TPP and Zenotravel, paired panels
    comparing Stage-1 validation-selected to Stage-2 validation-selected and
    Stage-1 final to Stage-2 validation-selected, with confidence intervals and
    exact paired tests once all ten seeds are available.

## Deliverables

The main thesis/advisor package should contain four primary figures (items
1--4), three appendix figures selected from items 5--10, one policy-results
table and one search-results table. Every figure exports SVG, PNG, interactive
HTML when useful, and an aggregate CSV. Plot captions state checkpoint rule,
VH mode, seed count, search configuration, timeout treatment and whether the
p-value is raw or multiplicity-corrected.

## Concrete file locations

The generated advisor package is indexed in
`experiment_tracking/advisor_figures/README.md`. Available now:

- `experiment_tracking/advisor_figures/mainstream_stage1_stage2_forest.{svg,png}`
  with `mainstream_stage1_stage2_forest_data.csv`;
- `experiment_tracking/advisor_figures/pw_kmin3_30min_coverage.{svg,png}`
  with `pw_kmin3_30min_sensitivity.csv`;
- `experiment_tracking/advisor_figures/pw_kmin3_success_runtime_ecdf.{svg,png}`;
- `experiment_tracking/advisor_figures/drone_mainterm_policy_mcts.{svg,png}`;
- `experiment_tracking/mprime_validation_ipc_scale_v1/corrected_learning_curves.{svg,png}`
  and its aggregate CSV.

The exact-checkpoint policy recaptures finished and reproduced all eight
authoritative scores. They confirmed whole-job policy runtimes of only minutes,
but the current policy evaluator still emits no per-instance elapsed record.
PRESERVE-4, Binding Horizon and SAFE-CONTEXT remain live; their final figures
must be regenerated only after their committed ledgers become terminal.

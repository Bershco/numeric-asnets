# Advisor meeting package — 10 September 2026

Start with `../advisor_report_20260909_current.md`. It contains the compact
story, all RQ result tables, complete experiment catalog, best result by domain,
live status, and the provenance contract.

## Recommended presentation order

1. `after_review/01_domain_scorecard.png` — concrete achievements by domain.
2. `after_review/02_two_stage_learning_dynamics.png` — why Stage-2 policy
   refinement is not the main positive result.
3. `after_review/03a_stage1_mcts_cutoff_forest.png` and
   `after_review/03b_stage2_mcts_cutoff_forest.png` — the inference-time wins,
   including 30-minute, 2-hour and 6-hour cutoffs.
4. `after_review/04_preserve3_validation_seed_robustness.png` — near-ceiling
   preservation plus the rare TPP failure.
5. `after_review/05_mprime_validation_problem.png` — why the only important
   live methodological experiment must finish before freezing MPrime.

The first versions remain under `before_review/`. The blind review and exact
revision rationale are in `independent_plot_review.md`.

Every figure's source rows are colocated as CSV or linked from
`artifact_index.csv`. Those result CSVs contain either direct job/log paths or
a `row_level_provenance` link to the seed-level source ledger.

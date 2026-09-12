# Advisor follow-up package — current index

This directory implements the decisions from the 10 September 2026 advisor
meeting. The canonical, current research-question narrative is
[`rq_report_validation_led_20260912.md`](rq_report_validation_led_20260912.md).
It is generated from the row-level CSV evidence in this directory and uses the
following fixed reporting contract:

- validation-led evidence is primary;
- terminal-led evidence is retained only in `terminal_led_archive_index.csv`;
- RQ1–RQ4 are reported separately;
- RQ3 and RQ4 include both direct VH-on effects and comparisons with the
  parallel VH-off cells;
- MCTS results report 30-minute, two-hour and six-hour per-instance cutoffs;
- fixed search and progressive widening remain separate statistical families;
- partial cells use `≥` and do not receive final confidence intervals or
  p-values.

## Current experimental decisions

- **MPrime Phase C is complete:** 347/347 evaluations and 40/40 primary
  lineages. It did not improve aggregate selection regret over Phase-B
  replicate A (1.650 versus 1.625 plans on the common reference). Phase-B
  replicate A is therefore frozen as the final validator. Existing MPrime
  Stage-2 networks are not admitted to the primary RQs because most were
  trained from superseded Stage-1 source checkpoints. See
  `../mprime_validation_phase_c_20260911/README.md`.
- **Counters tie-breaking screen is complete:** policy-prior tie-breaking
  recovered all three preselected VH-off failures by two hours; action-index
  and Q tie-breaking recovered none. The VH-on controls recovered none. This is
  strong targeted causal evidence, not yet a domain-wide estimate. See
  `../counters_tie_break_3way_20260911.md`.
- **FO Counters Stage-2 fixed MCTS:** the broad recovery finished. One exact
  evaluator instance remained non-durable after a worker exit and is being run
  alone as job `21219947`. The current VH-off mean is therefore `≥6.1/20` and
  can rise only to `6.2/20`.
- **MPrime anchor reranking is submitted:** smoke job `21221744` gates the
  28-task saved-checkpoint array `21221745[0-27]`; analysis-only finalizer
  `21221746` proposes the per-VH coefficient and cannot submit training.

## Canonical tables and plots

- `rq_primary_validation_led.csv` — paired fixed-search effects, CIs, raw and
  Holm-adjusted p-values, with provenance.
- `rq2_raw_means_validation_led.csv` and
  `rq4_raw_means_validation_led.csv` — raw policy/MCTS levels.
- `rq2_pw70_branch_latest.csv` and `rq4_pw70_branch_latest.csv` — ten-seed
  Stage-1 PW70 confirmation, with all three cutoffs.
- `rq1_stage2_training_vh_off.{svg,png}` — RQ1.
- `rq2_raw_means_by_stage.{svg,png}` and `rq2_mcts_vh_off.{svg,png}` — RQ2.
- `rq3_raw_means_and_interaction.{svg,png}` and
  `rq3_value_head_training.{svg,png}` — RQ3.
- `rq4_raw_means_6h_by_stage.{svg,png}`, `rq4_direct.{svg,png}`,
  `rq4_cross_cell.{svg,png}`, and `rq4_interaction.{svg,png}` — RQ4.
- `rq2_rq4_pw70_final.{svg,png}` — PW70 contribution to RQ2/RQ4.
- `result_csv_provenance_index_latest.csv` at the experiment-tracking root —
  file-level route back to raw training/evaluation evidence.

## Reproduction and follow-up material

- `paper_to_thesis_change_inventory.{md,csv}` — scientific differences from
  the original paper workflow.
- `generator_comparison.csv`, `generator_distribution_*.csv`, and
  `yarin_external_*` — generator audit and external-distribution screen.
- `counters_visit_audit_*` and `counters_tie_break_*` — Counters mechanism
  evidence and job manifests.
- `fo_stage2_validation_partial_recovery_manifest.csv` and
  `fo_stage2_validation_exact_recovery_20260912.csv` — exact FO recovery
  provenance.

The previous long-form version of this README contained time-sensitive queue
snapshots and is intentionally replaced by this index; Git history preserves
it. Operational state belongs in `../cluster_workload_latest.csv` and
`../live_experiment_status_latest.csv`.

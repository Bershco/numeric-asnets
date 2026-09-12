# Canonical experiment-documentation index

Updated: 2026-09-12T17:59:28+03:00

This index resolves the apparent duplication created by dated audit snapshots.
Dated files are immutable historical evidence, not current status.  New analysis
must read the canonical files below; it must not infer liveness from an old date.

| Purpose | Canonical file |
|---|---|
| Current Slurm jobs | `experiment_tracking/cluster_workload_latest.csv` |
| Current workload totals | `experiment_tracking/cluster_workload_summary_latest.csv` |
| Master experiment registry | `experiment_tracking/experiment_registry.csv` |
| Joined current catalog | `experiment_tracking/experiment_catalog_latest.csv` |
| RQ-separated primary statistics | `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv` |
| RQ2 raw levels | `experiment_tracking/advisor_followup_20260910/rq2_raw_means_validation_led.csv` |
| RQ3 raw levels and interaction | `experiment_tracking/advisor_followup_20260910/rq3_raw_means_validation_led.csv` |
| RQ4 raw levels | `experiment_tracking/advisor_followup_20260910/rq4_raw_means_validation_led.csv` |
| RQ2 PW70 branch | `experiment_tracking/advisor_followup_20260910/rq2_pw70_branch_latest.csv` |
| RQ4 PW70 branch | `experiment_tracking/advisor_followup_20260910/rq4_pw70_branch_latest.csv` |
| Advisor package index | `experiment_tracking/advisor_followup_20260910/README.md` |
| Current validation-led RQ narrative, tables and plots | `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md` |
| Dynamic job evidence | `experiment_tracking/dynamic_experiment_jobs_latest.csv` |
| MPrime Phase-B checkpoint evidence | `experiment_tracking/mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv` |
| MPrime Phase-B selector comparison | `experiment_tracking/mprime_validation_phase_b_20260906/phase_b_cell_selector_comparison_latest.csv` |
| Counters visit milestones | `experiment_tracking/advisor_followup_20260910/counters_visit_audit_latest_milestones.csv` |
| Counters tie-break counterfactual | `experiment_tracking/advisor_followup_20260910/counters_tie_break_counterfactual_latest.csv` |
| MPrime Phase-C design/provenance | `experiment_tracking/mprime_validation_phase_c_20260911/README.md` |
| MPrime Phase-C checkpoint candidates | `experiment_tracking/mprime_validation_phase_c_20260911/checkpoint_candidates.csv` |
| MPrime Phase-C completed cell summary | `experiment_tracking/mprime_validation_phase_c_20260911/phase_c_cell_summary_latest.csv` |
| MPrime final validator decision | `experiment_tracking/mprime_validation_phase_c_20260911/validator_decision_latest.csv` |
| Counters three-way tie-break result | `experiment_tracking/counters_tie_break_3way_20260911/results_latest.csv` |
| FO recovery progress | `experiment_tracking/advisor_followup_20260910/fo_stage2_validation_recovery_progress_latest.csv` |
| FO final exact single-instance recovery | `experiment_tracking/advisor_followup_20260910/fo_stage2_validation_exact_recovery_20260912.csv` |
| Adaptive-KL summary | `experiment_tracking/anchor_kl_control_summary_latest.csv` |
| CSV provenance audit | `experiment_tracking/result_csv_provenance_index_latest.csv` |
| Registry reference integrity | `experiment_tracking/registry_reference_audit_latest.csv` |
| Historical experiment-ID aliases | `experiment_tracking/experiment_id_aliases.csv` |
| Byte-identical duplicate audit | `experiment_tracking/documentation_redundancy_audit_latest.csv` |

## Retention rule

- Keep dated snapshots because they prove what was known at a particular time.
- Keep one current alias for each changing concept (`*_latest` or the master path).
- Do not cite an old `status_YYYYMMDD` file as current state.
- Current statistical tables must contain direct job/log fields or point to a
  seed-level ledger that contains them.

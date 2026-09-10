# Canonical experiment-documentation index

Updated: 2026-09-10T21:28:16+03:00

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
| Advisor narrative and tables | `experiment_tracking/advisor_followup_20260910/README.md` |
| Dynamic job evidence | `experiment_tracking/dynamic_experiment_jobs_latest.csv` |
| MPrime Phase-B checkpoint evidence | `experiment_tracking/mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv` |
| CSV provenance audit | `experiment_tracking/result_csv_provenance_index_latest.csv` |
| Byte-identical duplicate audit | `experiment_tracking/documentation_redundancy_audit_latest.csv` |

## Retention rule

- Keep dated snapshots because they prove what was known at a particular time.
- Keep one current alias for each changing concept (`*_latest` or the master path).
- Do not cite an old `status_YYYYMMDD` file as current state.
- Current statistical tables must contain direct job/log fields or point to a
  seed-level ledger that contains them.

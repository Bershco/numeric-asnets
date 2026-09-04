#!/usr/bin/env python3
"""Refresh authoritative tracking registries for the 2026-09-04 snapshot."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


registry_path = TRACK / "experiment_registry.csv"
registry = read(registry_path)
updates = {
    "PRESERVE-3-TERM": {
        "status": "endpoint-complete-learning-curve-tail-live",
        "results_file": "experiment_tracking/preserve3_terminal_led/terminal_led_selected_summary_20260903.csv",
        "held_reason": "",
        "next_action": "All six ten-seed selected-endpoint comparisons are complete. Finish the remaining TPP every-five policy retries to close the learning-curve artifacts.",
    },
    "MPRIME-VAL-ADEQUACY": {
        "status": "phase-a-complete-phase-b-held-top-priority",
        "results_file": "experiment_tracking/mprime_validation_ipc_scale_v1/validation_adequacy_phase_a_stage2_summary_20260903.csv",
        "held_reason": "Two independent harder validation sets must be generated and checksum-frozen before network scores are inspected.",
        "next_action": "Phase A is 290/290 Stage1 and 840/840 Stage2 checkpoint scores. Freeze two stratified IPC-scale replicate sets, then rescore existing checkpoints only.",
    },
    "MCTS-HORIZON-COUNTERS": {
        "status": "live-one-job-tail",
        "results_file": "experiment_tracking/mcts_horizon_binding/horizon_completion_records_20260902.csv",
        "next_action": "Seven of eight jobs are terminal; one unaware VH-on job is within roughly four hours of its 72-hour bound. Audit cutoff counters before any SAFE2 work.",
    },
    "MCTS-PW70-CROSS-DOMAIN": {
        "status": "live-two-job-tail",
        "results_file": "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_cutoffs_20260904.csv",
        "next_action": "Ten of twelve jobs are terminal; two Counters Stage1/off jobs run. Freeze all six two-seed cells when they terminate.",
    },
    "MCTS-PW70-CONFIRMATORY": {
        "status": "live-counters-tail-four-cells-complete",
        "results_file": "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_statistics_20260904.csv",
        "next_action": "FO Counters and Rover are complete at five matched seeds per VH. Four Counters Stage2 jobs remain live; finish them before the six-cell family conclusion.",
    },
    "MCTS-PW-COUNTERS-DIVERGENCE": {
        "status": "live-six-jobs",
        "results_file": "experiment_tracking/mcts_progressive_widening_cross_domain/counters_divergence_cutoffs_20260904.csv",
        "held_reason": "",
        "next_action": "All six exact-snapshot PW20/PW70 jobs are live; compare recovery of policy successes lost by fixed narrow search at 30m, 2h and 6h.",
    },
    "MCTS-LEGACY-FO": {
        "status": "live-thirteen-job-tail",
        "next_action": "Seven terminal-led endpoints are terminal and thirteen run. Reconcile fixed-budget 30m/2h/6h scores as each endpoint terminates.",
    },
    "MCTS-STAGE2-BRANCH-COMPLETION": {
        "status": "live-bg-fo-rover-counters-gated",
        "results_file": "experiment_tracking/stage2_mcts_branch_coverage_20260904.csv",
        "manifest_path": "experiment_tracking/stage2_mcts_branch_completion_approved_20260903.csv;experiment_tracking/stage2_mcts_rover_completion_approved_20260904.csv;experiment_tracking/stage2_mcts_counters_terminal_completion_ready_20260904.csv",
        "held_reason": "Counters terminal-led 20-job manifest remains locally ready but unsubmitted until all nineteen Rover completion identities have active replacements running and capacity remains.",
        "next_action": "BG narrow5/20, FO normal20/70 and Rover normal20/70 are submitted. Replace only pre-inference infrastructure failures. Do not submit Counters before its explicit gate.",
    },
}
for row in registry:
    if row["experiment_id"] in updates:
        row.update(updates[row["experiment_id"]])
write(registry_path, registry, list(registry[0]))

coverage_plan_path = TRACK / "evaluation_coverage_plan.csv"
coverage_plan = read(coverage_plan_path)
for row in coverage_plan:
    if row["experiment_id"] == "PRESERVE-3-TERM":
        row["status"] = (
            "60/60 training lineages terminal; all six ten-seed selected-endpoint "
            "comparisons complete; TPP every-five curve retries live"
        )
        row["next_action"] = (
            "finish the remaining TPP curve retries; no training or selected-endpoint "
            "evaluation is missing"
        )
write(coverage_plan_path, coverage_plan, list(coverage_plan[0]))

coverage = [
    ("block_grouping", "off", 8, 2, 0, 10, 0, 0, "validation branch completion live; narrow5/20"),
    ("block_grouping", "on", 8, 2, 0, 10, 0, 0, "validation completion live and terminal branch complete; narrow5/20"),
    ("drone", "off", 10, 0, 0, 10, 0, 0, "both branches complete; normal20/70"),
    ("drone", "on", 10, 0, 0, 10, 0, 0, "both branches complete; normal20/70"),
    ("fo_counters", "off", 6, 4, 0, 4, 6, 0, "validation completion plus terminal-led campaign live; normal20/70"),
    ("fo_counters", "on", 9, 1, 0, 3, 7, 0, "validation completion plus terminal-led campaign live; normal20/70"),
    ("rover", "off", 1, 9, 0, 10, 0, 0, "validation completion submitted; normal20/70"),
    ("rover", "on", 0, 10, 0, 10, 0, 0, "validation completion submitted; normal20/70"),
    ("counters", "off", 10, 0, 0, 0, 0, 10, "validation complete narrow5/20; terminal manifest ready but gated"),
    ("counters", "on", 10, 0, 0, 0, 0, 10, "validation complete narrow5/20; terminal manifest ready but gated"),
]
workload = read(TRACK / "cluster_workload_summary_latest.csv")
snapshot_time = workload[0]["snapshot_time_idt"] if workload else ""
coverage_rows = [{
    "snapshot_time_idt": snapshot_time,
    "domain": domain, "value_head": vh,
    "validation_terminal": vt, "validation_live_submitted": vl, "validation_unsubmitted": vm,
    "terminal_terminal": tt, "terminal_live_submitted": tl, "terminal_unsubmitted": tm,
    "notes": notes,
    "row_level_provenance": "experiment_tracking/stage2_mcts_historical_log_audit_20260902.csv;experiment_tracking/stage2_mcts_branch_completion_original_results_20260904.csv;experiment_tracking/cluster_workload_latest.csv",
} for domain, vh, vt, vl, vm, tt, tl, tm, notes in coverage]
write(TRACK / "stage2_mcts_branch_coverage_20260904.csv", coverage_rows)

live_rows = []
for row in workload:
    live_rows.append({
        "snapshot_time_idt": row["snapshot_time_idt"],
        "experiment": row["experiment"], "state": row["state"],
        "jobs": row["jobs"], "requested_cpus": row["requested_cpus"],
        "requested_memory_gib": row["requested_memory_gib"],
        "scheduler_source": "experiment_tracking/cluster_workload_latest.csv",
    })
write(TRACK / "live_experiment_status_latest.csv", live_rows)

index_path = TRACK / "result_provenance_index_20260902.csv"
index_rows = read(index_path)
new_mappings = {
    "experiment_tracking/preserve3_terminal_led/terminal_led_selected_summary_20260903.csv": ("PRESERVE-3 terminal-led aggregate", "experiment_tracking/preserve3_terminal_led/terminal_led_selected_seed_results_20260903.csv", "complete_companion_live"),
    "experiment_tracking/preserve3_terminal_led/delivery_policy_retry_results_20260904.csv": ("Delivery exact curve/endpoint recovery", "source_log;source_checkpoint_ref;slurm_job_id", "complete_inline"),
    "experiment_tracking/preserve3_terminal_led/tpp_on_policy_results_latest_20260904.csv": ("TPP/on original policy evaluations", "source_log;slurm_job_id", "complete_inline_live"),
    "experiment_tracking/preserve3_terminal_led/tpp_on_policy_node_retry_results_20260904.csv": ("TPP/on exact retry results", "source_log;slurm_job_id", "complete_inline_live"),
    "experiment_tracking/preserve3_terminal_led/tpp_on_policy_node_retry2_results_20260904.csv": ("TPP/on exact second-retry results", "source_log;slurm_job_id", "complete_inline"),
    "experiment_tracking/preserve3_terminal_led/tpp_on_policy_node_retry3_results_20260904.csv": ("TPP/on exact third-retry results", "source_log;slurm_job_id", "complete_inline_live"),
    "experiment_tracking/preserve3_terminal_led/terminal_stage2_tpp_off_policy_results_20260904.csv": ("TPP/off original policy evaluations", "source_log;slurm_job_id", "complete_inline"),
    "experiment_tracking/preserve3_terminal_led/tpp_off_curve_node_retry_results_20260904.csv": ("TPP/off exact curve retry results", "source_log;slurm_job_id", "complete_inline_live"),
    "experiment_tracking/preserve3_terminal_led/tpp_off_policy_retry2_results_latest_20260904.csv": ("TPP/off exact endpoint retry results", "source_log;slurm_job_id", "complete_inline"),
    "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_cutoffs_20260904.csv": ("PW70 correction 30m/2h/6h results", "source_log;experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_instances_20260904.csv", "complete_inline_live"),
    "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_statistics_20260904.csv": ("PW70 five-seed completed-cell statistics", "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_seed_results_20260904.csv", "complete_companion"),
    "experiment_tracking/mcts_progressive_widening_cross_domain/counters_divergence_cutoffs_20260904.csv": ("Counters exact-snapshot PW20/PW70 live cutoffs", "source_log;experiment_tracking/mcts_progressive_widening_cross_domain/counters_divergence_instances_20260904.csv", "complete_inline_live"),
    "experiment_tracking/stage2_mcts_branch_coverage_20260904.csv": ("Current Stage2 MCTS branch availability", "row_level_provenance", "complete_companion_live"),
    "experiment_tracking/stage2_mcts_branch_completion_current_results_20260904.csv": ("Current branch-completion job evidence", "source_evaluation_log;job_id", "complete_inline_live"),
}
by_path = {row["authoritative_result_file"]: row for row in index_rows}
for path, (scope, companion, state) in new_mappings.items():
    by_path[path] = {
        "authoritative_result_file": path, "scope": scope,
        "row_level_log_columns_or_companion": companion, "provenance_status": state,
    }
write(index_path, list(by_path.values()), list(index_rows[0]))
print("updated experiment registry, branch coverage, live status, and provenance index")

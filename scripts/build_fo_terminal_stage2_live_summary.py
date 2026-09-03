#!/usr/bin/env python3
"""Join terminal FO Stage-2 MCTS evidence to its matched policy endpoints."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "experiment_tracking" / "fo_terminal_stage2_mcts_reconciliation_20260903.csv"
POLICY = ROOT / "experiment_tracking" / "experiment_results.csv"
OUT = ROOT / "experiment_tracking" / "fo_terminal_stage2_mcts_terminal_results_20260903.csv"
SUMMARY = ROOT / "experiment_tracking" / "fo_terminal_stage2_mcts_terminal_summary_20260903.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    policy = {
        (row["value_head"], row["seed"]): row
        for row in read(POLICY)
        if row["experiment_id"] == "MAIN-TERM"
        and row["domain"] == "fo_counters"
        and row["stage"] == "stage2"
        and row["endpoint"] == "validation_selected"
    }
    rows: list[dict[str, str]] = []
    terminal_states = {"COMPLETED", "OUT_OF_MEMORY", "TIMEOUT", "FAILED", "CANCELLED"}
    for mcts in read(RECON):
        if mcts["domain"] != "fo_counters" or mcts["slurm_state"] not in terminal_states:
            continue
        matched = policy[(mcts["value_head"], mcts["seed"])]
        successful_times = [
            float(value) for value in mcts["successful_instance_runtime_seconds"].split(";") if value
        ]
        rows.append({
            "experiment_id": "MCTS-LEGACY-FO",
            "branch": "terminal_led",
            "job_id": mcts["job_id"],
            "value_head": mcts["value_head"],
            "seed": mcts["seed"],
            "policy_score": matched["score"],
            "mcts_30m_score": str(sum(value <= 1800 for value in successful_times)),
            "mcts_2h_score": str(sum(value <= 7200 for value in successful_times)),
            "mcts_6h_score": str(sum(value <= 21600 for value in successful_times)),
            "total": mcts["total"],
            "slurm_state": mcts["slurm_state"],
            "exit_code": mcts["exit_code"],
            "elapsed": mcts["elapsed"],
            "val_valid": mcts["val_valid"],
            "val_invalid": mcts["val_invalid"],
            "source_training_job_id": mcts["source_training_job_id"],
            "snapshot_epoch": mcts["snapshot_epoch"],
            "source_checkpoint": matched["checkpoint"],
            "source_training_log": matched["source_training_log"],
            "source_policy_log": matched["source_evaluation_log"],
            "source_mcts_log": mcts["source_evaluation_log"],
        })
    fields = list(rows[0]) if rows else ["experiment_id"]
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for value_head in ("off", "on"):
        group = [row for row in rows if row["value_head"] == value_head]
        if not group:
            continue
        summaries.append({
            "experiment_id": "MCTS-LEGACY-FO",
            "branch": "terminal_led",
            "value_head": value_head,
            "matched_n": len(group),
            "policy_mean": sum(float(row["policy_score"]) for row in group) / len(group),
            "mcts_30m_mean": sum(float(row["mcts_30m_score"]) for row in group) / len(group),
            "mcts_2h_mean": sum(float(row["mcts_2h_score"]) for row in group) / len(group),
            "mcts_6h_mean": sum(float(row["mcts_6h_score"]) for row in group) / len(group),
            "row_level_companion": "experiment_tracking/fo_terminal_stage2_mcts_terminal_results_20260903.csv",
        })
    fields = list(summaries[0]) if summaries else ["experiment_id"]
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"rows={len(rows)} summaries={len(summaries)}")


if __name__ == "__main__":
    main()

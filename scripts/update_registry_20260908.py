#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "experiment_tracking"

UPDATES = {
    "MPRIME-VAL-ADEQUACY": ("live-60-lineage-full-rescore",
        "1050/2260 checkpoint-replicates complete at 2026-09-08 10:57 IDT; all 60 tasks running; submit only resumable remainder after this allocation."),
    "MCTS-PW70-TEN-SEED": ("terminal-three-complete-cells-one-n9",
        "FO/on and both Rover cells have 10 usable seeds; FO/off has 9 plus one OOM-partial."),
    "TPP-CATASTROPHIC-MCTS": ("two-terminal-two-running",
        "Fixed narrow ended 4/20 OOM; PW20 ended 10/20 OOM with all 10 plans VAL-valid; fixed normal >=4 and PW70 >=7 remain live versus policy 9."),
    "MCTS-STAGE2-BRANCH-COMPLETION": ("four-running-tails",
        "FO terminal/on one and Counters terminal three remain live; finalize only after terminal reconciliation."),
    "MCTS-PW70-CROSS-DOMAIN": ("live-one-job-tail",
        "Counters Stage1/off seed 2011206605 remains live at >=21/59 with 42 classified."),
    "MCTS-PW-COUNTERS-DIVERGENCE": ("pw20-complete-pw70-two-terminal-one-live",
        "Seed 534933607 PW70 remains live at >=22/59; other PW70 scores are 21 and 19."),
    "MCTS-LEGACY-FO": ("live-one-job-tail",
        "Final terminal-led VH-on seed is live at >=5/20 with 17 classified."),
}

NEW = {
    "experiment_id": "ROVER-MCTS-INTERRUP-REC",
    "display_name": "Rover interrupted-instance recovery",
    "role": "operational-completion",
    "status": "25-of-29-classified-four-active",
    "scope": "Exact interrupted Rover MCTS instances across Stage1 and Stage2 branches",
    "primary_question": "Classify only instances omitted by scheduler or OOM interruption",
    "configuration_summary": "Preserve source checkpoint and width/iteration; six-hour instance cap; one-worker follow-up for cleanup-failed instances",
    "results_file": "experiment_tracking/dynamic_experiment_jobs_latest.csv",
    "manifest_path": "experiment_tracking/rover_interrupted_mcts_recovery_manifest_20260908.csv",
    "held_reason": "",
    "next_action": "Jobs 21107687_7 and 21114871_0 are running; finalize after both terminate.",
}

for path in (T / "experiments.csv", T / "experiment_registry.csv"):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames
    for row in rows:
        if row["experiment_id"] in UPDATES:
            row["status"], row["next_action"] = UPDATES[row["experiment_id"]]
        if row["experiment_id"] == "MCTS-PW70-TEN-SEED":
            row["results_file"] = "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_statistics_latest.csv"
        elif row["experiment_id"] == "MPRIME-VAL-ADEQUACY":
            row["results_file"] = "experiment_tracking/mprime_validation_phase_b_progress_latest.csv"
        elif row["experiment_id"] == "TPP-CATASTROPHIC-MCTS":
            row["results_file"] = "experiment_tracking/dynamic_experiment_jobs_latest.csv"
    if not any(row["experiment_id"] == NEW["experiment_id"] for row in rows):
        rows.append(NEW)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

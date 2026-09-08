#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "experiment_tracking"

UPDATES = {
    "MPRIME-VAL-ADEQUACY": ("live-60-lineage-full-rescore-68.5-percent",
        "1547/2260 checkpoint-replicates complete at 2026-09-08 23:39 IDT; six lineages complete. Exact idempotent continuation 21143254 plus predecessor/replacement tasks leave 54 MPrime tasks running."),
    "MCTS-PW70-TEN-SEED": ("terminal-four-ten-seed-cells-one-oom-partial",
        "All four cells contain ten declared-budget seeds; FO/off includes one explicitly starred 7/20 OOM-partial seed and has mean8.4."),
    "TPP-CATASTROPHIC-MCTS": ("three-terminal-one-running",
        "Fixed narrow ended 4/20 OOM; PW20 ended 10/20 OOM; PW70 ended OOM at 7/20 after 12 classified; fixed normal is 4/20 after 19 classified with one instance active; policy is 9/20."),
    "MCTS-STAGE2-BRANCH-COMPLETION": ("counters-terminal-fo-tail-live",
        "Counters terminal-led narrow is scheduler-terminal at off38.4 and on22.4 at6h; FO terminal/on retains one live resumable tail."),
    "MCTS-PW70-CROSS-DOMAIN": ("live-one-job-tail",
        "Counters Stage1/off seed 2011206605 remains live at >=21/59 with 47 classified."),
    "MCTS-PW-COUNTERS-DIVERGENCE": ("pw20-complete-pw70-two-terminal-one-live",
        "Seed 534933607 PW70 remains live at >=22/59 with 51 classified; other PW70 scores are 21 and 19."),
    "MCTS-LEGACY-FO": ("live-one-job-restarted-tail",
        "Final terminal-led VH-on seed is 5/20 after 19 durably classified; one instance remains active and should classify within six hours."),
    "ROVER-MCTS-INTERRUP-REC": ("27-of-29-classified-two-active",
        "Twenty-seven exact recovery opportunities are six-hour timeouts; job21107687_7 is running the final two independent instances and should finish within six hours of its 22:47 start."),
    "ANCHOR-KL-CONTROL": ("implemented-local-tested-awaiting-deploy",
        "Two adaptive-only jobs are ready because constant baselines already exist. Commits96a6d02d and9db17cc8 need explicit GitHub push approval before isolated-checkout pull, compute-node smoke, and submission."),
}

NEW = {
    "experiment_id": "ROVER-MCTS-INTERRUP-REC",
    "display_name": "Rover interrupted-instance recovery",
    "role": "operational-completion",
    "status": "27-of-29-classified-two-exact-opportunities",
    "scope": "Exact interrupted Rover MCTS instances across Stage1 and Stage2 branches",
    "primary_question": "Classify only instances omitted by scheduler or OOM interruption",
    "configuration_summary": "Preserve source checkpoint and width/iteration; six-hour instance cap; one-worker follow-up for cleanup-failed instances",
    "results_file": "experiment_tracking/dynamic_experiment_jobs_latest.csv",
    "manifest_path": "experiment_tracking/rover_interrupted_mcts_recovery_manifest_20260908.csv",
    "held_reason": "",
    "next_action": "Job21114871 classified instances16/17 as six-hour timeouts; job21107687_7 is running exact remaining instances17/19 with two independent workers and should finish within six hours of start.",
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
        elif row["experiment_id"] == "ANCHOR-KL-CONTROL":
            row["scope"] = "TPP/off collapsed seed 1972442430 and stable control seed 1963100312"
            row["configuration_summary"] = "Reuse constant-anchor baselines; run two adaptive target-KL jobs from exact Stage1 sources; start/min coefficient3; target KL0.1143; exact controller checkpoint restore"
            row["manifest_path"] = "experiment_tracking/anchor_kl_control_tpp_screen_20260908.csv"
            row["results_file"] = "experiment_tracking/anchor_kl_control_literature_and_design_20260902.md"
            row["held_reason"] = "Awaiting explicit authorization to push commits96a6d02d and9db17cc8 to GitHub; then compute-node smoke and submit"
    if not any(row["experiment_id"] == NEW["experiment_id"] for row in rows):
        rows.append(NEW)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

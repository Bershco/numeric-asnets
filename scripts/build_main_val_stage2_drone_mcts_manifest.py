#!/usr/bin/env python3
"""Materialize the 16 genuinely missing MAIN-VAL Drone Stage-2 MCTS jobs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment_tracking" / "experiment_results.csv"
GAP = ROOT / "experiment_tracking" / "main_val_stage2_drone_mcts_gap.csv"
OUTPUT = (
    ROOT / "experiment_tracking" / "main_val_stage2_drone_mcts_manifest.csv"
)

FIELDS = [
    "manifest_id", "cohort", "seed", "domain", "domain_label", "value_head",
    "rq_scope", "task_type", "stage", "variant", "architecture", "teacher",
    "source_checkpoint_ref", "supervised_lr", "max_epochs",
    "original_training_set", "estimator", "puct", "tree_sampling", "anchor",
    "width", "iterations", "workers", "jpddl_heap", "cpus", "memory",
    "time_limit", "instance_timeout", "completion_mode", "dependency_ref",
    "checkpoint_selection", "status", "notes", "source_training_job_id",
    "snapshot_epoch", "analysis_roles",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    endpoints = {
        (row["value_head"], row["seed"]): row
        for row in read(RESULTS)
        if row["experiment_id"] == "MAIN-VAL"
        and row["task_type"] == "policy_eval"
        and row["domain"] == "drone"
        and row["stage"] == "stage2"
        and row["endpoint"] == "validation_selected"
    }
    gaps = [row for row in read(GAP) if row["mcts_state"] == "ready_held"]
    if len(endpoints) != 20 or len(gaps) != 16:
        raise RuntimeError(
            f"Expected 20 policy endpoints and 16 gaps; got {len(endpoints)}, {len(gaps)}"
        )

    rows = []
    for gap in gaps:
        endpoint = endpoints[(gap["value_head"], gap["seed"])]
        if endpoint["source_training_job_id"] != gap["source_training_job_id"]:
            raise RuntimeError(f"Training lineage mismatch: {gap}")
        rows.append({
            "manifest_id": (
                f"main-val-drone-{gap['value_head']}-{gap['seed']}-"
                "stage2-selected-mcts-w20-i70"
            ),
            "cohort": "validation-selected-ten",
            "seed": gap["seed"],
            "domain": "drone",
            "domain_label": "Drone",
            "value_head": gap["value_head"],
            "rq_scope": "RQ1/RQ2",
            "task_type": "mcts_eval",
            "stage": "stage2",
            "variant": "main_val_stage2_selected_mcts_w20_i70",
            "architecture": "experiments_numeric.architecture_2.drone_mcts",
            "teacher": "hadd-astar",
            "source_checkpoint_ref": endpoint["checkpoint"],
            "supervised_lr": "",
            "max_epochs": "",
            "original_training_set": "True",
            "estimator": "0.5",
            "puct": "0.1",
            "tree_sampling": "",
            "anchor": "",
            "width": "20",
            "iterations": "70",
            "workers": "3",
            "jpddl_heap": "4g",
            "cpus": "6",
            "memory": "120G",
            "time_limit": "3-00:00:00",
            "instance_timeout": "21600",
            "completion_mode": "rolling+VAL",
            "dependency_ref": f"training:{gap['source_training_job_id']}",
            "checkpoint_selection": "validation_selected",
            "status": "ready",
            "notes": (
                "Confirmed absent after accounting audit of all Drone MCTS jobs; "
                "same selected checkpoint as the matched MAIN-VAL policy endpoint"
            ),
            "source_training_job_id": gap["source_training_job_id"],
            "snapshot_epoch": gap["snapshot_epoch"],
            "analysis_roles": "main_val_stage2_selected_policy_vs_mcts",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

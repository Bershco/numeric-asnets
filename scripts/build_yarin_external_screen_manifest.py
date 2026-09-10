#!/usr/bin/env python3
"""Build the approved FO Counters/Rover three-seed external test manifest."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "advisor_followup_20260910" / "yarin_external_screen_manifest.csv"
SEEDS = {"1073581256", "1963100312", "2011206605"}
DOMAINS = {"fo_counters", "rover"}


def main() -> None:
    with (TRACK / "experiment_results.csv").open(newline="", encoding="utf-8-sig") as stream:
        source = list(csv.DictReader(stream))
    selected = [row for row in source if row["experiment_id"] == "MAIN-VAL"
                and row["task_type"] == "policy_eval" and row["stage"] == "stage1"
                and row["endpoint"] == "validation_selected" and row["domain"] in DOMAINS
                and row["seed"] in SEEDS]
    if len(selected) != 12:
        raise ValueError(f"expected 12 unique checkpoints, found {len(selected)}")
    rows = []
    for index, row in enumerate(sorted(selected, key=lambda item: (item["domain"], item["value_head"], item["seed"]))):
        rows.append({
            "array_index": index, "experiment_id": "EXTERNAL-GENERATOR-SCREEN",
            "domain": row["domain"], "value_head": row["value_head"], "seed": row["seed"],
            "source_training_job_id": row["source_training_job_id"],
            "source_policy_job_id": row["job_id"], "source_checkpoint": row["checkpoint"],
            "external_generator_repo": "https://github.com/SPL-BGU/pddl-problem-generator",
            "external_generator_commit": "b64e5d086117ebd5c1d53fbe9a9d93ae609a59fb",
            "external_freeze_seed": "20260910", "instances": 20, "workers": 6,
            "cpus": 6, "memory_gib": 40, "walltime": "12:00:00",
            "submission_state": "approved_ready",
            "source_result_row": "experiment_tracking/experiment_results.csv",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {OUT} with {len(rows)} tasks")


if __name__ == "__main__":
    main()


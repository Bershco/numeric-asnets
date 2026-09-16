#!/usr/bin/env python3
"""Materialize the immutable Slurm route for the final MPrime S2 searches."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiment_tracking" / "mprime_final_stage2_search_20260916"
MANIFEST = CAMPAIGN / "manifest.csv"
OUT = CAMPAIGN / "submissions.tsv"
ARRAY_JOB = "21388436"
SUBMITTED_AT = "2026-09-16T10:36:44+03:00"


with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 40 or len({row["manifest_id"] for row in rows}) != 40:
    raise RuntimeError("expected exactly forty unique MPrime search identities")

with OUT.open("w", newline="", encoding="utf-8") as stream:
    columns = [
        "array_index", "manifest_id", "search_method", "value_head", "seed",
        "selected_epoch", "source_training_job_id", "source_policy_job_id",
        "checkpoint", "checkpoint_sha256", "slurm_array_job_id",
        "slurm_task_job_id", "submitted_at",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        index = row["array_index"]
        writer.writerow({
            **{field: row[field] for field in columns[:10]},
            "slurm_array_job_id": ARRAY_JOB,
            "slurm_task_job_id": f"{ARRAY_JOB}_{index}",
            "submitted_at": SUBMITTED_AT,
        })
print(f"wrote {len(rows)} rows to {OUT}")

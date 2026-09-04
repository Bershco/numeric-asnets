#!/usr/bin/env python3
"""Build exact Rover retries for ten pre-inference semaphore failures."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiment_tracking" / "stage2_mcts_rover_completion_approved_20260904.csv"
LEDGER = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_rover_submissions_20260904.tsv"
OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_rover_node_retries_20260904.csv"
FAILED_JOB_IDS = {
    "20943986", "20943987", "20943989", "20943990", "20943991",
    "20943992", "20943993", "20943996", "20943997", "20943998",
}

with LEDGER.open(newline="", encoding="utf-8") as stream:
    by_manifest = {
        row["manifest_id"]: row["slurm_job_id"]
        for row in csv.DictReader(stream, delimiter="\t")
        if row["slurm_job_id"] in FAILED_JOB_IDS
    }
with MANIFEST.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    rows = [row for row in reader if row["manifest_id"] in by_manifest]

if len(by_manifest) != 10 or len(rows) != 10:
    raise RuntimeError(f"expected ten exact Rover failures; ledger={len(by_manifest)} rows={len(rows)}")

for row in rows:
    old_id = row["manifest_id"]
    row["manifest_id"] = old_id + "-node-retry1"
    row["excluded_nodes"] = "ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
    row["notes"] += (
        f"; exact retry of job {by_manifest[old_id]}, which failed before inference "
        "with node-local POSIX-semaphore ENOSPC on ise-cpu128-04"
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)

print(f"wrote {len(rows)} exact Rover retries to {OUTPUT}")

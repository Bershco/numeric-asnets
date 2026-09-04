#!/usr/bin/env python3
"""Build the missing non-endpoint TPP/off curve retries after node-local ENOSPC."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiment_tracking" / "four_domain_preservation"
READY = BASE / "terminal_stage2_tpp_off_policy_ready_20260903.csv"
RESULTS = ROOT / "experiment_tracking" / "preserve3_terminal_led" / "terminal_stage2_tpp_off_policy_results_20260904.csv"
ACTIVE_RETRIES = BASE / "terminal_stage2_tpp_off_policy_retry2_20260904.csv"
OUTPUT = BASE / "terminal_stage2_tpp_off_curve_node_retries_20260904.csv"

with RESULTS.open(newline="", encoding="utf-8") as stream:
    failed = {
        row["manifest_id"]: row["slurm_job_id"]
        for row in csv.DictReader(stream)
        if row["slurm_state"] == "FAILED" and not row["score"]
    }
with ACTIVE_RETRIES.open(newline="", encoding="utf-8") as stream:
    already_active = {
        row["manifest_id"].replace("-retry2", "")
        for row in csv.DictReader(stream)
    }
with READY.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    rows = [
        row for row in reader
        if row["manifest_id"] in failed and row["manifest_id"] not in already_active
    ]

if len(failed) != 39 or len(already_active) != 2 or len(rows) != 37:
    raise RuntimeError(
        f"expected failed=39 active=2 new=37; got {len(failed)}, {len(already_active)}, {len(rows)}"
    )
if "excluded_nodes" not in fields:
    fields.append("excluded_nodes")
if "notes" not in fields:
    fields.append("notes")

for row in rows:
    old_id = row["manifest_id"]
    row["manifest_id"] = old_id + "-node-retry1"
    row["excluded_nodes"] = "ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
    row["notes"] = (
        f"Exact curve-point retry of job {failed[old_id]}; failed before inference "
        "with node-local POSIX-semaphore ENOSPC. The two selected-endpoint gaps "
        "already have separate active retry2 jobs and are excluded here."
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} exact TPP/off curve retries to {OUTPUT}")

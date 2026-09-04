#!/usr/bin/env python3
"""Build exact TPP/on policy retries for node-local semaphore failures."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "experiment_tracking" / "preserve3_terminal_led" / "tpp_on_policy_ready_20260904.csv"
RESULTS = ROOT / "experiment_tracking" / "preserve3_terminal_led" / "tpp_on_policy_results_partial_20260904.csv"
OUTPUT = ROOT / "experiment_tracking" / "four_domain_preservation" / "terminal_stage2_tpp_on_policy_node_retries_20260904.csv"

with RESULTS.open(newline="", encoding="utf-8") as stream:
    failed = {
        row["manifest_id"]: row["slurm_job_id"]
        for row in csv.DictReader(stream)
        if row["slurm_state"] == "FAILED"
    }

with READY.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    rows = [row for row in reader if row["manifest_id"] in failed]

if len(rows) != len(failed) or len(rows) != 43:
    raise RuntimeError(f"expected 43 exact failed rows, found {len(rows)} for {len(failed)} failures")
if "excluded_nodes" not in fields:
    fields.append("excluded_nodes")
if "notes" not in fields:
    fields.append("notes")

for row in rows:
    old_id = row["manifest_id"]
    row["manifest_id"] = old_id + "-node-retry1"
    row["excluded_nodes"] = "ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
    row["notes"] = (
        f"Exact policy retry of job {failed[old_id]}; failed before inference on "
        "ise-cpu128-03 or ise-cpu128-04 with node-local POSIX-semaphore ENOSPC. "
        "Exclusion is limited to three independently demonstrated bad nodes."
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} exact TPP/on policy retries to {OUTPUT}")

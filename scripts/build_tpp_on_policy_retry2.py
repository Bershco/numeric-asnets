#!/usr/bin/env python3
"""Build four exact TPP/on second retries after exclusions were ignored."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiment_tracking" / "four_domain_preservation"
SOURCE = BASE / "terminal_stage2_tpp_on_policy_node_retries_20260904.csv"
RESULTS = ROOT / "experiment_tracking" / "preserve3_terminal_led" / "tpp_on_policy_node_retry_results_20260904.csv"
OUTPUT = BASE / "terminal_stage2_tpp_on_policy_node_retry2_20260904.csv"

with RESULTS.open(newline="", encoding="utf-8") as stream:
    failed = {
        row["manifest_id"]: row["slurm_job_id"]
        for row in csv.DictReader(stream)
        if row["slurm_state"] == "FAILED"
    }
with SOURCE.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    rows = [row for row in reader if row["manifest_id"] in failed]

if len(failed) != 4 or len(rows) != 4:
    raise RuntimeError(f"expected four failed retries; failed={len(failed)} rows={len(rows)}")
for row in rows:
    old_id = row["manifest_id"]
    row["manifest_id"] = old_id.removesuffix("-node-retry1") + "-node-retry2"
    row["excluded_nodes"] = "ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
    row["notes"] += (
        f" Second exact retry after job {failed[old_id]}; the nested cluster "
        "submission wrapper ignored its environment-only exclusion. The replacement "
        "uses explicit post-submission scontrol application and verification."
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
print(f"wrote {len(rows)} exact retry2 rows to {OUTPUT}")

#!/usr/bin/env python3
"""Create three FO Counters second retries after node-local ENOSPC."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_node_retry_20260904.csv"
OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_fo_node_retry2_20260904.csv"
TARGETS = {
    "stage2-gap-validation-fo_counters-off-923500475-w20_i70-node-retry1": "20943878",
    "stage2-gap-validation-fo_counters-off-1972442430-w20_i70-node-retry1": "20943882",
    "stage2-gap-validation-fo_counters-on-2011206605-w20_i70-node-retry1": "20943887",
}

with SOURCE.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    rows = [row for row in reader if row["manifest_id"] in TARGETS]

if len(rows) != len(TARGETS):
    raise RuntimeError(f"expected {len(TARGETS)} rows, found {len(rows)}")

for row in rows:
    old_id = row["manifest_id"]
    row["manifest_id"] = old_id.removesuffix("-node-retry1") + "-node-retry2"
    row["excluded_nodes"] = "ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
    row["notes"] += (
        f"; second exact retry after job {TARGETS[old_id]} also failed before inference "
        "with POSIX-semaphore ENOSPC on ise-cpu128-04"
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} exact FO retries to {OUTPUT}")

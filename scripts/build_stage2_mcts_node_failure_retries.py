#!/usr/bin/env python3
"""Create exact retries for Stage-2 MCTS jobs lost to node-local ENOSPC."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_approved_20260903.csv"
OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_node_retry_20260904.csv"
FAILED = {
    "stage2-gap-validation-block_grouping-off-2011206605-w5_i20": "20891028",
    "stage2-gap-validation-block_grouping-off-2082152039-w5_i20": "20891029",
    "stage2-gap-validation-block_grouping-on-1073581256-w5_i20": "20891031",
    "stage2-gap-validation-block_grouping-on-1472491096-w5_i20": "20891032",
    "stage2-gap-validation-fo_counters-off-923500475-w20_i70": "20891033",
    "stage2-gap-validation-fo_counters-off-1510771779-w20_i70": "20891034",
    "stage2-gap-validation-fo_counters-off-1972442430-w20_i70": "20891035",
    "stage2-gap-validation-fo_counters-off-2082152039-w20_i70": "20891036",
    "stage2-gap-validation-fo_counters-on-2011206605-w20_i70": "20891037",
}

with SOURCE.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
    fields = list(stream.seek(0) or csv.DictReader(stream).fieldnames or [])

# Re-read the header directly: DictReader.fieldnames is no longer conveniently
# available after the context manager above on every supported Python version.
with SOURCE.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])

selected = []
for row in rows:
    old_job = FAILED.get(row["manifest_id"])
    if not old_job:
        continue
    row = dict(row)
    row["manifest_id"] += "-node-retry1"
    row["notes"] += (
        f"; exact retry of job {old_job}, which failed before inference on "
        "ise-cpu128-03 with POSIX-semaphore ENOSPC"
    )
    row["excluded_nodes"] = "ise-cpu128-03,ise-cpu-intl-13"
    selected.append(row)

if len(selected) != len(FAILED):
    found = {row["manifest_id"].removesuffix("-node-retry1") for row in selected}
    raise RuntimeError(f"expected {len(FAILED)} retry rows, found {len(selected)}; missing={set(FAILED)-found}")
if "excluded_nodes" not in fields:
    fields.append("excluded_nodes")

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
print(f"wrote {len(selected)} exact node-failure retries to {OUTPUT}")

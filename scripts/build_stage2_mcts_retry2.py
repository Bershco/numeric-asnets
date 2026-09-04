#!/usr/bin/env python3
"""Create the one exact second retry required after another node-local ENOSPC."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_node_retry_20260904.csv"
OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_node_retry2_20260904.csv"
TARGET = "stage2-gap-validation-block_grouping-on-1073581256-w5_i20-node-retry1"

with SOURCE.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    matches = [row for row in reader if row["manifest_id"] == TARGET]

if len(matches) != 1:
    raise RuntimeError(f"expected one {TARGET!r} row, found {len(matches)}")

row = matches[0]
row["manifest_id"] = row["manifest_id"].removesuffix("-node-retry1") + "-node-retry2"
row["excluded_nodes"] = "ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
row["notes"] += (
    "; second exact retry after job 20943873 also failed before inference with "
    "POSIX-semaphore ENOSPC on ise-cpu128-04; exclusion remains node-specific"
)

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)

print(f"wrote exact retry to {OUTPUT}")

#!/usr/bin/env python3
"""Add Holm-adjusted p-values to complete Stage-2 policy/MCTS rows."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiment_tracking" / "stage2_policy_mcts_comparison_by_branch_20260902.csv"

with PATH.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
    fields = list(reader.fieldnames or [])

complete = [(index, float(row["raw_signflip_p"])) for index, row in enumerate(rows)
            if row.get("status") == "complete" and row.get("raw_signflip_p")]
ordered = sorted(complete, key=lambda item: item[1])
running_max = 0.0
adjusted: dict[int, float] = {}
count = len(ordered)
for rank, (index, raw) in enumerate(ordered, start=1):
    running_max = max(running_max, min(1.0, (count - rank + 1) * raw))
    adjusted[index] = running_max

if "holm_p" not in fields:
    insert_at = fields.index("raw_signflip_p") + 1
    fields.insert(insert_at, "holm_p")
for index, row in enumerate(rows):
    row["holm_p"] = f"{adjusted[index]:.9g}" if index in adjusted else ""

with PATH.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"updated {len(complete)} complete rows in {PATH}")

#!/usr/bin/env python3
"""Build held width-5/20 replacements for Counters MAIN-TERM Stage-2 MCTS."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "thesis_reproducibility_bundle" / "provenance"
    / "statistical_replication_ready_stage2_mcts.csv"
)
OUTPUT_DIR = ROOT / "experiment_tracking" / "mcts_counters_width_sensitivity"
OUTPUT = OUTPUT_DIR / "main_term_stage2_narrow_held_manifest.csv"
EMPTY_POLICY = OUTPUT_DIR / "empty_policy_manifest.csv"


with SOURCE.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames
    rows = [row for row in reader if row["domain"] == "counters"]

if fields is None or len(rows) != 20:
    raise ValueError(f"Expected twenty Counters Stage-2 rows, got {len(rows)}")
if len({(row["value_head"], row["seed"]) for row in rows}) != 20:
    raise ValueError("Expected ten unique seeds per Counters VH mode")

for row in rows:
    row["manifest_id"] += "-narrow-w5-i20"
    row["variant"] = "primary_narrow_w5_i20"
    row["width"] = "5"
    row["iterations"] = "20"
    row["status"] = "ready"
    row["notes"] = (
        "Held replacement for obsolete width-20/70 MAIN-TERM Stage-2 endpoint; "
        "aligned with confirmatory Counters Stage-1 width-5/20 search"
    )
    row["analysis_roles"] = "stage2_selected_mcts_counters_narrow_held"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

with EMPTY_POLICY.open("w", newline="", encoding="utf-8") as stream:
    csv.DictWriter(stream, fieldnames=fields, lineterminator="\n").writeheader()

print(f"{OUTPUT}={len(rows)}")

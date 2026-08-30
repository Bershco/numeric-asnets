#!/usr/bin/env python3
"""Build the twelve-row PW70 follow-up for prior 20-simulation cells."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiment_tracking/mcts_progressive_widening_cross_domain/pilot_manifest.csv"
OUT = ROOT / "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_manifest.csv"


def main() -> None:
    with SRC.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    selected = []
    for row in rows:
        if row["domain"] not in {"block_grouping", "counters"}:
            continue
        if row["iterations"] != "20":
            raise RuntimeError(f"unexpected source budget: {row['manifest_id']}")
        row = dict(row)
        row["experiment_id"] = "MCTS-PW70-CROSS-DOMAIN"
        row["manifest_id"] = row["manifest_id"].replace("pw-kmin3-", "pw70-kmin3-")
        row["iterations"] = "70"
        row["matched_fixed_width"] = "20"
        row["notes"] = (
            "Seventy-simulation follow-up to the separately retained 20-simulation "
            "budget-matched arm; report policy, fixed 5/20, fixed 20/70 when present, "
            "PW20, and PW70 distinctly"
        )
        selected.append(row)
    if len(selected) != 12:
        raise RuntimeError(f"expected 12 Block Grouping/Counters rows, got {len(selected)}")
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(selected)
    print(f"rows={len(selected)} output={OUT}")


if __name__ == "__main__":
    main()

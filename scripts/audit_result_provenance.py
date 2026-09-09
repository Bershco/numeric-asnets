#!/usr/bin/env python3
"""Inventory score-bearing CSVs and whether they expose source evidence.

This is deliberately structural: it does not reinterpret scientific results.
An authoritative aggregate may point to a named companion ledger through
result_provenance_index_20260902.csv instead of duplicating row-level log paths.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
INDEX = TRACKING / "result_provenance_index_20260902.csv"
OUT = TRACKING / "result_provenance_audit_20260902.csv"
LATEST = TRACKING / "result_provenance_audit_latest.csv"

SCORE_WORDS = ("score", "coverage", "success", "mcts_", "policy_mean", "auc")
PROVENANCE_WORDS = ("log", "path", "job_id", "slurm_job", "source", "ledger", "evidence")


def read_header(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return next(csv.reader(handle), [])
    except (OSError, UnicodeError, csv.Error):
        return []


companions: dict[str, str] = {}
if INDEX.exists():
    with INDEX.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            companions[row["authoritative_result_file"].replace("\\", "/")] = row[
                "row_level_log_columns_or_companion"
            ]

rows = []
for path in sorted(TRACKING.rglob("*.csv")):
    rel = path.relative_to(ROOT).as_posix()
    if path == OUT or path.name.startswith("result_provenance_audit"):
        continue
    header = read_header(path)
    lowered = [column.lower() for column in header]
    if not any(any(word in column for word in SCORE_WORDS) for column in lowered):
        continue
    direct = [
        column
        for column in header
        if any(word in column.lower() for word in PROVENANCE_WORDS)
    ]
    companion = companions.get(rel, "")
    rows.append(
        {
            "csv_file": rel,
            "score_columns": ";".join(
                column
                for column in header
                if any(word in column.lower() for word in SCORE_WORDS)
            ),
            "direct_provenance_columns": ";".join(direct),
            "companion_or_mapping": companion,
            "provenance_status": "direct" if direct else ("companion" if companion else "needs_mapping"),
        }
    )

for destination in (OUT, LATEST):
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print(f"wrote {len(rows)} score-bearing CSV rows to {OUT}")

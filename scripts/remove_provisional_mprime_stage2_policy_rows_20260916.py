#!/usr/bin/env python3
"""Remove MPrime S2 rows selected by the obsolete internal validator."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiment_tracking" / "policy_paired_seed_results.csv"
with PATH.open(newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))
fields = list(rows[0])
kept = [
    row for row in rows
    if not (row["experiment_id"] == "MAIN-VAL" and row["domain"] == "mprime")
]
if len(rows) - len(kept) not in {0, 20}:
    raise RuntimeError("unexpected number of provisional MPrime rows")
descriptor, temporary = tempfile.mkstemp(prefix=f".{PATH.name}.", suffix=".tmp", dir=PATH.parent)
with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(kept)
    stream.flush(); os.fsync(stream.fileno())
os.replace(temporary, PATH)
print(f"removed {len(rows)-len(kept)} provisional rows; canonical rows={len(kept)}")

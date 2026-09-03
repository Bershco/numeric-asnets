#!/usr/bin/env python3
"""Stream compact cutoff and VAL summaries for existing MCTS stdout logs."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


INSTANCE_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) "
    r"status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)"
)
TIMEOUT_RE = re.compile(r"\[EVAL INSTANCE\] timeout number=(\d+)")
VAL_RE = re.compile(r"\[VAL\] VAL-valid plans\s*:\s*(\d+)")


def main() -> None:
    rows = []
    for raw_path in sys.argv[1:]:
        path = Path(raw_path)
        text = path.read_text(errors="ignore")
        records = INSTANCE_RE.findall(text)
        successful = [record for record in records if float(record[4]) == 1.0]
        elapsed = [float(record[3]) for record in successful]
        val = VAL_RE.findall(text)
        rows.append({
            "job_id": path.name.split("_", 1)[0],
            "classified_instances": len(records),
            "successes_30m": sum(value <= 1800 for value in elapsed),
            "successes_2h": sum(value <= 7200 for value in elapsed),
            "successes_6h": sum(value <= 21600 for value in elapsed),
            "explicit_timeouts": len(set(TIMEOUT_RE.findall(text))),
            "max_completed_steps": max((int(record[5]) for record in records), default=0),
            "val_valid": int(val[-1]) if val else "",
            "source_log": str(path),
        })
    fields = list(rows[0]) if rows else ["job_id"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()

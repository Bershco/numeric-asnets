#!/usr/bin/env python3
"""Merge compatible evaluation CSV ledgers without duplicating job IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    fields: list[str] | None = None
    by_job: dict[str, dict[str, str]] = {}
    for path in args.inputs:
        current_fields, rows = read(path)
        if fields is None:
            fields = current_fields
        elif current_fields != fields:
            raise RuntimeError(f"incompatible CSV header: {path}")
        for row in rows:
            job_id = row.get("job_id", "")
            if not job_id:
                raise RuntimeError(f"missing job_id: {path}")
            previous = by_job.get(job_id)
            if previous is not None and previous != row:
                raise RuntimeError(f"conflicting duplicate job_id={job_id}")
            by_job[job_id] = row
    rows = sorted(by_job.values(), key=lambda row: (
        row["domain"], row["value_head"], int(row["seed"]),
        int(row["snapshot_epoch"]), int(row["job_id"])))
    if fields is None:
        raise RuntimeError("no inputs")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze only scientifically missing Counters rich-trace identities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def is_complete(row: dict[str, str]) -> tuple[bool, str]:
    output = Path(row["remote_output"])
    ledgers = sorted(output.glob("*.completed.jsonl"))
    for ledger in ledgers:
        try:
            records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        if len(records) != 1 or int(records[0].get("instance_number", -1)) != int(row["instance_number"]):
            continue
        stem = ledger.name.removesuffix(".completed.jsonl")
        if (output / f"{stem}_steps.csv").is_file():
            return True, str(ledger)
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    args = parser.parse_args()

    with args.source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["array_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("source manifest indices are not contiguous")

    recovery: list[dict[str, str | int]] = []
    retained: list[dict[str, str | int]] = []
    for row in rows:
        complete, ledger = is_complete(row)
        if complete:
            retained.append({
                "source_array_index": int(row["array_index"]),
                "tie_break": row["tie_break"],
                "seed": row["seed"],
                "instance_number": int(row["instance_number"]),
                "ledger": ledger,
            })
            continue
        updated: dict[str, str | int] = dict(row)
        updated["source_trace_index"] = int(row["array_index"])
        updated["array_index"] = len(recovery)
        updated["remote_output"] = str(
            args.recovery_root
            / row["tie_break"]
            / row["seed"]
            / f"instance_{row['instance_number']}"
        )
        recovery.append(updated)

    identities = [
        (row["tie_break"], row["seed"], int(row["instance_number"]))
        for row in recovery
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("recovery manifest contains duplicate identities")
    if len(recovery) + len(retained) != len(rows):
        raise RuntimeError("reconciliation lost source identities")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if recovery:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(recovery[0]))
            writer.writeheader()
            writer.writerows(recovery)
    summary = {
        "source_tasks": len(rows),
        "retained_complete": len(retained),
        "recovery_tasks": len(recovery),
        "source_manifest": str(args.source),
        "recovery_manifest": str(args.output),
        "recovery_root": str(args.recovery_root),
        "retained": retained,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("source_tasks", "retained_complete", "recovery_tasks")}, sort_keys=True))


if __name__ == "__main__":
    main()

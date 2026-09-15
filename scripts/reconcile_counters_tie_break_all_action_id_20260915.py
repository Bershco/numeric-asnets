#!/usr/bin/env python3
"""Reconcile and require all ten Counters action-ID terminal ledgers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reconcile_counters_tie_break_terminal_records_20260915 import reconcile


SOURCE_ARRAY_JOB_ID = "21233925"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["tie_break"] == "action_id"]
    if len(rows) != 10:
        raise RuntimeError(f"expected ten action-ID rows, found {len(rows)}")
    summaries = []
    for row in rows:
        task = int(row["array_index"])
        output = Path(row["remote_output"])
        ledger = output / f"{SOURCE_ARRAY_JOB_ID}_{task}.completed.jsonl"
        logs = sorted(output.glob("*.txt"))
        if not ledger.is_file() or not logs:
            raise RuntimeError(f"missing task {task} ledger/log evidence")
        summary = reconcile(ledger, logs)
        if summary["terminal_after"] != 59:
            raise RuntimeError(
                f"task {task} remains incomplete after reconciliation: "
                f"{summary['terminal_after']}/59"
            )
        summary["task"] = task
        summary["seed"] = row["seed"]
        summaries.append(summary)
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()

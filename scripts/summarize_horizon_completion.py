#!/usr/bin/env python3
"""Summarize resumable Horizon evaluation ledgers without reading stdout."""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("output", type=Path)
    parser.add_argument("--live-mcts", type=Path)
    parser.add_argument("--source-ledger-root")
    args = parser.parse_args()

    log_by_job: dict[str, str] = {}
    if args.live_mcts:
        with args.live_mcts.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                log_path = row.get("source_evaluation_log", "")
                log_by_job[str(row.get("job_id", ""))] = log_path.replace(
                    "%x", row.get("job_name", ""))

    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    maxima: dict[tuple[str, str], int] = {}
    rows: list[dict[str, object]] = []
    for name in sorted(glob.glob(args.pattern)):
        job_id = Path(name).stem
        with open(name, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                hit_goal = bool(record.get("hit_goal", False))
                key = (str(record.get("status", "")), str(hit_goal))
                steps = int(record.get("steps") or 0)
                counts[key] += 1
                maxima[key] = max(maxima.get(key, 0), steps)
                rows.append({
                    "job_id": job_id,
                    "instance_number": record.get("instance_number", ""),
                    "instance_path": record.get("instance_path", ""),
                    "status": key[0],
                    "hit_goal": key[1],
                    "steps": steps,
                    "elapsed_seconds": record.get("elapsed_seconds", ""),
                    "evaluation_signature": record.get("evaluation_signature", ""),
                    "source_evaluation_log": log_by_job.get(job_id, ""),
                    "source_completion_ledger": (
                        str(PurePosixPath(args.source_ledger_root) / Path(name).name)
                        if args.source_ledger_root else name
                    ),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_id", "instance_number", "instance_path", "status", "hit_goal",
        "steps", "elapsed_seconds", "evaluation_signature",
        "source_evaluation_log", "source_completion_ledger",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"records={len(rows)}")
    for key in sorted(counts):
        print(f"status={key[0]} success={key[1]} count={counts[key]} max_steps={maxima[key]}")


if __name__ == "__main__":
    main()

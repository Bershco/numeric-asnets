#!/usr/bin/env python3
"""Convert the canonical pipe-delimited squeue snapshot into live_jobs.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def experiment(job: str) -> str:
    rules = (
        ("Re-Tr_mprime", "MPRIME-ANCHOR"),
        ("Re-Tr_delivery", "PRESERVE-DELIVERY-S2"),
        ("Re-Tr_tpp", "PRESERVE-TPP-S2"),
        ("Ev_mprime", "MPRIME-ANCHOR-POLICY"),
        ("Ev_delivery", "PRESERVE-DELIVERY-S2-POLICY"),
        ("context-full", "MCTS-SAFE-CONTEXT"),
        ("horizon-", "MCTS-HORIZON"),
        ("pw-kmin3", "MCTS-PW-CROSS-DOMAIN"),
        ("HORIZON_POSTHOC_VAL", "MCTS-HORIZON-VAL"),
    )
    for prefix, name in rules:
        if job.startswith(prefix):
            return name
    if job.startswith("Ev_counters") and "SR10M" in job:
        return "MCTS-WIDTH-COUNTERS-S2"
    if job.startswith("Ev_drone") and "SR10M" in job:
        return "MAIN-VAL-S2-MCTS"
    if job.startswith("Ev_rover"):
        return "MCTS-RESOURCE-ROVER"
    if job.startswith("Ev_fo_counters"):
        return "MCTS-RESOURCE-FO-HELD"
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-time", default=datetime.now().astimezone().isoformat(timespec="seconds"))
    args = parser.parse_args()
    rows = []
    for raw in sys.stdin:
        raw = raw.rstrip("\r\n")
        if not raw:
            continue
        job_id, job, state, reason, elapsed, limit, cpus, memory = raw.split("|", 7)
        rows.append({
            "snapshot_time": args.snapshot_time,
            "experiment_id": experiment(job),
            "job_id": job_id,
            "state": state,
            "reason": reason,
            "elapsed": elapsed,
            "time_limit": limit,
            "cpus": cpus,
            "memory": memory,
            "job_name": job,
        })
    fields = [
        "snapshot_time", "experiment_id", "job_id", "state", "reason",
        "elapsed", "time_limit", "cpus", "memory", "job_name",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} live jobs to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize conservative MCTS coverage at alternate per-instance cutoffs.

The input is one or more reconciliation CSVs containing ``source_log`` and a
Slurm job identifier.  Completion records are deduplicated by stable instance
number.  A successful instance counts at a cutoff only when its recorded
elapsed time is no greater than that cutoff.  Unsolved, timed-out, crashed, and
unclassified instances remain failures.

This is a deterministic post-hoc counterfactual over recorded executions.  It
does not model the extra cluster throughput that a genuinely shorter timeout
would create.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RECORD_RE = re.compile(
    r"\[EVAL INSTANCE\] (?:completed|skip completed) "
    r"number=(?P<number>\d+) path=(?P<path>\S+) "
    r"status=(?P<status>\S+) elapsed=(?P<elapsed>[0-9.]+)s "
    r"success=(?P<success>\S+) steps=(?P<steps>-?\d+)"
)
TIMEOUT_RE = re.compile(
    r"\[EVAL INSTANCE\] timeout number=(?P<number>\d+) "
    r"path=(?P<path>\S+) limit=(?P<elapsed>[0-9.]+)s"
)


def truthy(value: str) -> bool:
    return value.lower() in {"1", "1.0", "true"}


def normalize_status(value: str) -> str:
    return {
        "finished_unsolved": "unsolved",
        "finished_success": "success",
    }.get(value, value)


def parse_log(path: Path) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = RECORD_RE.search(line)
            if match:
                number = int(match.group("number"))
                record = {
                    "instance_number": number,
                    "instance_path": match.group("path"),
                    "status": normalize_status(match.group("status")),
                    "elapsed_seconds": float(match.group("elapsed")),
                    "success": truthy(match.group("success")),
                    "steps": int(match.group("steps")),
                }
            else:
                timeout = TIMEOUT_RE.search(line)
                if not timeout:
                    continue
                number = int(timeout.group("number"))
                record = {
                    "instance_number": number,
                    "instance_path": timeout.group("path"),
                    "status": "timeout",
                    "elapsed_seconds": float(timeout.group("elapsed")),
                    "success": False,
                    "steps": -1,
                }
            previous = records.get(number)
            if previous is not None and previous != record:
                # A rolling evaluation may time an instance out, restart its
                # worker, and later emit the ordinary terminal record.  The
                # terminal record is the authoritative outcome and elapsed
                # time; a later replayed timeout must not overwrite it.
                previous_timeout = previous["status"] == "timeout"
                current_timeout = record["status"] == "timeout"
                if previous_timeout and not current_timeout:
                    records[number] = record
                    continue
                if not previous_timeout and current_timeout:
                    continue
                raise RuntimeError(
                    f"conflicting terminal records in {path} for instance "
                    f"{number}: {previous!r} versus {record!r}"
                )
            records[number] = record
    return records


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--jobs-out", required=True, type=Path)
    parser.add_argument("--instances-out", required=True, type=Path)
    parser.add_argument(
        "--cutoffs", default="1800,7200,21600",
        help="comma-separated per-instance cutoffs in seconds",
    )
    parser.add_argument(
        "--include-active", action="store_true",
        help="also summarize running jobs as explicitly provisional lower bounds",
    )
    args = parser.parse_args()
    cutoffs = [int(value) for value in args.cutoffs.split(",")]

    job_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    seen_jobs: set[str] = set()
    for input_path in args.inputs:
        with input_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                job_id = first(row, "slurm_job_id", "mcts_job_id", "job_id")
                source_log = first(row, "source_log", "mcts_log", "source_evaluation_log")
                state = first(row, "slurm_state", "mcts_state", "state")
                if not args.include_active and state.upper() in {
                    "RUNNING", "PENDING", "REQUEUED", "COMPLETING", "CONFIGURING"
                }:
                    continue
                if not job_id or not source_log or job_id in seen_jobs:
                    continue
                path = Path(source_log)
                if not path.is_file():
                    raise FileNotFoundError(path)
                records = parse_log(path)
                seen_jobs.add(job_id)
                metadata = {
                    "experiment_id": row.get("experiment_id", ""),
                    "manifest_id": row.get("manifest_id", ""),
                    "domain": row.get("domain", ""),
                    "stage": row.get("stage", ""),
                    "value_head": row.get("value_head", ""),
                    "seed": row.get("seed", ""),
                    "arm": row.get("arm", ""),
                    "iterations": row.get("iterations", ""),
                    "job_id": job_id,
                    "slurm_state": state,
                    "source_log": source_log,
                }
                known_total = first(row, "total", "total_instances")
                job_record: dict[str, object] = {
                    **metadata,
                    "recorded_instances": len(records),
                    "total_instances": known_total,
                    "unclassified_instances": (
                        int(float(known_total)) - len(records) if known_total else ""
                    ),
                    "recorded_successes": sum(bool(r["success"]) for r in records.values()),
                    "recorded_unsolved": sum(
                        r["status"] == "unsolved" for r in records.values()
                    ),
                    "recorded_timeouts": sum(
                        r["status"] == "timeout" for r in records.values()
                    ),
                }
                for cutoff in cutoffs:
                    job_record[f"successes_le_{cutoff}s"] = sum(
                        bool(record["success"])
                        and float(record["elapsed_seconds"]) <= cutoff
                        for record in records.values()
                    )
                job_rows.append(job_record)
                for number in sorted(records):
                    record = records[number]
                    instance_record = {**metadata, **record}
                    for cutoff in cutoffs:
                        instance_record[f"success_le_{cutoff}s"] = int(
                            bool(record["success"])
                            and float(record["elapsed_seconds"]) <= cutoff
                        )
                    instance_rows.append(instance_record)

    args.jobs_out.parent.mkdir(parents=True, exist_ok=True)
    args.instances_out.parent.mkdir(parents=True, exist_ok=True)
    job_fields = list(job_rows[0]) if job_rows else []
    instance_fields = list(instance_rows[0]) if instance_rows else []
    with args.jobs_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=job_fields)
        writer.writeheader()
        writer.writerows(job_rows)
    with args.instances_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=instance_fields)
        writer.writeheader()
        writer.writerows(instance_rows)
    print(
        f"jobs={len(job_rows)} instances={len(instance_rows)} "
        f"jobs_out={args.jobs_out} instances_out={args.instances_out}"
    )


if __name__ == "__main__":
    main()

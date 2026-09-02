#!/usr/bin/env python3
"""Write compact running-MCTS lower bounds with direct log provenance."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


INSTANCE_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) "
    r"status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)"
)
TIMEOUT_RE = re.compile(r"\[EVAL INSTANCE\] timeout number=(\d+)")
DEPTH_RE = re.compile(r"horizon_cutoffs=\((count=\d+[^)]*)\)")


def classify(name: str) -> str:
    upper = name.upper()
    if "PW-COUNTERS-DIVERGENCE" in upper:
        return "MCTS-PW-COUNTERS-DIVERGENCE"
    if "PW70-CONFIRM" in upper:
        return "MCTS-PW70-CONFIRMATORY"
    if "PW70-KMIN3" in upper:
        return "MCTS-PW70-CROSS-DOMAIN"
    if "HORIZON-COUNTERS" in upper:
        return "MCTS-HORIZON-COUNTERS"
    if "SR10TCM" in upper:
        return "MCTS-LEGACY-FO"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--user", default="hersco")
    args = parser.parse_args()
    queue = subprocess.check_output(
        ["squeue", "-h", "-u", args.user,
         "-o", "%i|%T|%j|%M|%l|%C|%m|%R"], text=True,
    )
    jobs = []
    for line in queue.splitlines():
        fields = line.split("|", 7)
        experiment = classify(fields[2]) if len(fields) == 8 else ""
        if experiment and fields[1] == "RUNNING":
            jobs.append((experiment, fields))
    ids = [fields[0] for _, fields in jobs]
    stdout = {}
    if ids:
        account = subprocess.check_output(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
             "-o", "JobIDRaw,StdOut%1000"], text=True,
        )
        for line in account.splitlines():
            job_id, _, path = line.partition("|")
            if job_id in ids:
                stdout[job_id] = path.replace("%j", job_id)
    rows = []
    for experiment, fields in jobs:
        job_id, state, name, elapsed, limit, cpus, memory, reason = fields
        path = Path(stdout.get(job_id, ""))
        text = path.read_text(errors="ignore") if path.is_file() else ""
        instances = INSTANCE_RE.findall(text)
        if experiment == "MCTS-HORIZON-COUNTERS":
            completion = Path(
                "/home/hersco/training_new_domains/2026-08-31/"
                f"mcts_horizon_counters/completion/{job_id}.jsonl"
            )
        elif experiment.startswith("MCTS-PW"):
            completion = Path(
                "/home/hersco/training_new_domains/2026-08-30/"
                f"mcts_pw_cross_domain/completion/{job_id}.jsonl"
            )
        else:
            completion = Path()
        ledger_records = []
        if str(completion) != "." and completion.is_file():
            for line in completion.read_text(errors="ignore").splitlines():
                try:
                    ledger_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        successful = [entry for entry in instances if float(entry[4]) == 1.0]
        if ledger_records:
            classified = len({int(record["instance_number"]) for record in ledger_records})
            successful_elapsed = [
                float(record["elapsed_seconds"])
                for record in ledger_records if record.get("hit_goal")
            ]
            max_steps = max((int(record.get("steps", 0)) for record in ledger_records), default=0)
        else:
            classified = len(instances)
            successful_elapsed = [float(entry[3]) for entry in successful]
            max_steps = max((int(entry[5]) for entry in instances), default=0)
        horizons = DEPTH_RE.findall(text)
        rows.append({
            "experiment_id": experiment, "job_id": job_id,
            "job_name": name, "state": state, "elapsed": elapsed,
            "time_limit": limit, "cpus": cpus, "memory": memory,
            "classified_instances": classified,
            "successes_30m_lower_bound": sum(value <= 1800 for value in successful_elapsed),
            "successes_2h_lower_bound": sum(value <= 7200 for value in successful_elapsed),
            "successes_6h_lower_bound": sum(value <= 21600 for value in successful_elapsed),
            "explicit_instance_timeouts": len(set(TIMEOUT_RE.findall(text))),
            "max_completed_steps": max_steps,
            "last_horizon_cutoff_summary": horizons[-1] if horizons else "",
            "source_evaluation_log": str(path),
            "source_completion_ledger": str(completion) if str(completion) != "." else "",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["experiment_id"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} running MCTS rows to {args.output}")


if __name__ == "__main__":
    main()

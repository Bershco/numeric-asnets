#!/usr/bin/env python3
"""Summarize Slurm MCTS evaluation jobs from compact completion records.

The script is intentionally read-only.  It joins Slurm accounting, stdout-log
paths, and per-instance JSONL completion records, then emits one CSV row per
job.  It never reads or copies the often very large plan payloads beyond JSON
decoding the record that contains them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")


def accounting(job_ids: list[str]) -> dict[str, dict[str, str]]:
    fields = [
        "JobIDRaw", "JobName", "State", "Elapsed", "Start", "End",
        "NodeList", "ReqCPUS", "ReqMem",
    ]
    output = subprocess.check_output(
        [
            "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
            "--format=" + ",".join(fields),
        ],
        text=True,
    )
    rows: dict[str, dict[str, str]] = {}
    for values in csv.reader(output.splitlines(), delimiter="|"):
        if len(values) < len(fields) or values[0] not in job_ids:
            continue
        rows[values[0]] = dict(zip(fields, values))
    return rows


def first_path(patterns: list[str]) -> str:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    return sorted(matches)[0] if matches else ""


def completion_summary(path: str) -> dict[str, int]:
    by_instance: dict[int, dict[str, object]] = {}
    if path:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                number = record.get("instance_number")
                if isinstance(number, int):
                    by_instance[number] = record
    successes = [
        row for row in by_instance.values()
        if row.get("status") == "success" and row.get("hit_goal") is True
    ]
    def at_most(seconds: float) -> int:
        return sum(float(row.get("elapsed_seconds", float("inf"))) <= seconds
                   for row in successes)
    return {
        "classified_instances": len(by_instance),
        "success_30m": at_most(1800),
        "success_2h": at_most(7200),
        "success_6h": at_most(21600),
        "success_full_record": len(successes),
        "ordinary_unsolved": sum(
            row.get("status") == "finished_unsolved"
            for row in by_instance.values()
        ),
    }


def log_summary(path: str) -> dict[str, str | int]:
    result: dict[str, str | int] = {
        "latest_eval_final_success": "",
        "latest_eval_final_total": "",
        "logged_timeouts": 0,
        "val_valid": "",
        "val_invalid": "",
    }
    if not path:
        return result
    final_re = re.compile(r"\[EVAL FINAL\].*?success=(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
    val_re = re.compile(r"valid(?:_plans)?\s*[=:]\s*(\d+).*?invalid(?:_plans)?\s*[=:]\s*(\d+)", re.I)
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = final_re.search(line)
            if match:
                result["latest_eval_final_success"] = match.group(1)
                result["latest_eval_final_total"] = match.group(2)
            if "[EVAL INSTANCE] timeout" in line:
                result["logged_timeouts"] = int(result["logged_timeouts"]) + 1
            match = val_re.search(line)
            if match:
                result["val_valid"] = match.group(1)
                result["val_invalid"] = match.group(2)
            for key, pattern in (("val_valid", r"VAL-valid plans\s*:\s*(\d+)"),
                                 ("val_invalid", r"VAL-invalid plans\s*:\s*(\d+)")):
                match = re.search(pattern, line)
                if match:
                    result[key] = match.group(1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_ids", nargs="+")
    args = parser.parse_args()
    job_ids = [str(int(job_id)) for job_id in args.job_ids]
    acct = accounting(job_ids)
    fieldnames = [
        "job_id", "job_name", "state", "elapsed", "start", "end", "node",
        "requested_cpus", "requested_memory", "classified_instances",
        "success_30m", "success_2h", "success_6h", "success_full_record",
        "ordinary_unsolved", "logged_timeouts", "latest_eval_final_success",
        "latest_eval_final_total", "val_valid", "val_invalid", "log_path",
        "completion_record_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for job_id in job_ids:
        row = acct.get(job_id, {})
        log_path = first_path([str(ROOT / "*" / "*" / f"{job_id}_*.txt")])
        completion_path = first_path([
            str(ROOT / "*" / "*" / ".resume_state" / f"{job_id}.eval_completed.jsonl"),
            str(ROOT / "*" / "*" / "completion" / f"{job_id}.jsonl"),
        ])
        writer.writerow({
            "job_id": job_id,
            "job_name": row.get("JobName", ""),
            "state": row.get("State", ""),
            "elapsed": row.get("Elapsed", ""),
            "start": row.get("Start", ""),
            "end": row.get("End", ""),
            "node": row.get("NodeList", ""),
            "requested_cpus": row.get("ReqCPUS", ""),
            "requested_memory": row.get("ReqMem", ""),
            **completion_summary(completion_path),
            **log_summary(log_path),
            "log_path": log_path,
            "completion_record_path": completion_path,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write compact live training progress with direct Slurm-log provenance."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
EPOCH_RE = re.compile(r"epoch:\s*.*?\b(\d+)/(\d+)\b")
CHECKPOINT_RE = re.compile(r"Last valid checkpoint is (\S+)")
VALIDATION_RE = re.compile(r"Current network validation success rate:\s*([0-9.]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--user", default="hersco")
    args = parser.parse_args()

    queue = subprocess.check_output(
        ["squeue", "-h", "-u", args.user,
         "-o", "%i|%T|%j|%M|%l|%C|%m|%R"],
        text=True,
    )
    jobs = []
    for line in queue.splitlines():
        fields = line.split("|", 7)
        if len(fields) != 8 or not fields[2].startswith("Re-Tr_"):
            continue
        jobs.append(fields)

    ids = [fields[0] for fields in jobs]
    stdout = {}
    if ids:
        acct = subprocess.check_output(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
             "-o", "JobIDRaw,StdOut%1000"],
            text=True,
        )
        for line in acct.splitlines():
            job_id, _, path = line.partition("|")
            if job_id in ids:
                stdout[job_id] = path.replace("%j", job_id)

    rows = []
    for job_id, state, name, elapsed, limit, cpus, memory, reason in jobs:
        path = Path(stdout.get(job_id, ""))
        text = ""
        if path.is_file():
            text = ANSI_RE.sub("", path.read_text(errors="ignore"))
        epochs = EPOCH_RE.findall(text)
        checkpoints = CHECKPOINT_RE.findall(text)
        validations = VALIDATION_RE.findall(text)
        rows.append({
            "job_id": job_id, "state": state, "job_name": name,
            "elapsed": elapsed, "time_limit": limit, "cpus": cpus,
            "memory": memory, "reason_or_node": reason,
            "latest_epoch": epochs[-1][0] if epochs else "",
            "epoch_limit": epochs[-1][1] if epochs else "",
            "latest_validation_coverage": validations[-1] if validations else "",
            "last_valid_checkpoint": checkpoints[-1] if checkpoints else "",
            "source_training_log": str(path),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["job_id"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} training rows to {args.output}")


if __name__ == "__main__":
    main()

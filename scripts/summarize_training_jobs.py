#!/usr/bin/env python3
"""Summarize exact Slurm training jobs with their final logged epoch marker."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
EPOCH_RE = re.compile(r"epoch:\s*.*?\b(\d+)/(\d+)\b")
VAL_RE = re.compile(r"\[VALIDATION\] Current network validation success rate: ([0-9.]+)")
FIELDS = ["job_id", "state", "elapsed", "time_limit", "epoch", "max_epoch", "latest_validation", "source_log"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", args.jobs,
         "-o", "JobIDRaw,State,Elapsed,Timelimit,StdOut%1000"], text=True
    )
    rows = []
    for line in text.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        job, state, elapsed, limit, stdout = parts
        log = Path(stdout.replace("%j", job))
        epoch = max_epoch = validation = ""
        if log.is_file():
            clean = ANSI_RE.sub("", log.read_text(encoding="utf-8", errors="replace"))
            epochs = EPOCH_RE.findall(clean)
            vals = VAL_RE.findall(clean)
            if epochs: epoch, max_epoch = epochs[-1]
            if vals: validation = vals[-1]
        rows.append({"job_id": job, "state": state.split()[0], "elapsed": elapsed,
                     "time_limit": limit, "epoch": epoch, "max_epoch": max_epoch,
                     "latest_validation": validation, "source_log": str(log)})
    stream = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    finally:
        if args.output: stream.close()


if __name__ == "__main__":
    main()

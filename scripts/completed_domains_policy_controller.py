#!/usr/bin/env python3
"""Idempotently submit ready four-domain policy-evaluation rows only."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")
SUBMITTER = ROOT / "submit_training.sh"
DOMAINS = {"delivery", "mprime", "tpp", "zenotravel"}
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def validate(rows: list[dict[str, str]]) -> None:
    ids = [row["manifest_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate manifest IDs")
    for row in rows:
        if row["status"] != "ready" or row["task_type"] != "policy_eval" or row["stage"] != "stage1":
            raise ValueError(f"Invalid policy row {row['manifest_id']}")
        if row["domain"] not in DOMAINS or row["workers"] != "10" or row["memory"] != "20G" or row["time_limit"] != "04:00:00":
            raise ValueError(f"Invalid configuration {row['manifest_id']}")
        if not Path(row["source_checkpoint_ref"]).is_dir():
            raise ValueError(f"Missing checkpoint {row['source_checkpoint_ref']}")


def command(row: dict[str, str], dry: bool) -> tuple[list[str], dict[str, str]]:
    cmd = [
        str(SUBMITTER), f"--dom-{row['domain']}", "--original-only",
        "--domain-architecture", "policy", "--seed", row["seed"],
        "--workers", "10", "--jpddl-max-heap", "4g", "--time", "04:00:00",
        "--mem", "20G", "--cpus", "6", "--eval-from", row["source_checkpoint_ref"],
        "--job-suffix", f"CD4P_src{row['source_training_job_id']}_e{int(row['snapshot_epoch']):04d}",
        "--output-subdir", "completed_domains_stage1_policy_eval",
    ]
    if row["value_head"] == "off":
        cmd.append("--vh-off")
    if dry:
        cmd.append("--dry-run")
    env = os.environ.copy()
    env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
    return cmd, env


def submit(row: dict[str, str], dry: bool) -> str:
    cmd, env = command(row, dry)
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    if dry:
        if "Expected jobs: 1" not in result.stdout or "[DRY]" not in result.stdout:
            raise RuntimeError(result.stdout)
        return "DRY"
    matches = JOB_RE.findall(result.stdout)
    if len(matches) != 1:
        raise RuntimeError(result.stdout)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue-cap", type=int, default=1999)
    parser.add_argument("--max-per-cycle", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    rows = read(args.manifest)
    validate(rows)
    existing = {row["manifest_id"] for row in read(args.ledger, "\t")} if args.ledger.exists() else set()
    representatives = {}
    for row in rows:
        representatives.setdefault((row["domain"], row["value_head"]), row)
    for row in representatives.values():
        submit(row, True)
    print(f"[VALID] rows={len(rows)} dry_runs={len(representatives)}", flush=True)
    if args.validate_only:
        return
    fields = ["manifest_id", "task_type", "domain", "value_head", "seed", "source_training_job_id", "snapshot_epoch", "slurm_job_id", "submitted_at", "source_checkpoint"]
    while True:
        remaining = [row for row in rows if row["manifest_id"] not in existing]
        if not remaining:
            print(f"[COMPLETE] submitted={len(existing)}", flush=True)
            return
        queued = len(subprocess.check_output(["squeue", "-u", "hersco", "-h"], text=True).splitlines())
        allowance = min(max(0, args.queue_cap - queued), args.max_per_cycle, len(remaining))
        print(f"[CONTROLLER] queue={queued} remaining={len(remaining)} allowance={allowance}", flush=True)
        for row in remaining[:allowance]:
            job_id = submit(row, False)
            new = not args.ledger.exists()
            with args.ledger.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
                if new:
                    writer.writeheader()
                writer.writerow({
                    "manifest_id": row["manifest_id"], "task_type": row["task_type"], "domain": row["domain"],
                    "value_head": row["value_head"], "seed": row["seed"], "source_training_job_id": row["source_training_job_id"],
                    "snapshot_epoch": row["snapshot_epoch"], "slurm_job_id": job_id,
                    "submitted_at": datetime.now(timezone.utc).isoformat(), "source_checkpoint": row["source_checkpoint_ref"],
                })
                stream.flush(); os.fsync(stream.fileno())
            existing.add(row["manifest_id"])
            print(f"[SUBMITTED] {job_id} {row['manifest_id']}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()

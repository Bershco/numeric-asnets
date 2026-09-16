#!/usr/bin/env python3
"""Submit the 17 persistent native-failure MPrime curve points with one worker."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")
FIELDS = [
    "manifest_id", "slurm_job_id", "submitted_at", "source_training_job_id",
    "snapshot_epoch", "source_checkpoint", "workers", "cpus", "memory", "walltime",
]


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--submitter", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.manifest)
    if len(rows) != 17 or len({row["manifest_id"] for row in rows}) != 17:
        raise RuntimeError("recovery manifest must contain 17 unique rows")
    existing = {
        row["manifest_id"] for row in read(args.ledger, "\t")
    } if args.ledger.exists() else set()

    for row in rows:
        if row["manifest_id"] in existing:
            continue
        suffix = (
            f"MPFINALV_S2POL_1W_src{row['source_training_job_id']}_"
            f"e{int(row['snapshot_epoch']):04d}"
        )
        command = [
            str(args.submitter), "--dom-mprime", "--original-only",
            "--domain-architecture", "mcts", "--seed", row["seed"],
            "--workers", "1", "--jpddl-max-heap", "4g",
            "--time", "08:00:00", "--mem", "20G", "--cpus", "2",
            "--eval-from", row["source_checkpoint_ref"],
            "--job-suffix", suffix,
            "--output-subdir", "mprime_final_validation_stage2_policy_1w_mprime",
        ]
        if row["value_head"] == "off":
            command.append("--vh-off")
        env = os.environ.copy()
        env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
        result = subprocess.run(
            command, cwd="/home/hersco/training_new_domains", env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(result.stdout)
        matches = JOB_RE.findall(result.stdout)
        if len(matches) != 1:
            raise RuntimeError(result.stdout)
        new = not args.ledger.exists()
        with args.ledger.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow({
                "manifest_id": row["manifest_id"], "slurm_job_id": matches[0],
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "source_training_job_id": row["source_training_job_id"],
                "snapshot_epoch": row["snapshot_epoch"],
                "source_checkpoint": row["source_checkpoint_ref"],
                "workers": "1", "cpus": "2", "memory": "20G", "walltime": "08:00:00",
            })
        print(f"[SUBMITTED] {matches[0]} {row['manifest_id']}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retry the two native-import startup failures after the compute smoke passes."""

from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains/2026-09-08/anchor_kl_control")
MANIFEST = ROOT / "anchor_kl_control_tpp_screen_20260908.csv"
WRAPPER = ROOT / "anchor_kl_adaptive_tpp.sbatch"
LEDGER = ROOT / "anchor_kl_control_retry_submissions_20260909.tsv"
FAILED = {"1972442430": "21144210", "1963100312": "21144211"}


def failed_state(job_id: str) -> str:
    output = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "State"],
        text=True,
    )
    return output.strip().splitlines()[0].split("|", 1)[0]


def main() -> None:
    if LEDGER.exists():
        raise SystemExit(f"refusing duplicate retry: {LEDGER} already exists")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if {row["seed"] for row in rows} != set(FAILED):
        raise SystemExit("retry manifest does not contain the exact failed seed set")
    if any(failed_state(job_id) != "FAILED" for job_id in FAILED.values()):
        raise SystemExit("one or more source jobs are not in FAILED state")

    submitted = []
    for row in rows:
        checkpoint = Path(row["source_checkpoint"])
        if not checkpoint.exists():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        env = os.environ.copy()
        env.update(
            ANCHOR_SOURCE_CHECKPOINT=str(checkpoint),
            ANCHOR_SEED=row["seed"],
            ANCHOR_TEACHER=row["teacher"],
            ANCHOR_TARGET=row["target_kl"],
        )
        job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                f"--job-name={row['manifest_id']}-retry1",
                f"--output={row['output_log'].replace('.txt', '-retry1.txt')}",
                "--export=ALL",
                str(WRAPPER),
            ],
            text=True,
            env=env,
        ).strip().split(";", 1)[0]
        submitted.append(
            {
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_id": row["manifest_id"],
                "job_id": job_id,
                "retry_of": FAILED[row["seed"]],
                "seed": row["seed"],
                "role": row["role"],
                "source_training_job_id": row["source_training_job_id"],
                "source_epoch": row["source_epoch"],
                "source_checkpoint": str(checkpoint),
                "teacher": row["teacher"],
                "target_kl": row["target_kl"],
                "cpus": row["cpus"],
                "memory_gib": row["memory_gib"],
                "time_limit": row["time_limit"],
                "output_log": row["output_log"].replace("%j", job_id).replace(".txt", "-retry1.txt"),
            }
        )

    with LEDGER.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(submitted[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(submitted)
    for row in submitted:
        print(f"{row['manifest_id']}={row['job_id']} retry_of={row['retry_of']}")


if __name__ == "__main__":
    main()

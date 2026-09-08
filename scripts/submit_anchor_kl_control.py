#!/usr/bin/env python3
"""Submit the two predeclared adaptive-KL TPP jobs exactly once."""

from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains/2026-09-08/anchor_kl_control")
MANIFEST = ROOT / "anchor_kl_control_tpp_screen_20260908.csv"
WRAPPER = ROOT / "anchor_kl_adaptive_tpp.sbatch"
LEDGER = ROOT / "anchor_kl_control_submissions_20260909.tsv"


def main() -> None:
    if LEDGER.exists():
        raise SystemExit(f"refusing duplicate submission: {LEDGER} already exists")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2:
        raise SystemExit(f"expected exactly two manifest rows, found {len(rows)}")

    expected_seeds = {"1972442430", "1963100312"}
    if {row["seed"] for row in rows} != expected_seeds:
        raise SystemExit("unexpected seed set")
    for row in rows:
        required = {
            "arm": "adaptive_target",
            "value_head": "off",
            "initial_coefficient": "3",
            "target_kl": "0.1143",
            "cpus": "6",
            "memory_gib": "48",
            "time_limit": "3-00:00:00",
            "status": "ready_to_submit",
        }
        for field, value in required.items():
            if row[field] != value:
                raise SystemExit(f"{row['manifest_id']}: {field}={row[field]!r}, expected {value!r}")
        if not Path(row["source_checkpoint"]).exists():
            raise SystemExit(f"missing checkpoint: {row['source_checkpoint']}")

    submitted = []
    for row in rows:
        env = os.environ.copy()
        env.update(
            ANCHOR_SOURCE_CHECKPOINT=row["source_checkpoint"],
            ANCHOR_SEED=row["seed"],
            ANCHOR_TEACHER=row["teacher"],
            ANCHOR_TARGET=row["target_kl"],
        )
        job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                f"--job-name={row['manifest_id']}",
                f"--output={row['output_log']}",
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
                "seed": row["seed"],
                "role": row["role"],
                "source_training_job_id": row["source_training_job_id"],
                "source_epoch": row["source_epoch"],
                "source_checkpoint": row["source_checkpoint"],
                "teacher": row["teacher"],
                "target_kl": row["target_kl"],
                "cpus": row["cpus"],
                "memory_gib": row["memory_gib"],
                "time_limit": row["time_limit"],
                "output_log": row["output_log"].replace("%j", job_id),
            }
        )

    with LEDGER.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(submitted[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(submitted)
    for row in submitted:
        print(f"{row['manifest_id']}={row['job_id']}")


if __name__ == "__main__":
    main()

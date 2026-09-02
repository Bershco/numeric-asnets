#!/usr/bin/env python3
"""Submit the cross-domain PW pilot idempotently from its frozen manifest."""

from __future__ import annotations

import argparse
import base64
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SBATCH = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context/scripts/mcts_pw_cross_domain.sbatch")
OUT = Path("/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print submissions without calling Slurm or writing the ledger.",
    )
    parser.add_argument(
        "--nice", type=int, default=0,
        help="Positive Slurm nice value; larger values lower scheduling priority.",
    )
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if len(rows) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} rows, got {len(rows)}")
    existing = set()
    if args.ledger.exists():
        existing = {
            row["manifest_id"]
            for row in csv.DictReader(args.ledger.open(newline="", encoding="utf-8"), delimiter="\t")
        }
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ("manifest_id", "slurm_job_id", "submitted_at")
    if args.dry_run:
        for row in rows:
            checkpoint = Path(row["source_checkpoint"])
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            print(
                f"READY|{row['manifest_id']}|iterations={row['iterations']}|"
                f"checkpoint={checkpoint}"
            )
        return

    with args.ledger.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        for row in rows:
            if row["manifest_id"] in existing:
                continue
            encoded = base64.b64encode(row["source_checkpoint"].encode()).decode()
            exports = ",".join((
                "ALL", f"CHECKPOINT_B64={encoded}", f"SEED={row['seed']}",
                f"VALUE_HEAD={row['value_head']}", f"MANIFEST_ID={row['manifest_id']}",
                f"DOMAIN={row['domain']}", f"TEACHER={row['teacher']}",
                f"PW_MIN={row['pw_min_width']}", f"PW_C={row['pw_c']}",
                f"PW_ALPHA={row['pw_alpha']}", f"ITERATIONS={row['iterations']}",
            ))
            command = [
                "sbatch", f"--job-name={row['manifest_id']}",
                f"--output={OUT}/%x_%j.out", f"--export={exports}",
            ]
            if args.nice:
                command.append(f"--nice={args.nice}")
            command.append(str(SBATCH))
            result = subprocess.run(
                command,
                check=True, text=True, capture_output=True,
            )
            job_id = result.stdout.strip().split()[-1]
            writer.writerow({
                "manifest_id": row["manifest_id"], "slurm_job_id": job_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
            stream.flush()
            print(f"SUBMITTED|{row['manifest_id']}|{job_id}")


if __name__ == "__main__":
    main()

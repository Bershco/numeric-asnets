#!/usr/bin/env python3
"""Retry only operationally failed Stage-2 policy evaluations.

The primary submission ledger is immutable provenance: appearing there means
"submitted", not "scientifically complete".  This controller keeps a separate
attempt ledger and retries only terminal Slurm failures for identities that are
still present in the current ready manifest.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracking.submit_stage2_policy import read, submit, validate


FIELDS = [
    "manifest_id", "attempt", "previous_slurm_job_id", "previous_state",
    "slurm_job_id", "submitted_at", "source_checkpoint",
]
RETRYABLE = {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "TIMEOUT"}
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}


def states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    output = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "--format=JobIDRaw,State"], text=True,
    )
    result: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] in job_ids:
            result[fields[0]] = fields[1].split()[0].split("+")[0]
    return result


def append(path: Path, row: dict[str, str]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-ledger", type=Path, required=True)
    parser.add_argument("--retry-ledger", type=Path, required=True)
    parser.add_argument("--suffix-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-active", type=int, default=12)
    parser.add_argument("--max-per-cycle", type=int, default=12)
    args = parser.parse_args()

    rows = read(args.manifest)
    validate(rows)
    by_id = {row["manifest_id"]: row for row in rows}
    primary = read(args.primary_ledger, "\t") if args.primary_ledger.exists() else []
    retries = read(args.retry_ledger, "\t") if args.retry_ledger.exists() else []

    attempts: dict[str, list[tuple[str, str]]] = {}
    for row in primary:
        attempts.setdefault(row["manifest_id"], []).append((row["slurm_job_id"], "primary"))
    for row in retries:
        attempts.setdefault(row["manifest_id"], []).append((row["slurm_job_id"], "retry"))

    latest_ids = [items[-1][0] for identity, items in attempts.items() if identity in by_id]
    current = states(latest_ids)
    active_count = sum(current.get(job_id) in ACTIVE for job_id in latest_ids)
    allowance = min(args.max_per_cycle, max(0, args.max_active - active_count))

    candidates: list[tuple[dict[str, str], str, str, int]] = []
    for identity, row in by_id.items():
        history = attempts.get(identity, [])
        if not history:
            continue
        last_job = history[-1][0]
        state = current.get(last_job, "UNKNOWN")
        retry_count = len(history) - 1
        if state in RETRYABLE and retry_count < args.max_attempts:
            candidates.append((row, last_job, state, retry_count + 1))

    print(
        f"[RETRY SCAN] ready={len(rows)} submitted_identities={len(attempts)} "
        f"active={active_count} retryable={len(candidates)} allowance={allowance}",
        flush=True,
    )
    for row, previous, previous_state, attempt in candidates[:allowance]:
        job_id = submit(row, False, args.suffix_prefix, args.output_prefix)
        append(args.retry_ledger, {
            "manifest_id": row["manifest_id"], "attempt": str(attempt),
            "previous_slurm_job_id": previous, "previous_state": previous_state,
            "slurm_job_id": job_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": row["source_checkpoint_ref"],
        })
        print(f"[RETRIED] {previous} -> {job_id} {row['manifest_id']}", flush=True)


if __name__ == "__main__":
    main()

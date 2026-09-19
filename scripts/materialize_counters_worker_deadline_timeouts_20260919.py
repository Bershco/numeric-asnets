#!/usr/bin/env python3
"""Persist Counters timeouts when the rolling evaluator omits timeout JSONL.

The affected recovery launched only the two remaining identities.  Both
workers remained live until the evaluator's six-hour deadline, after which
``run_experiment`` returned non-zero without printing its usual timeout line.
This reconciler accepts that evidence only under strict, reproducible guards;
it is not a general way to turn failed jobs into scientific failures.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


START_RE = re.compile(
    r"\[EVAL INSTANCE\] started number=(?P<number>\d+) "
    r"path=(?P<path>\S+) pid=(?P<pid>\d+)"
)
FATAL_RE = re.compile(r"Traceback|Out of memory|oom-kill|Killed process", re.I)


def elapsed_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    hours, minutes, seconds = map(int, value.split(":"))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def materialize(
    ledger: Path,
    log: Path,
    *,
    job_id: str,
    state: str,
    exit_code: str,
    elapsed: int,
    timeout_seconds: int = 21600,
) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_number = {int(record["instance_number"]): record for record in records}
    if sorted(by_number) != list(range(1, 58)):
        raise RuntimeError("deadline reconciliation requires exact 1..57 ledger")
    signatures = {record.get("evaluation_signature") for record in records}
    if len(signatures) != 1 or None in signatures:
        raise RuntimeError("ledger has missing or mixed evaluation signatures")

    text = log.read_text(errors="replace")
    if FATAL_RE.search(text):
        raise RuntimeError("recovery log contains an operational-failure marker")
    starts = {
        int(match.group("number")): match.group("path")
        for match in START_RE.finditer(text)
    }
    if sorted(starts) != [58, 59]:
        raise RuntimeError(f"expected only starts 58 and 59, found {sorted(starts)}")
    for number in starts:
        if re.search(rf"\[EVAL INSTANCE\] (?:timeout|completed).*number={number}\b", text):
            raise RuntimeError(f"instance {number} already has explicit terminal evidence")

    if state != "FAILED" or exit_code != "1:0":
        raise RuntimeError(f"unexpected Slurm outcome: {state} {exit_code}")
    # Allow setup/teardown grace, but require that the scientific deadline was
    # reached well before the eight-hour allocation itself could expire.
    if not timeout_seconds <= elapsed <= timeout_seconds + 600:
        raise RuntimeError(f"job elapsed {elapsed}s is not a six-hour deadline exit")

    additions = []
    signature = next(iter(signatures))
    for number in (58, 59):
        additions.append({
            "elapsed_seconds": float(timeout_seconds),
            "evaluation_signature": signature,
            "hit_goal": False,
            "instance_number": number,
            "instance_path": starts[number],
            "plan": None,
            "status": "hard_timeout",
            "steps": -2,
            "terminal_evidence_exit_code": exit_code,
            "terminal_evidence_job_id": job_id,
            "terminal_evidence_log": str(log),
            "terminal_evidence_reason": "worker deadline exit without JSONL serialization",
            "terminal_evidence_slurm_elapsed_seconds": elapsed,
        })
    fd = os.open(ledger, os.O_WRONLY | os.O_APPEND)
    try:
        for record in additions:
            os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "job_id": job_id,
        "terminal_before": 57,
        "timeouts_added": [58, 59],
        "terminal_after": 59,
    }


def slurm_outcome(job_id: str) -> tuple[str, str, int]:
    output = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", job_id,
        "-o", "JobIDRaw,State,ExitCode,Elapsed",
    ], text=True)
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    matches = [row for row in rows if row[0] == job_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one accounting row for {job_id}, found {len(matches)}")
    _, state, exit_code, elapsed = matches[0]
    return state.split("+", 1)[0], exit_code, elapsed_seconds(elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    state, exit_code, elapsed = slurm_outcome(args.job_id)
    print(json.dumps(materialize(
        args.ledger, args.log, job_id=args.job_id, state=state,
        exit_code=exit_code, elapsed=elapsed,
    ), sort_keys=True))


if __name__ == "__main__":
    main()

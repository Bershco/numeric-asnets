#!/usr/bin/env python3
"""Materialize terminal hard-timeout evidence into Counters completion ledgers.

The rolling evaluator historically persisted successful and ordinary-unsolved
results but printed hard timeouts only to stdout.  This reconciler joins every
strict/recovery log in one task's output directory with its durable JSONL file,
adds only previously absent hard-timeout identities, and refuses conflicting
paths or duplicate terminal identities.  Crashes and merely started instances
remain unclassified and are therefore the only identities a recovery reruns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


TIMEOUT_RE = re.compile(
    r"\[EVAL INSTANCE\] timeout number=(?P<number>\d+) "
    r"path=(?P<path>\S+) limit=(?P<limit>[0-9.]+)s"
)


def load_records(path: Path) -> tuple[list[dict], dict[int, dict]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_number: dict[int, dict] = {}
    for record in records:
        number = int(record["instance_number"])
        previous = by_number.get(number)
        if previous is not None and previous != record:
            raise RuntimeError(f"conflicting durable records for instance {number}")
        by_number[number] = record
    return records, by_number


def collect_timeouts(logs: list[Path], expected_limit: float = 21600.0) -> dict[int, dict]:
    terminal: dict[int, dict] = {}
    for log in sorted(logs):
        for line_number, line in enumerate(
                log.read_text(errors="replace").splitlines(), start=1):
            match = TIMEOUT_RE.search(line)
            if match is None:
                continue
            number = int(match.group("number"))
            if not 1 <= number <= 59:
                raise RuntimeError(
                    f"timeout instance outside 1..59 in {log}:{line_number}: {number}"
                )
            limit = float(match.group("limit"))
            if limit != expected_limit:
                raise RuntimeError(
                    f"unexpected timeout cap in {log}:{line_number}: {limit}"
                )
            record = {
                "instance_number": number,
                "instance_path": match.group("path"),
                "status": "hard_timeout",
                "hit_goal": False,
                "steps": -2,
                "plan": None,
                "elapsed_seconds": limit,
                "terminal_evidence_log": str(log),
                "terminal_evidence_line": line_number,
            }
            previous = terminal.get(number)
            if previous is not None and previous["instance_path"] != record["instance_path"]:
                raise RuntimeError(
                    f"conflicting timeout paths for instance {number}: "
                    f"{previous['instance_path']} versus {record['instance_path']}"
                )
            terminal[number] = record
    return terminal


def reconcile(ledger: Path, logs: list[Path]) -> dict[str, object]:
    records, durable = load_records(ledger)
    if not records:
        raise RuntimeError(f"empty durable ledger: {ledger}")
    signatures = {record.get("evaluation_signature") for record in records}
    if len(signatures) != 1 or None in signatures:
        raise RuntimeError(f"ledger has missing or mixed signatures: {ledger}")
    signature = next(iter(signatures))
    timeouts = collect_timeouts(logs)
    additions = []
    for number, timeout in sorted(timeouts.items()):
        previous = durable.get(number)
        if previous is not None:
            if previous.get("instance_path") != timeout["instance_path"]:
                raise RuntimeError(
                    f"timeout/durable path mismatch for instance {number}"
                )
            continue
        timeout["evaluation_signature"] = signature
        additions.append(timeout)
    if additions:
        fd = os.open(ledger, os.O_WRONLY | os.O_APPEND)
        try:
            for record in additions:
                os.write(
                    fd,
                    (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"),
                )
            os.fsync(fd)
        finally:
            os.close(fd)
    return {
        "ledger": str(ledger),
        "logs": [str(path) for path in logs],
        "durable_before": len(durable),
        "timeout_evidence": len(timeouts),
        "timeouts_added": len(additions),
        "timeout_events_already_durable": len(timeouts) - len(additions),
        "terminal_after": len(durable) + len(additions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.ledger, args.log), sort_keys=True))


if __name__ == "__main__":
    main()

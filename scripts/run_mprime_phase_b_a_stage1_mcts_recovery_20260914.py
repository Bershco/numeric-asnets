#!/usr/bin/env python3
"""Run exactly one previously unclassified MPrime Stage-1 MCTS instance."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--recovery-index", type=int, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    with args.recovery_manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 15:
        raise ValueError(f"expected 15 exact recovery rows, found {len(rows)}")
    matches = [row for row in rows if int(row["recovery_index"]) == args.recovery_index]
    if len(matches) != 1:
        raise ValueError(f"recovery index {args.recovery_index}: found {len(matches)} rows")
    row = matches[0]
    mode = row["value_head"]
    if mode not in {"off", "on"}:
        raise ValueError(f"invalid value-head mode: {mode}")
    manifest = args.checkout / "experiment_tracking" / "mprime_phase_b_a_stage1_mcts_20260913" / f"manifest_{mode}.csv"
    command = [
        sys.executable,
        str(args.checkout / "scripts" / "run_mprime_phase_b_a_stage1_mcts_20260913.py"),
        "--manifest", str(manifest),
        "--index", row["manifest_index"],
        "--checkout", str(args.checkout),
        "--production", str(args.production),
        "--container", str(args.container),
        "--validator", str(args.validator),
        "--output-root", str(args.output_root),
        "--code-commit", args.code_commit,
        "--smoke-instance", row["instance_number"],
        "--smoke-timeout", "21600",
    ]
    print(
        f"[MPRIME S1 EXACT RECOVERY] index={args.recovery_index} mode={mode} "
        f"seed={row['seed']} instance={row['instance_number']} source={row['source_job_id']}"
    )
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())

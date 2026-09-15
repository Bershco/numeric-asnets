#!/usr/bin/env python3
"""Resume only truly unclassified identities from a terminal Counters cell."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ARRAY_JOB_ID = "21233925"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= args.task < len(rows):
        raise RuntimeError(f"recovery task outside manifest: {args.task}")
    row = rows[args.task]
    checkpoint = Path(row["checkpoint"])
    output = Path(row["remote_output"])
    completion = output / f"{SOURCE_ARRAY_JOB_ID}_{args.task}.completed.jsonl"
    if not checkpoint.is_dir() or not completion.is_file():
        raise RuntimeError(f"missing checkpoint or source ledger: {checkpoint}, {completion}")
    logs = sorted(output.glob("*.txt"))
    if not logs:
        raise RuntimeError(f"no source/recovery logs to reconcile: {output}")
    reconciler = Path(__file__).with_name(
        "reconcile_counters_tie_break_terminal_records_20260915.py"
    )
    reconcile_command = [
        sys.executable, str(reconciler), "--ledger", str(completion),
    ]
    for source_log in logs:
        reconcile_command.extend(("--log", str(source_log)))
    subprocess.run(reconcile_command, check=True)
    before = sum(1 for line in completion.read_text().splitlines() if line.strip())
    if before > 59:
        raise RuntimeError(f"reconciled ledger exceeds 59 identities: {before}")
    if before == 59:
        print(f"[RECOVERY COMPLETE] task={args.task} terminal=59/59")
        return
    recovery_job = os.environ["SLURM_ARRAY_JOB_ID"]
    log = output / f"{recovery_job}_{args.task}_recovery_{row['tie_break']}_{row['seed']}.txt"
    command = [
        "./run_experiment",
        "experiments_numeric.architecture_2.counters_mcts",
        "experiments_numeric.domain.counters",
        "--resume-from", str(checkpoint),
        "--eval-with-mcts",
        "--eval-mcts-root-visit-tie-break", row["tie_break"],
        "--mcts-expansion-size", "5",
        "--mcts-iterations", "20",
        "--mcts-exploration-weight", "0.1",
        "--use-estimator", "0.5",
        "--eval-scheduling", "rolling",
        "--eval-completion-file", str(completion),
        "--eval-instance-timeout", "21600",
        "--eval-max-actions", "10000",
        "--num-workers", "1",
        "--jpddl-max-heap", "4g",
        "--worker-logs",
        "--disable-value-head",
        "--random-seed", row["seed"],
    ]
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            "[COUNTERS TIE STRICT RECOVERY] "
            f"source_array={SOURCE_ARRAY_JOB_ID} task={args.task} "
            f"classified_before={before} rule={row['tie_break']} seed={row['seed']}\n"
        )
        stream.flush()
        result = subprocess.run(
            command, cwd=args.repo / "asnets", stdout=stream,
            stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

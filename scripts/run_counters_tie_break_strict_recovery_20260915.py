#!/usr/bin/env python3
"""Resume only unclassified identities from the three strict Counters OOM cells."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


SOURCE_ARRAY_JOB_ID = "21233925"
ALLOWED_TASKS = {4, 6, 14}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    if args.task not in ALLOWED_TASKS:
        raise RuntimeError(f"recovery task must be one of {sorted(ALLOWED_TASKS)}")
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[args.task]
    checkpoint = Path(row["checkpoint"])
    output = Path(row["remote_output"])
    completion = output / f"{SOURCE_ARRAY_JOB_ID}_{args.task}.completed.jsonl"
    if not checkpoint.is_dir() or not completion.is_file():
        raise RuntimeError(f"missing checkpoint or source ledger: {checkpoint}, {completion}")
    before = sum(1 for line in completion.read_text().splitlines() if line.strip())
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
        "--num-workers", "3",
        "--jpddl-max-heap", "4g",
        "--worker-logs",
        "--disable-value-head",
        "--random-seed", row["seed"],
    ]
    with log.open("w", encoding="utf-8") as stream:
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

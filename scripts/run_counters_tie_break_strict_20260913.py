#!/usr/bin/env python3
"""Run one full Counters tie-break confirmation cell from the frozen manifest."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or [int(row["array_index"]) for row in rows] != list(range(20)):
        raise RuntimeError("strict tie-break manifest must contain contiguous rows 0..19")
    row = rows[args.task]
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    output = Path(row["remote_output"])
    output.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_ARRAY_JOB_ID"]
    task = os.environ["SLURM_ARRAY_TASK_ID"]
    log = output / f"{job}_{task}_{row['tie_break']}_{row['seed']}.txt"
    completion = output / f"{job}_{task}.completed.jsonl"
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
            "[COUNTERS TIE STRICT] "
            f"rule={row['tie_break']} seed={row['seed']} source_policy_job={row['source_policy_job_id']} "
            f"checkpoint={checkpoint}\n"
        )
        stream.flush()
        result = subprocess.run(
            command,
            cwd=args.repo / "asnets",
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    validator = "/home/hersco/tools/VAL/build/bin/Validate"
    summary = output / f"{job}_{task}_{row['tie_break']}_{row['seed']}.val.csv"
    subprocess.run([
        "python", str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
        "--log", str(log), "--domain", "counters",
        "--validator", validator, "--summary-csv", str(summary),
    ], cwd=args.repo / "asnets", check=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

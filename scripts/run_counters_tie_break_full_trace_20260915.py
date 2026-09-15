#!/usr/bin/env python3
"""Run one exact Counters full-root trace identity from a frozen manifest."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


TOTAL_INSTANCES = 59


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["array_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("trace manifest must contain contiguous array indices")
    row = rows[args.task]
    number = int(row["instance_number"])
    if row["tie_break"] not in {"action_id", "policy"}:
        raise RuntimeError(f"unsupported tie-break rule: {row['tie_break']}")
    if not 1 <= number <= TOTAL_INSTANCES:
        raise RuntimeError(f"invalid evaluator identity: {number}")
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    output = Path(row["remote_output"])
    output.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_ARRAY_JOB_ID"]
    task = os.environ["SLURM_ARRAY_TASK_ID"]
    stem = f"{job}_{task}_{row['tie_break']}_{row['seed']}_instance_{number}"
    log = output / f"{stem}.txt"
    completion = output / f"{stem}.completed.jsonl"
    skip = ",".join(
        str(candidate)
        for candidate in range(1, TOTAL_INSTANCES + 1)
        if candidate != number
    )
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
        "--skip-instance-numbers", skip,
        "--num-workers", "1",
        "--jpddl-max-heap", "4g",
        "--worker-logs",
        "--action-debug",
        "--disable-value-head",
        "--random-seed", row["seed"],
    ]
    with log.open("w", encoding="utf-8") as stream:
        stream.write(
            "[COUNTERS TIE FULL TRACE] "
            f"rule={row['tie_break']} seed={row['seed']} instance={number} "
            f"source_strict_task={row['source_strict_task']} "
            f"source_policy_job={row['source_policy_job_id']} "
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
    if result.returncode:
        raise SystemExit(result.returncode)
    records = [line for line in completion.read_text().splitlines() if line.strip()]
    if len(records) != 1 or f'"instance_number": {number}' not in records[0]:
        raise RuntimeError(
            f"trace task did not classify exactly evaluator identity {number}: {completion}"
        )
    validator = "/home/hersco/tools/VAL/build/bin/Validate"
    summary = output / f"{stem}.val.csv"
    subprocess.run([
        "python", str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
        "--log", str(log), "--domain", "counters",
        "--validator", validator, "--summary-csv", str(summary),
    ], cwd=args.repo / "asnets", check=True)
    prefix = output / stem
    subprocess.run([
        "python", str(args.repo / "scripts/summarize_mcts_visit_distribution.py"),
        str(log), "--output-prefix", str(prefix),
    ], cwd=args.repo, check=True)


if __name__ == "__main__":
    main()

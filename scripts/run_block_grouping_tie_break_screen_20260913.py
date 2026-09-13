#!/usr/bin/env python3
"""Run one matched Block Grouping tie-break diagnostic arm."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


SKIP_NON_TARGETS = "3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,20"


def load_row(manifest: Path, task: int) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 or [int(row["array_index"]) for row in rows] != [0, 1]:
        raise RuntimeError("Block Grouping screen manifest must contain rows 0 and 1")
    row = rows[task]
    if row["tie_break"] not in {"action_id", "policy"}:
        raise RuntimeError(f"unexpected tie-break: {row['tie_break']}")
    if row["target_slots_one_based"] != "1,2,6,19":
        raise RuntimeError("target-slot identity changed")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    row = load_row(args.manifest, args.task)
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    output = Path(row["remote_output"])
    output.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ["SLURM_JOB_ID"])
    task = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.task))
    stem = f"{job}_{task}_{row['tie_break']}_{row['seed']}"
    log = output / f"{stem}.txt"
    completion = output / f"{stem}.completed.jsonl"
    timeout = "600" if args.preflight else "21600"
    skip = "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" if args.preflight else SKIP_NON_TARGETS
    command = [
        "./run_experiment",
        "experiments_numeric.architecture_2.block_grouping_mcts",
        "experiments_numeric.domain.block_grouping",
        "--resume-from", str(checkpoint),
        "--eval-with-mcts",
        "--eval-mcts-root-visit-tie-break", row["tie_break"],
        "--mcts-expansion-size", "5",
        "--mcts-iterations", "20",
        "--mcts-exploration-weight", "0.1",
        "--use-estimator", "0.5",
        "--eval-scheduling", "rolling",
        "--eval-completion-file", str(completion),
        "--eval-instance-timeout", timeout,
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
            "[BLOCK GROUPING TIE SCREEN] "
            f"rule={row['tie_break']} seed={row['seed']} targets={row['target_instances']} "
            f"source_policy_job={row['source_policy_job_id']} source_mcts_job={row['source_mcts_job_id']}\n"
        )
        stream.flush()
        result = subprocess.run(command, cwd=args.repo / "asnets", stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    summary = output / f"{stem}.val.csv"
    subprocess.run([
        "python", str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
        "--log", str(log), "--domain", "block_grouping",
        "--validator", "/home/hersco/tools/VAL/build/bin/Validate",
        "--summary-csv", str(summary),
    ], cwd=args.repo / "asnets", check=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

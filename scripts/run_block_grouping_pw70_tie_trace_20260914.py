#!/usr/bin/env python3
"""Trace four predeclared Block Grouping policy-success/PW70-timeout cases."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


def load_row(path: Path, task: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4 or [int(row["array_index"]) for row in rows] != list(range(4)):
        raise RuntimeError("expected the frozen four-row manifest")
    row = rows[task]
    if row["value_head"] not in {"off", "on"}:
        raise RuntimeError("invalid value-head mode")
    if not 1 <= int(row["target_slot_one_based"]) <= 20:
        raise RuntimeError("invalid target slot")
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
    job = os.environ["SLURM_JOB_ID"]
    task = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.task))
    stem = f"{job}_{task}_{row['value_head']}_{row['seed']}_{Path(row['target_instance']).stem}"
    log = output / f"{stem}.txt"
    completion = output / f"{stem}.completed.jsonl"
    target = int(row["target_slot_one_based"])
    skip = ",".join(str(slot) for slot in range(1, 21) if slot != target)
    command = [
        "./run_experiment",
        "experiments_numeric.architecture_2.block_grouping_mcts",
        "experiments_numeric.domain.block_grouping",
        "--resume-from", str(checkpoint),
        "--eval-with-mcts",
        "--eval-mcts-terminal-safe-action-selection",
        "--eval-mcts-root-visit-tie-break", "action_id",
        "--mcts-progressive-widening",
        "--mcts-pw-min-width", "3",
        "--mcts-pw-c", "0.6",
        "--mcts-pw-alpha", "0.5",
        "--mcts-expansion-size", "20",
        "--mcts-iterations", "70",
        "--mcts-exploration-weight", "0.1",
        "--use-estimator", "0.5",
        "--eval-scheduling", "rolling",
        "--eval-completion-file", str(completion),
        "--eval-instance-timeout", "180" if args.preflight else "21600",
        "--eval-max-actions", "10000",
        "--skip-instance-numbers", skip,
        "--num-workers", "1",
        "--jpddl-max-heap", "4g",
        "--worker-logs",
        "--action-debug",
        "--random-seed", row["seed"],
    ]
    if row["value_head"] == "off":
        command.append("--disable-value-head")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(
            "[BG PW70 TIE TRACE] logging-only exact replay "
            f"seed={row['seed']} vh={row['value_head']} target={row['target_instance']} "
            f"policy_job={row['source_policy_job_id']} pw_job={row['source_pw_job_id']}\n"
        )
        stream.flush()
        result = subprocess.run(command, cwd=args.repo / "asnets", stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

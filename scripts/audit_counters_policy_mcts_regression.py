#!/usr/bin/env python3
"""Join policy successes to rolling MCTS outcomes for one evaluation pair."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


COMPLETED_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(?P<number>\d+) .*?"
    r"status=(?P<status>\S+) .*?steps=(?P<steps>\d+)"
)
TIMEOUT_RE = re.compile(r"\[EVAL INSTANCE\] timeout number=(?P<number>\d+)")
PLAN_RE = re.compile(
    r"\[EVAL\]\[PLAN\].*?fz_instance_(?P<pddl_number>\d+)\.pddl"
    r"\s*\|\s*steps=(?P<steps>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--policy-log", type=Path, required=True)
    parser.add_argument("--mcts-log", type=Path, required=True)
    parser.add_argument("--completion-jsonl", type=Path, required=True)
    parser.add_argument("--mcts-job-id", required=True)
    parser.add_argument("--total", type=int, default=59)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_text = args.policy_log.read_text(errors="replace")
    mcts_text = args.mcts_log.read_text(errors="replace")

    policy = {
        int(match.group("number")): {
            "status": match.group("status"),
            "steps": int(match.group("steps")),
        }
        for match in COMPLETED_RE.finditer(policy_text)
    }
    if not policy:
        # Legacy wave-scheduled policy logs predate per-instance completion lines.
        policy = {
            int(match.group("pddl_number")) - 1: {
                "status": "success",
                "steps": int(match.group("steps")),
            }
            for match in PLAN_RE.finditer(policy_text)
        }
    mcts = {}
    with args.completion_jsonl.open() as handle:
        for line in handle:
            record = json.loads(line)
            mcts[int(record["instance_number"])] = record
    timeout_instances = {
        int(match.group("number")) for match in TIMEOUT_RE.finditer(mcts_text)
    }

    rows = []
    for number in range(1, args.total + 1):
        policy_record = policy.get(number, {})
        mcts_record = mcts.get(number)
        if mcts_record is not None:
            mcts_status = mcts_record["status"]
            mcts_steps = mcts_record.get("steps", "")
        elif number in timeout_instances:
            mcts_status = "instance_timeout_6h"
            mcts_steps = ""
        else:
            mcts_status = "oom_interrupted_without_terminal_record"
            mcts_steps = ""
        rows.append(
            {
                "seed": args.seed,
                "instance_number": number,
                "policy_status": policy_record.get("status", "not_recorded"),
                "policy_steps": policy_record.get("steps", ""),
                "mcts_status": mcts_status,
                "mcts_steps": mcts_steps,
                "mcts_job_id": args.mcts_job_id,
                "policy_log": args.policy_log,
                "mcts_log": args.mcts_log,
                "completion_jsonl": args.completion_jsonl,
            }
        )

    if args.summary:
        status_counts = {}
        for row in rows:
            status_counts[row["mcts_status"]] = status_counts.get(row["mcts_status"], 0) + 1
        lost = [
            row
            for row in rows
            if row["policy_status"] == "success" and row["mcts_status"] != "success"
        ]
        lost_steps = [int(row["policy_steps"]) for row in lost if row["policy_steps"] != ""]
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "mcts_job_id": args.mcts_job_id,
                    "policy_successes": sum(row["policy_status"] == "success" for row in rows),
                    "mcts_status_counts": status_counts,
                    "policy_successes_lost_by_mcts": len(lost),
                    "lost_policy_steps_min": min(lost_steps) if lost_steps else None,
                    "lost_policy_steps_median": statistics.median(lost_steps) if lost_steps else None,
                    "lost_policy_steps_max": max(lost_steps) if lost_steps else None,
                    "lost_by_mcts_status": {
                        status: sum(row["mcts_status"] == status for row in lost)
                        for status in sorted({row["mcts_status"] for row in lost})
                    },
                    "policy_log": str(args.policy_log),
                    "mcts_log": str(args.mcts_log),
                    "completion_jsonl": str(args.completion_jsonl),
                },
                sort_keys=True,
            )
        )
        return

    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=[
            "seed",
            "instance_number",
            "policy_status",
            "policy_steps",
            "mcts_status",
            "mcts_steps",
            "mcts_job_id",
            "policy_log",
            "mcts_log",
            "completion_jsonl",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()

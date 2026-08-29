#!/usr/bin/env python3
"""Extract compact trajectory evidence for Counters policy-only successes."""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path


PLAN_RE = re.compile(r"\[EVAL\]\[PLAN\].*?\|\s*([^|]+\.pddl)\s*\| steps=(\d+) \| plan=(\[.*\])$")
TARGETS = {
    "1963100312": [f"fz_instance_{number}.pddl" for number in range(51, 61)],
    "1073581256": ["fz_instance_53.pddl", "fz_instance_54.pddl"],
}


def policy_plans(path: Path) -> dict[str, list[str]]:
    plans: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = PLAN_RE.search(line.rstrip())
            if match:
                plans[match.group(1).strip()] = list(ast.literal_eval(match.group(3)))
    return plans


def completion_plans(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            records[Path(str(row["instance_path"])).name] = row
    return records


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-1963100312", type=Path, required=True)
    parser.add_argument("--policy-1073581256", type=Path, required=True)
    parser.add_argument("--mcts-1963100312", type=Path, required=True)
    parser.add_argument("--mcts-1073581256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy_sources = {
        "1963100312": args.policy_1963100312,
        "1073581256": args.policy_1073581256,
    }
    mcts_sources = {
        "1963100312": args.mcts_1963100312,
        "1073581256": args.mcts_1073581256,
    }
    output: list[dict[str, object]] = []
    for seed, targets in TARGETS.items():
        policies = policy_plans(policy_sources[seed])
        mcts = completion_plans(mcts_sources[seed])
        for instance in targets:
            policy = policies.get(instance, [])
            record = mcts.get(instance)
            mcts_plan = list(record.get("plan", [])) if record else []
            common = 0
            for left, right in zip(policy, mcts_plan):
                if left != right:
                    break
                common += 1
            policy_counts, mcts_counts = Counter(policy), Counter(mcts_plan)
            over = sum(max(0, mcts_counts[action] - policy_counts[action]) for action in mcts_counts)
            decrement_count = sum(count for action, count in mcts_counts.items() if action.startswith("decrement "))
            output.append({
                "seed": seed, "instance": instance,
                "policy_steps": len(policy),
                "mcts_status": record.get("status", "not_completed") if record else "not_completed",
                "mcts_steps_persisted": len(mcts_plan),
                "mcts_elapsed_seconds": record.get("elapsed_seconds", "") if record else "",
                "common_prefix_steps": common,
                "mcts_actions_above_policy_multiset": over,
                "mcts_decrement_actions": decrement_count,
                "policy_actions_logged": bool(policy),
                "mcts_actions_logged": bool(record),
                "policy_log": str(policy_sources[seed]),
                "mcts_completion_jsonl": str(mcts_sources[seed]),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(f"rows={len(output)} output={args.output}")


if __name__ == "__main__":
    main()

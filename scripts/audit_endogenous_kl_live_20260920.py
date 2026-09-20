#!/usr/bin/env python3
"""Read-only live snapshot of the endogenous corrected-KL campaign."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


CAMPAIGN = Path("/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot")


def load_jsonl(path: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    if not path.is_file():
        return result
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            number = int(record["instance_number"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        result[number] = record
    return result


def main() -> None:
    manifest = list(csv.DictReader((CAMPAIGN / "selected_mcts_manifest.csv").open(newline="")))
    rows = []
    for row in manifest:
        completion = (
            CAMPAIGN / "selected_mcts" / row["method"]
            / f"arm_{row['arm_index']}" / "completion.jsonl"
        )
        records = load_jsonl(completion)
        statuses = Counter(str(record.get("status", "unknown")) for record in records.values())
        scores = {}
        for label, cutoff in (("30m", 1800), ("2h", 7200), ("6h", 21600)):
            scores[label] = sum(
                str(record.get("status")) == "success"
                and float(record.get("elapsed_seconds", float("inf"))) <= cutoff
                for record in records.values()
            )
        rows.append({
            "array_index": int(row["array_index"]),
            "domain": row["domain"],
            "seed": int(row["seed"]),
            "kl_semantics": row["semantics"],
            "method": row["method"],
            "selected_epoch": int(row["selected_epoch"]),
            "classified": len(records),
            **scores,
            "statuses": dict(statuses),
            "completion": str(completion),
        })
    policy_manifest = list(csv.DictReader((CAMPAIGN / "policy_eval_manifest.csv").open(newline="")))
    policy_results = list((CAMPAIGN / "outputs").glob("**/policy_epoch_*/result.json"))
    missing_policy = []
    for index, row in enumerate(policy_manifest):
        result = (
            CAMPAIGN / "outputs"
            / f"arm_{row['arm_index']}_{row['domain']}_{row['seed']}_{row['semantics']}"
            / f"policy_epoch_{int(row['epoch']):04d}" / "result.json"
        )
        if not result.is_file():
            missing_policy.append({
                "manifest_index": index,
                "identity": row["identity"],
                "result": str(result),
            })
    selected = {
        int(row["arm_index"]): int(row["selected_epoch"])
        for row in csv.DictReader((CAMPAIGN / "selected_endpoints.csv").open(newline=""))
    }
    policy_trajectories = []
    for arm_index in range(8):
        arm_dirs = list((CAMPAIGN / "outputs").glob(f"arm_{arm_index}_*"))
        if len(arm_dirs) != 1:
            continue
        arm_dir = arm_dirs[0]
        epochs = (0, selected[arm_index], 99)
        records = {
            epoch: json.loads(
                (arm_dir / f"policy_epoch_{epoch:04d}" / "result.json").read_text()
            )
            for epoch in epochs
        }
        base = records[0]
        policy_trajectories.append({
            "arm_index": arm_index,
            "domain": base["domain"],
            "seed": base["seed"],
            "kl_semantics": base["semantics"],
            "start_epoch": 0,
            "start_score": records[0]["score"],
            "selected_epoch": selected[arm_index],
            "selected_score": records[selected[arm_index]]["score"],
            "final_epoch": 99,
            "final_score": records[99]["score"],
        })
    print(json.dumps({
        "policy_manifest_identities": len(policy_manifest),
        "policy_result_files": len(policy_results),
        "missing_policy": missing_policy,
        "policy_trajectories": policy_trajectories,
        "mcts": rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

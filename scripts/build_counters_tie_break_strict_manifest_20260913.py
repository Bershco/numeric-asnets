#!/usr/bin/env python3
"""Build the strict same-build Counters Stage-1 VH-off tie-break manifest."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking/experiment_results.csv"
OUTPUT = ROOT / "experiment_tracking/counters_tie_break_strict_stage1_20260913/manifest.csv"
RULES = ("action_id", "policy")


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    policy = [
        row for row in rows
        if row["task_type"] == "policy_eval"
        and row["domain"] == "counters"
        and row["value_head"] == "off"
        and row["stage"] == "stage1"
        and row["endpoint"] == "validation_selected"
    ]
    by_seed = {row["seed"]: row for row in policy}
    if len(policy) != 10 or len(by_seed) != 10:
        raise RuntimeError(f"expected ten unique Stage-1 VH-off policy rows, got {len(policy)}/{len(by_seed)}")
    seeds = sorted(by_seed, key=int)
    manifest = []
    for rule_index, rule in enumerate(RULES):
        for seed_index, seed in enumerate(seeds):
            source = by_seed[seed]
            manifest.append({
                "array_index": str(rule_index * 10 + seed_index),
                "tie_break": rule,
                "seed": seed,
                "value_head": "off",
                "stage": "stage1",
                "endpoint": "validation_selected",
                "policy_score": source["score"],
                "source_training_job_id": source["source_training_job_id"],
                "source_policy_job_id": source["job_id"],
                "checkpoint": source["checkpoint"],
                "source_training_log": source["source_training_log"],
                "source_policy_log": source["source_evaluation_log"],
                "search": "narrow_5_children_20_simulations",
                "instances": "1-59",
                "workers": "3",
                "cpus": "6",
                "ram_gib": "120",
                "instance_timeout": "6h",
                "job_walltime": "72h",
                "nice": "10000",
                "remote_output": (
                    "/home/hersco/training_new_domains/2026-09-13/"
                    f"counters_tie_break_strict_stage1/{rule}/{seed}"
                ),
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(manifest)
    print(f"rows={len(manifest)} seeds={len(seeds)} rules={','.join(RULES)} output={OUTPUT}")


if __name__ == "__main__":
    main()

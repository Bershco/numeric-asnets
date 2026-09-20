#!/usr/bin/env python3
"""Static manifest and optional four-arm smoke verification."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ARMS = {
    "rollout": ("policy_rollout", 0.0),
    "learned-only": ("value", 0.0),
    "enhsp-only": ("value", 1.0),
    "blend": ("value", 0.5),
}


def arm_name(manifest_id: str) -> str:
    for suffix in ARMS:
        if manifest_id.endswith("-" + suffix):
            return suffix
    raise ValueError(f"unknown arm suffix: {manifest_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--smoke-done", type=Path)
    parser.add_argument("--write-gate", type=Path)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16:
        raise SystemExit(f"expected 16 rows, found {len(rows)}")
    if len({row["manifest_id"] for row in rows}) != len(rows):
        raise SystemExit("manifest_id values must be unique")

    cells = defaultdict(list)
    for row in rows:
        arm = arm_name(row["manifest_id"])
        expected_evaluator, expected_alpha = ARMS[arm]
        if row["leaf_evaluator"] != expected_evaluator:
            raise SystemExit(f"wrong evaluator for {row['manifest_id']}")
        if float(row["use_estimator"]) != expected_alpha:
            raise SystemExit(f"wrong estimator coefficient for {row['manifest_id']}")
        if row["value_head"] != "on" or row["stage"] != "stage1":
            raise SystemExit(f"pilot must use Stage-1 VH-on: {row['manifest_id']}")
        cells[(row["domain"], row["seed"])].append(row)

    if set(cells) != {
        (domain, seed)
        for domain in ("drone", "fo_counters")
        for seed in ("1963100312", "2011206605")
    }:
        raise SystemExit(f"unexpected matched cells: {sorted(cells)}")
    for cell, cell_rows in cells.items():
        if Counter(arm_name(row["manifest_id"]) for row in cell_rows) != Counter(ARMS.keys()):
            raise SystemExit(f"incomplete arm set for {cell}")
        if len({row["source_checkpoint"] for row in cell_rows}) != 1:
            raise SystemExit(f"arms do not share one checkpoint for {cell}")
        for field, expected in {
            "width": "20",
            "iterations": "70",
            "puct": "0.1",
            "workers": "3",
            "instance_timeout_seconds": "21600",
        }.items():
            if {row[field] for row in cell_rows} != {expected}:
                raise SystemExit(f"{field} is not frozen at {expected} for {cell}")

    payload = {"manifest_rows": 16, "matched_cells": 4, "static_checks": "passed"}
    if args.smoke_done:
        expected = [row for row in rows[:4]]
        records = []
        for row in expected:
            record_path = args.smoke_done / f"{row['manifest_id']}.json"
            if not record_path.is_file():
                raise SystemExit(f"missing smoke marker: {record_path}")
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        if len({record["checkpoint_sha256"] for record in records}) != 1:
            raise SystemExit("smoke arms did not use one identical checkpoint hash")
        if Counter(record["leaf_evaluator"] for record in records) != Counter(
            ["policy_rollout", "value", "value", "value"]
        ):
            raise SystemExit("smoke markers contain the wrong evaluator arms")
        payload.update({"smoke_checks": "passed", "smoke_records": 4})

    if args.write_gate:
        if not args.smoke_done:
            raise SystemExit("--write-gate requires --smoke-done")
        args.write_gate.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

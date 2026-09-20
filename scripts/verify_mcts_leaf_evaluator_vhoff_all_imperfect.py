#!/usr/bin/env python3
"""Verify the prepared all-domain VH-off rollout-versus-current screen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SEEDS = {"1963100312", "2011206605"}
DOMAINS = {
    "block_grouping", "drone", "fo_counters", "rover", "counters", "mprime"
}
EXPECTED_COMMIT = "2b243a6e8f21723f99cfd84136770ad32d220ffa"
EXPECTED = {
    "block_grouping": ("5", "20", "hadd-gbfs", "20"),
    "drone": ("20", "70", "hadd-astar", "20"),
    "fo_counters": ("20", "70", "hmrmax-astar", "20"),
    "rover": ("20", "70", "hmrp-ha-gbfs", "20"),
    "counters": ("5", "20", "hmrmax-astar", "59"),
    "mprime": ("20", "70", "hmrp-ha-gbfs", "20"),
}
ARMS = {
    "rollout": ("policy_rollout", 0.0),
    "current": ("value", 0.5),
}
SMOKE_IDS = {f"{domain}-off-1963100312-rollout" for domain in DOMAINS}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    if len(rows) != 24:
        raise ValueError(f"expected 24 rows, found {len(rows)}")
    ids = [row["manifest_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest_id values must be unique")

    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        domain = row["domain"]
        if domain not in DOMAINS or row["seed"] not in SEEDS:
            raise ValueError(f"unexpected domain/seed: {domain}/{row['seed']}")
        if row["value_head"] != "off" or row["stage"] != "stage1":
            raise ValueError(f"manifest must be Stage-1 VH-off: {row['manifest_id']}")
        if row["arm"] not in ARMS:
            raise ValueError(f"unexpected arm: {row['manifest_id']}")
        if not row["manifest_id"].endswith("-" + row["arm"]):
            raise ValueError(f"arm suffix mismatch: {row['manifest_id']}")
        evaluator, coefficient = ARMS[row["arm"]]
        if row["leaf_evaluator"] != evaluator:
            raise ValueError(f"wrong evaluator: {row['manifest_id']}")
        if float(row["use_estimator"]) != coefficient:
            raise ValueError(f"wrong estimator coefficient: {row['manifest_id']}")
        if row["code_commit"] != EXPECTED_COMMIT:
            raise ValueError(f"wrong code commit: {row['manifest_id']}")
        if row["status"] != "prepared_not_submitted":
            raise ValueError(f"unexpected status: {row['manifest_id']}")

        width, iterations, teacher, policy_total = EXPECTED[domain]
        expected_fields = {
            "rollout_horizon": "3", "width": width,
            "iterations": iterations, "teacher": teacher,
            "policy_total": policy_total, "puct": "0.1", "workers": "3",
            "cpus": "6", "memory_gib": "120",
            "time_limit": "3-00:00:00", "instance_timeout_seconds": "21600",
        }
        for field, expected in expected_fields.items():
            if row[field] != expected:
                raise ValueError(
                    f"{field} must be {expected} for {row['manifest_id']}"
                )
        selection = row["checkpoint_selection"]
        expected_selection = (
            "mprime_corrected_validation_v1"
            if domain == "mprime"
            else "main_val_validation_selected"
        )
        if selection != expected_selection:
            raise ValueError(f"wrong checkpoint selection: {row['manifest_id']}")
        cells[(domain, row["seed"])].append(row)

    expected_cells = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    if set(cells) != expected_cells:
        raise ValueError(f"unexpected matched cells: {sorted(cells)}")
    for cell, cell_rows in cells.items():
        if Counter(row["arm"] for row in cell_rows) != Counter(ARMS.keys()):
            raise ValueError(f"incomplete arm set for {cell}")
        frozen = (
            "source_checkpoint", "source_training_job_id", "source_policy_job_id",
            "snapshot_epoch", "policy_score", "policy_total", "width", "iterations",
            "puct", "teacher", "code_commit",
        )
        for field in frozen:
            if len({row[field] for row in cell_rows}) != 1:
                raise ValueError(f"{field} differs within matched cell {cell}")

    return {
        "manifest_rows": 24,
        "matched_cells": 12,
        "science_tasks": 24,
        "value_head": "off",
        "static_checks": "passed",
    }


def verify_smoke(rows: list[dict[str, str]], smoke_done: Path) -> dict[str, object]:
    by_id = {row["manifest_id"]: row for row in rows}
    records = []
    for manifest_id in sorted(SMOKE_IDS):
        marker = smoke_done / f"{manifest_id}.json"
        if not marker.is_file():
            raise ValueError(f"missing compatibility marker: {marker}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        row = by_id[manifest_id]
        required = {
            "manifest_id": manifest_id,
            "mode": "smoke",
            "value_head": "off",
            "leaf_evaluator": "policy_rollout",
            "code_commit": row["code_commit"],
            "width": int(row["width"]),
            "iterations": int(row["iterations"]),
        }
        for field, expected in required.items():
            if record.get(field) != expected:
                raise ValueError(f"smoke marker {marker} has wrong {field}")
        if not record.get("checkpoint_sha256"):
            raise ValueError(f"smoke marker {marker} lacks checkpoint hash")
        records.append(record)
    return {
        "compatibility_checks": "passed",
        "compatibility_records": len(records),
        "release_basis": "compatibility_only_not_performance",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--smoke-done", type=Path)
    parser.add_argument("--write-compatibility", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    payload = verify_manifest(rows)
    if args.smoke_done:
        payload.update(verify_smoke(rows, args.smoke_done))
    if args.write_compatibility:
        if not args.smoke_done:
            raise SystemExit("--write-compatibility requires --smoke-done")
        args.write_compatibility.parent.mkdir(parents=True, exist_ok=True)
        args.write_compatibility.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

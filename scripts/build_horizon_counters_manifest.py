#!/usr/bin/env python3
"""Build the eight-job fresh Counters horizon-aware/unaware pilot."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment_tracking/experiment_results.csv"
OUT = ROOT / "experiment_tracking/mcts_horizon_binding/counters_manifest.csv"
SEEDS = {"1963100312", "2011206605"}


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        source = [r for r in csv.DictReader(stream) if
                  r["experiment_id"] == "MAIN-VAL" and r["task_type"] == "policy_eval"
                  and r["domain"] == "counters" and r["stage"] == "stage1"
                  and r["endpoint"] == "validation_selected" and r["seed"] in SEEDS]
    if len(source) != 4:
        raise RuntimeError(f"expected four Counters sources, got {len(source)}")
    fields = ["manifest_id", "domain", "value_head", "seed", "arm",
              "source_training_job_id", "snapshot_epoch", "source_checkpoint",
              "iterations", "width", "action_limit", "status", "notes"]
    rows = []
    for src in source:
        for arm in ("unaware", "aware"):
            rows.append({
                "manifest_id": f"horizon-counters-{src['value_head']}-{src['seed']}-{arm}",
                "domain": "counters", "value_head": src["value_head"],
                "seed": src["seed"], "arm": arm,
                "source_training_job_id": src["source_training_job_id"],
                "snapshot_epoch": src["epoch"], "source_checkpoint": src["checkpoint"],
                "iterations": "70", "width": "20", "action_limit": "10000",
                "status": "ready", "notes": "Fresh same-commit efficacy pilot; PW disabled",
            })
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: r["manifest_id"]))
    print(f"rows={len(rows)} output={OUT}")


if __name__ == "__main__":
    main()

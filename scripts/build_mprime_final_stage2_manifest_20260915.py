#!/usr/bin/env python3
"""Freeze the clean validation-led MPrime Stage-2 submission manifest."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "mprime_phase_b_a_stage1_mcts_20260913"
OUT = ROOT / "experiment_tracking" / "mprime_final_stage2_validation_20260915"
ANCHOR = {"off": "30", "on": "10"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for mode in ("off", "on"):
        with (SOURCE / f"manifest_{mode}.csv").open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                rows.append({
                    "array_index": str(len(rows)),
                    "experiment_id": "MPRIME-FINAL-S2",
                    "branch": "validation_led",
                    "value_head": mode,
                    "seed": source["seed"],
                    "anchor": ANCHOR[mode],
                    "source_training_job_id": source["source_training_job_id"],
                    "source_epoch": source["selected_epoch"],
                    "source_checkpoint": source["checkpoint"],
                    "source_training_log": source["source_training_log"],
                    "selector_source": source["selector_source"],
                    "selector_sha256": source["selector_sha256"],
                    "architecture_module": "experiments_numeric.architecture_2.mprime_mcts",
                    "teacher": "hmrp-ha-gbfs",
                    "learning_rate": "0.0003",
                    "max_epochs": "100",
                    "mcts_expansion_size": "20",
                    "mcts_iterations": "0",
                    "puct": "0.1",
                    "estimator": "0.5",
                    "workers": "3",
                    "cpus": "6",
                    "memory": "48G",
                    "walltime": "3-00:00:00",
                    "submission_status": "ready",
                })
    assert len(rows) == 20
    assert len({(r["value_head"], r["seed"]) for r in rows}) == 20
    fields = list(rows[0])
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

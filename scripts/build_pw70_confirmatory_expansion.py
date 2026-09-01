#!/usr/bin/env python3
"""Build the five-seed PW70 expansion for six promising screen cells."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment_tracking/experiment_results.csv"
S2 = ROOT / (
    "experiment_tracking/mcts_counters_width_sensitivity/"
    "main_term_stage2_narrow_held_manifest.csv"
)
OUTPUT = ROOT / (
    "experiment_tracking/mcts_progressive_widening_cross_domain/"
    "pw70_confirmatory_expansion_manifest.csv"
)
NEW_SEEDS = {"534933607", "923500475", "1073581256"}
TEACHERS = {
    "fo_counters": "hmrmax-astar",
    "rover": "hmrp-ha-gbfs",
    "counters": "hmrmax-astar",
}
FIELDS = [
    "experiment_id", "manifest_id", "domain", "value_head", "seed", "stage",
    "mainstream_lineage", "source_checkpoint", "source_training_job_id",
    "snapshot_epoch", "policy_score", "matched_fixed_width", "iterations",
    "pw_min_width", "pw_c", "pw_alpha", "terminal_safe", "workers", "cpus",
    "memory", "time_limit", "instance_timeout", "teacher", "status", "notes",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def common(src: dict[str, str], *, domain: str, stage: str) -> dict[str, str]:
    return {
        "experiment_id": "MCTS-PW70-CONFIRMATORY",
        "manifest_id": f"pw70-confirm-{domain}-{src['value_head']}-{src['seed']}-{stage}",
        "domain": domain,
        "value_head": src["value_head"],
        "seed": src["seed"],
        "stage": stage,
        # The promoted Counters Stage-2 screen uses MAIN-VAL selected endpoints.
        # The historical source manifest filename contains "main_term", but its
        # source job IDs (204304xx) and experiment-results ledger are MAIN-VAL.
        "mainstream_lineage": "MAIN-VAL",
        "source_checkpoint": "",
        "source_training_job_id": "",
        "snapshot_epoch": "",
        "policy_score": "",
        "matched_fixed_width": "5" if domain == "counters" else "20",
        "iterations": "70",
        "pw_min_width": "3",
        "pw_c": "0.6",
        "pw_alpha": "0.5",
        "terminal_safe": "true",
        "workers": "3",
        "cpus": "6",
        "memory": "120G",
        "time_limit": "3-00:00:00",
        "instance_timeout": "21600",
        "teacher": TEACHERS[domain],
        "status": "ready-low-priority",
        "notes": (
            "Adds three held-out seeds to the two-seed screen for five matched seeds; "
            "submitted with lower scheduler priority than pre-existing ordinary pending work; "
            "PW70 is never pooled with PW20"
        ),
    }


def main() -> None:
    rows: list[dict[str, str]] = []
    result_rows = read(RESULTS)
    for src in result_rows:
        if not (
            src["experiment_id"] == "MAIN-VAL"
            and src["task_type"] == "policy_eval"
            and src["stage"] == "stage1"
            and src["endpoint"] == "validation_selected"
            and src["domain"] in {"fo_counters", "rover"}
            and src["seed"] in NEW_SEEDS
        ):
            continue
        row = common(src, domain=src["domain"], stage="stage1")
        row.update(
            source_checkpoint=src["checkpoint"],
            source_training_job_id=src["source_training_job_id"],
            snapshot_epoch=src["epoch"],
            policy_score=src["score"],
        )
        rows.append(row)

    for src in read(S2):
        if src["seed"] not in NEW_SEEDS:
            continue
        policy = next(
            row for row in result_rows
            if row["experiment_id"] == "MAIN-VAL"
            and row["task_type"] == "policy_eval"
            and row["stage"] == "stage2"
            and row["endpoint"] == "validation_selected"
            and row["domain"] == "counters"
            and row["value_head"] == src["value_head"]
            and row["seed"] == src["seed"]
            and row["source_training_job_id"] == src["source_training_job_id"]
        )
        row = common(src, domain="counters", stage="stage2")
        row.update(
            source_checkpoint=src["source_checkpoint_ref"],
            source_training_job_id=src["source_training_job_id"],
            snapshot_epoch=src["snapshot_epoch"],
            policy_score=policy["score"],
        )
        rows.append(row)

    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 rows (six cells x three new seeds), got {len(rows)}")
    keys = {(r["domain"], r["stage"], r["value_head"], r["seed"]) for r in rows}
    if len(keys) != len(rows):
        raise RuntimeError("Duplicate expansion rows detected")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["manifest_id"]))
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

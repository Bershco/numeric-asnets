#!/usr/bin/env python3
"""Build the frozen 50-lineage VH-off Stage-1 manifest for the KL screen."""

import csv
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "experiment_results.csv"
OUTDIR = ROOT / "experiment_tracking" / "imperfect_kl_primary_lineage_crossover_20260917"
DOMAIN_ORDER = {
    "block_grouping": 0,
    "drone": 1,
    "fo_counters": 2,
    "rover": 3,
    "counters": 4,
}
REUSED = {
    ("counters", "534933607"): 8,
    ("counters", "2082152039"): 9,
}


def normalize_checkpoint(path: str) -> str:
    return path[:-len("/weights.joblib")] if path.endswith("/weights.joblib") else path


with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
    rows = [
        row
        for row in csv.DictReader(handle)
        if row["experiment_id"] == "MAIN-VAL"
        and row["task_type"] == "policy_eval"
        and row["stage"] == "stage1"
        and row["endpoint"] == "validation_selected"
        and row["value_head"] == "off"
        and row["domain"] in DOMAIN_ORDER
    ]

assert len(rows) == 50, f"expected 50 primary rows, found {len(rows)}"
assert len({(row["domain"], row["seed"]) for row in rows}) == 50
rows.sort(key=lambda row: (DOMAIN_ORDER[row["domain"]], int(row["seed"])))

fieldnames = [
    "lineage_index",
    "new_pair_index",
    "domain",
    "seed",
    "total",
    "source_training_job_id",
    "source_policy_job_id",
    "starting_policy_score",
    "source_checkpoint",
    "source_checkpoint_epoch",
    "source_training_log",
    "source_policy_log",
    "optimizer_rng_base_seed",
    "reuse_status",
    "existing_pair_index",
]
new_index = 0
output = []
for lineage_index, row in enumerate(rows):
    key = (row["domain"], row["seed"])
    reuse = key in REUSED
    output.append(
        {
            "lineage_index": lineage_index,
            "new_pair_index": "" if reuse else new_index,
            "domain": row["domain"],
            "seed": row["seed"],
            "total": row["total"],
            "source_training_job_id": row["source_training_job_id"],
            "source_policy_job_id": row["job_id"],
            "starting_policy_score": row["score"],
            "source_checkpoint": normalize_checkpoint(row["checkpoint"]),
            "source_checkpoint_epoch": row["epoch"],
            "source_training_log": row["source_training_log"],
            "source_policy_log": row["source_evaluation_log"],
            "optimizer_rng_base_seed": 880000 + REUSED[key] if reuse else 890000 + lineage_index,
            "reuse_status": "reuse_verified_existing_pair" if reuse else "new_capture_and_pair",
            "existing_pair_index": REUSED.get(key, ""),
        }
    )
    if not reuse:
        new_index += 1

assert new_index == 48
OUTDIR.mkdir(parents=True, exist_ok=True)
for name, selected in (
    ("primary_lineages.csv", output),
    ("new_pairs.csv", [row for row in output if row["reuse_status"] == "new_capture_and_pair"]),
):
    with (OUTDIR / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

print(f"wrote 50 primary lineages and {new_index} new matched pairs to {OUTDIR}")

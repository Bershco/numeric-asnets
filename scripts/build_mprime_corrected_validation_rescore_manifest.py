#!/usr/bin/env python3
"""Build the 20-lineage MPrime checkpoint-rescoring manifest."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "policy_endpoint_results.csv"
OUTPUT = (
    ROOT / "experiment_tracking" / "mprime_validation_ipc_scale_v1"
    / "lineage_manifest.tsv"
)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["experiment_id"] == "PRESERVE-4"
        and row["domain"] == "mprime"
        and row["stage"] == "stage1"
        and row["endpoint"] == "validation_selected"
    ]
    manifest = []
    for row in sorted(selected, key=lambda r: (r["value_head"], int(r["seed"]))):
        checkpoint = PurePosixPath(row["checkpoint"])
        manifest.append({
            "manifest_id": f"mprime-{row['value_head']}-{row['seed']}-corrected-validation-v1",
            "seed": row["seed"],
            "value_head": row["value_head"],
            "source_training_job_id": row["source_training_job_id"],
            "snapshots_dir": checkpoint.parent.as_posix(),
            "source_training_log": row["source_training_log"],
            "validation_version": "mprime-validation-ipc-scale-v1",
            "status": "ready-after-planner-sanity",
        })
    signatures = {(r["value_head"], r["seed"]) for r in manifest}
    if len(manifest) != 20 or len(signatures) != 20:
        raise ValueError(
            f"Expected 20 unique MPrime lineages, got rows={len(manifest)} "
            f"signatures={len(signatures)}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(manifest[0]), delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest)
    print(f"rows={len(manifest)} output={OUTPUT}")


if __name__ == "__main__":
    main()

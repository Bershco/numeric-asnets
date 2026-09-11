"""Freeze the candidate-limited checkpoint inventory for MPrime Phase C."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
PHASE_B = TRACK / "mprime_validation_phase_b_20260906" / "phase_b_checkpoint_scores_latest.csv"
STAGE1 = TRACK / "mprime_validation_ipc_scale_v1" / "validation_test_checkpoint_audit.csv"
STAGE2 = TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage2_checkpoints_20260903.csv"
OUTPUT = TRACK / "mprime_validation_phase_c_20260911" / "checkpoint_candidates.csv"


def read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    phase_b = [
        row for row in read(PHASE_B)
        if row["branch"] in {"stage1", "validation_led_stage2"}
    ]
    grouped = defaultdict(list)
    for row in phase_b:
        grouped[row["lineage"]].append(row)
    assert len(grouped) == 40

    original_epoch = {}
    for row in read(STAGE1):
        if "selected" in row["roles"].split(";"):
            original_epoch[f"stage1-{row['value_head']}-{row['seed']}"] = int(row["epoch"])
    for row in read(STAGE2):
        if row["branch"] == "validation_led" and "validation_selected_policy" in row["roles"]:
            original_epoch[f"validation_led-{row['value_head']}-{row['seed']}"] = int(row["epoch"])
    assert len(original_epoch) == 40

    output = []
    for lineage, rows in sorted(grouped.items()):
        by_epoch = {int(row["epoch"]): row for row in rows}
        roles = defaultdict(set)
        for rank, row in enumerate(sorted(rows, key=lambda item: (-float(item["rep0_score"]), int(item["epoch"])))[:5], 1):
            roles[int(row["epoch"])].add(f"phase_b_a_top{rank}")
        for rank, row in enumerate(sorted(rows, key=lambda item: (-float(item["rep1_score"]), int(item["epoch"])))[:5], 1):
            roles[int(row["epoch"])].add(f"phase_b_b_top{rank}")
        roles[max(by_epoch)].add("final")
        roles[original_epoch[lineage]].add("original_validation_selected")
        assert len(roles) <= 12
        for epoch, selected_roles in sorted(roles.items()):
            row = by_epoch[epoch]
            output.append({
                "task": len(output),
                "lineage_task": sorted(grouped).index(lineage),
                "lineage": lineage,
                "branch": row["branch"],
                "value_head": row["value_head"],
                "seed": row["seed"],
                "epoch": epoch,
                "candidate_roles": ";".join(sorted(selected_roles)),
                "checkpoint": row["checkpoint"],
                "training_job": row["training_job"],
                "training_log": row["training_log"],
                "policy_job": row["policy_job"],
                "policy_log": row["policy_log"],
                "phase_b_rep0_score": row["rep0_score"],
                "phase_b_rep1_score": row["rep1_score"],
            })
    # Rescoring is one Slurm task per lineage, not one task per row.
    lineage_order = {lineage: index for index, lineage in enumerate(sorted(grouped))}
    for row in output:
        row["task"] = lineage_order[row["lineage"]]
        row["lineage_task"] = lineage_order[row["lineage"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    counts = defaultdict(int)
    for row in output:
        counts[row["lineage"]] += 1
    assert min(counts.values()) >= 6 and max(counts.values()) <= 12
    print(f"lineages={len(counts)} checkpoint_evaluations={len(output)} max_per_lineage={max(counts.values())}")


if __name__ == "__main__":
    main()

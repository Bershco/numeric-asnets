#!/usr/bin/env python3
"""Build the corrected-MPrime terminal-led Stage-2 manifest safely.

Without explicit frozen per-VH anchor coefficients, rows are emitted as blocked
planning records.  This prevents a guessed coefficient from accidentally
creating runnable training work.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE / "validation_test_checkpoint_audit.csv"
OUTPUT = HERE / "terminal_stage2_planned_manifest.csv"
FIELDS = [
    "manifest_id", "experiment_id", "domain", "value_head", "seed",
    "stage1_endpoint", "source_stage1_training_job", "source_checkpoint_ref",
    "source_epoch", "anchor", "status", "teacher", "policy_requirement",
    "mcts_requirement", "source_training_log",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-off")
    parser.add_argument("--anchor-on")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    anchors = {"off": args.anchor_off, "on": args.anchor_on}

    with AUDIT.open(newline="", encoding="utf-8") as stream:
        audit = list(csv.DictReader(stream))
    finals = [row for row in audit if "final" in row["roles"].split(";")]
    keys = {(row["value_head"], row["seed"]) for row in finals}
    if len(finals) != 20 or len(keys) != 20:
        raise RuntimeError(
            f"expected 20 unique corrected Stage-1 final endpoints, got "
            f"rows={len(finals)} keys={len(keys)}")

    rows = []
    for source in sorted(finals, key=lambda row: (row["value_head"], int(row["seed"]))):
        vh = source["value_head"]
        anchor = anchors[vh]
        rows.append({
            "manifest_id": f"mprime-terminal-s2-{vh}-{source['seed']}",
            "experiment_id": "MAIN-TERM-EXT6-MPRIME",
            "domain": "mprime", "value_head": vh, "seed": source["seed"],
            "stage1_endpoint": "final",
            "source_stage1_training_job": source["training_job"],
            "source_checkpoint_ref": source["checkpoint"],
            "source_epoch": source["epoch"],
            "anchor": anchor or f"PENDING_FREEZE_{vh.upper()}",
            "status": "ready" if anchor is not None else "blocked_anchor_freeze",
            "teacher": "hmrp-ha-gbfs",
            "policy_requirement": "every-five;stage2_validation_selected;stage2_final",
            "mcts_requirement": "stage2_validation_selected_only",
            "source_training_log": source["training_log"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(
        f"rows={len(rows)} ready={sum(row['status'] == 'ready' for row in rows)} "
        f"blocked={sum(row['status'] != 'ready' for row in rows)} output={args.output}")


if __name__ == "__main__":
    main()

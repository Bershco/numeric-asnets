#!/usr/bin/env python3
"""Freeze the Phase-B-A-selected endpoint for every final MPrime S2 lineage."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EPOCH = re.compile(r"epoch_(\d+)_phase_b_a\.val\.csv$")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def score(path: Path) -> int:
    rows = read(path)
    if len(rows) > 30:
        raise RuntimeError(f"too many VAL rows in {path}")
    return sum(int(row["val_valid"]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--rescore-root", type=Path, required=True)
    parser.add_argument("--all-scores", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    args = parser.parse_args()
    manifest, ready = read(args.manifest), read(args.ready_manifest)
    if len(manifest) != 20 or len(ready) != 420:
        raise RuntimeError("incomplete manifest inputs")
    by_identity = {(row["value_head"], row["seed"], row["snapshot_epoch"]): row for row in ready}
    all_rows, selected_rows = [], []
    for lineage in manifest:
        summaries = sorted((args.rescore_root / lineage["manifest_id"]).glob("epoch_*_phase_b_a.val.csv"))
        if len(summaries) != 21:
            raise RuntimeError(f"{lineage['manifest_id']}: expected 21 summaries, got {len(summaries)}")
        curve = []
        for summary in summaries:
            match = EPOCH.search(summary.name)
            if not match:
                continue
            epoch, value = int(match.group(1)), score(summary)
            curve.append((epoch, value, summary))
            all_rows.append({
                "manifest_id": lineage["manifest_id"], "value_head": lineage["value_head"],
                "seed": lineage["seed"], "epoch": epoch, "phase_b_a_score": value,
                "validation_summary": str(summary),
            })
        if sorted(epoch for epoch, _, _ in curve) != list(range(0, 100, 5)) + [99]:
            raise RuntimeError(f"{lineage['manifest_id']}: unexpected epoch set")
        epoch, value, summary = min(curve, key=lambda item: (-item[1], item[0]))
        source = by_identity[(lineage["value_head"], lineage["seed"], str(epoch))]
        selected_rows.append({
            **source,
            "checkpoint_selection": "phase_b_replicate_a",
            "selected_validation_score": str(value),
            "phase_b_a_validation_summary": str(summary),
        })
    for path, rows in ((args.all_scores, all_rows), (args.selected, selected_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    print("selected=" + ",".join(f"{r['value_head']}/{r['seed']}:e{r['snapshot_epoch']}={r['selected_validation_score']}" for r in selected_rows))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the Phase-B-A-selected endpoint for every final MPrime S2 lineage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


EPOCH = re.compile(r"epoch_(\d+)_phase_b_a\.val\.csv$")
FINAL_SCORE = re.compile(r"\[EVAL FINAL\].*?success=(\d+)(?:\.0+)?/30(?:\.0+)?")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def score(path: Path) -> int:
    rows = read(path)
    log = path.with_name(path.name.replace(".val.csv", ".log"))
    scores = FINAL_SCORE.findall(log.read_text(errors="replace"))
    if len(scores) != 1:
        raise RuntimeError(f"expected one terminal 30-instance score in {log}")
    value = int(scores[0])
    if len(rows) != value or any(row.get("val_valid") != "1" for row in rows):
        raise RuntimeError(f"VAL rows do not match terminal score {value}/30 in {path}")
    return value


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
            source = by_identity[(lineage["value_head"], lineage["seed"], str(epoch))]
            done = summary.with_name(summary.name.replace(".val.csv", ".done.json"))
            expected_identity = {
                "checkpoint": source["source_checkpoint_ref"],
                "checkpoint_sha256": source["source_checkpoint_sha256"],
                "training_job_id": source["source_training_job_id"],
                "validation_manifest_sha256": lineage["validation_manifest_sha256"],
                "validation_module_sha256": lineage["validation_module_sha256"],
                "code_commit": "ef274cd23605de815800656c97b360ee7ff584b8",
            }
            if not done.is_file() or json.loads(done.read_text(encoding="utf-8")) != expected_identity:
                raise RuntimeError(f"{lineage['manifest_id']} epoch {epoch}: missing/mismatched done identity")
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

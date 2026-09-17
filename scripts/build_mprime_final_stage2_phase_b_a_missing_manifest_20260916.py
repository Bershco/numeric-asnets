#!/usr/bin/env python3
"""Freeze one row per incomplete MPrime Phase-B-A checkpoint evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


FINAL_SCORE = re.compile(r"\[EVAL FINAL\].*?success=(\d+)(?:\.0+)?/30(?:\.0+)?")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(done: Path, summary: Path, identity: dict[str, str]) -> bool:
    if not done.is_file() or not summary.is_file() or summary.stat().st_size == 0:
        return False
    try:
        rows = read(summary)
        log = summary.with_name(summary.name.replace(".val.csv", ".log"))
        scores = FINAL_SCORE.findall(log.read_text(errors="replace"))
        return (
            json.loads(done.read_text(encoding="utf-8")) == identity
            and len(scores) == 1
            and len(rows) == int(scores[0])
            and all(row.get("val_valid") == "1" for row in rows)
        )
    except (OSError, json.JSONDecodeError, csv.Error):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--rescore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    lineages, ready = read(args.manifest), read(args.ready_manifest)
    if len(lineages) != 20 or len(ready) != 420:
        raise RuntimeError("expected 20 lineages and 420 ready checkpoints")
    by_lineage = {(row["value_head"], row["seed"]): row for row in lineages}
    expected_ready_hashes = {row["ready_manifest_sha256"] for row in lineages}
    if expected_ready_hashes != {sha(args.ready_manifest)}:
        raise RuntimeError("ready-manifest checksum does not match lineage manifest")

    missing: list[dict[str, str]] = []
    valid = 0
    for item in sorted(ready, key=lambda row: (row["value_head"], int(row["seed"]), int(row["snapshot_epoch"]))):
        lineage = by_lineage[(item["value_head"], item["seed"])]
        epoch = int(item["snapshot_epoch"])
        stem = args.rescore_root / lineage["manifest_id"] / f"epoch_{epoch}_phase_b_a"
        identity = {
            "checkpoint": item["source_checkpoint_ref"],
            "checkpoint_sha256": item["source_checkpoint_sha256"],
            "training_job_id": item["source_training_job_id"],
            "validation_manifest_sha256": lineage["validation_manifest_sha256"],
            "validation_module_sha256": lineage["validation_module_sha256"],
            "code_commit": args.code_commit,
        }
        if complete(stem.with_suffix(".done.json"), stem.with_suffix(".val.csv"), identity):
            valid += 1
            continue
        missing.append({
            "array_index": str(len(missing)),
            "manifest_id": lineage["manifest_id"],
            "lineage_task": lineage["array_index"],
            "value_head": lineage["value_head"],
            "seed": lineage["seed"],
            "snapshot_epoch": str(epoch),
            "source_checkpoint_ref": item["source_checkpoint_ref"],
            "source_checkpoint_sha256": item["source_checkpoint_sha256"],
            "source_training_job_id": item["source_training_job_id"],
            "validation_module": lineage["validation_module"],
            "validation_manifest_sha256": lineage["validation_manifest_sha256"],
            "validation_module_sha256": lineage["validation_module_sha256"],
            "ready_manifest_sha256": lineage["ready_manifest_sha256"],
            "code_commit": args.code_commit,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "array_index", "manifest_id", "lineage_task", "value_head", "seed",
        "snapshot_epoch", "source_checkpoint_ref", "source_checkpoint_sha256",
        "source_training_job_id", "validation_module", "validation_manifest_sha256",
        "validation_module_sha256", "ready_manifest_sha256", "code_commit",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(missing)
    audit = {
        "total_expected": len(ready),
        "identity_valid_complete": valid,
        "missing": len(missing),
        "missing_manifest": str(args.output),
        "missing_manifest_sha256": sha(args.output),
        "completion_definition": "nonempty summary plus exact matching done identity",
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

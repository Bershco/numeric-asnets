#!/usr/bin/env python3
"""Freeze the twenty-lineage Phase-B-A rescore for final MPrime Stage 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--validator-manifest", type=Path, required=True)
    parser.add_argument("--validator-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ready = read(args.ready_manifest)
    if len(ready) != 420:
        raise RuntimeError(f"expected 420 ready checkpoints, found {len(ready)}")
    validator = read(args.validator_manifest)
    rep_a = [row for row in validator if row["replicate"] == "0"]
    if len(validator) != 60 or len(rep_a) != 30:
        raise RuntimeError("frozen Phase-B manifest must contain 60 rows and 30 replicate-A rows")
    if len({row["sha256"] for row in rep_a}) != 30:
        raise RuntimeError("replicate A contains duplicate PDDL hashes")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in ready:
        grouped.setdefault((row["value_head"], row["seed"]), []).append(row)
    if len(grouped) != 20:
        raise RuntimeError(f"expected twenty lineages, found {len(grouped)}")

    rows = []
    for index, ((mode, seed), checkpoints) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1])))
    ):
        checkpoints.sort(key=lambda row: int(row["snapshot_epoch"]))
        if len(checkpoints) != 21:
            raise RuntimeError(f"{mode}/{seed}: expected 21 checkpoints, found {len(checkpoints)}")
        epochs = [int(row["snapshot_epoch"]) for row in checkpoints]
        if epochs != list(range(0, 100, 5)) + [99]:
            raise RuntimeError(f"{mode}/{seed}: unexpected checkpoint epochs {epochs}")
        first = checkpoints[0]
        invariant = (
            "source_training_job_id", "training_log", "teacher",
        )
        for field in invariant:
            if len({row[field] for row in checkpoints}) != 1:
                raise RuntimeError(f"{mode}/{seed}: {field} changes within lineage")
        rows.append({
            "array_index": str(index),
            "manifest_id": f"mprime-final-s2-{mode}-{seed}-phase-b-a-rescore",
            "value_head": mode,
            "seed": seed,
            "source_training_job_id": first["source_training_job_id"],
            "source_training_log": first["training_log"],
            "expected_checkpoint_count": "21",
            "validation_module": "mprime_phase_b_20260906_0",
            "validation_replicate": "phase_b_a",
            "validation_instances": "30",
            "validation_manifest_sha256": digest(args.validator_manifest),
            "validation_module_sha256": digest(args.validator_module),
            "ready_manifest": str(args.ready_manifest).replace("\\", "/"),
            "ready_manifest_sha256": digest(args.ready_manifest),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} lineages / {len(ready)} checkpoint evaluations")


if __name__ == "__main__":
    main()

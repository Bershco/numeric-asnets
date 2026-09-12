#!/usr/bin/env python3
"""Freeze the 28-lineage MPrime Phase-B-A anchor-rescore manifest.

This joins the original tuning-job ledger to the source-log inventory used by
the completed IPC-scale rescore.  It performs no cluster work and never reads
test-policy scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1"
    / "anchor_selection_invalid.csv"
)
DEFAULT_TUNING = (
    ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1"
    / "anchor_tuning_submissions.tsv"
)
DEFAULT_VALIDATOR = (
    ROOT / "experiment_tracking/mprime_validation_phase_b_20260906"
    / "frozen_validation_manifest.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "experiment_tracking/mprime_anchor_phase_b_a_20260912"
    / "manifest.csv"
)
DEFAULT_MODULE = (
    ROOT / "asnets/experiments_numeric/domain/mprime_phase_b_20260906_0.py"
)

ANCHORS = ("0", "0.03", "0.3", "1", "3", "10", "30")
SEEDS = ("1963100312", "2011206605")
VALUE_HEADS = ("off", "on")
VALIDATION_MODULE = "mprime_phase_b_20260906_0"
REMOTE_OUTPUT_ROOT = (
    "/home/hersco/training_new_domains/2026-09-12/"
    "mprime_anchor_phase_b_a/rescore"
)


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_validator_manifest(path: Path) -> str:
    rows = read_csv(path)
    if len(rows) != 60:
        raise ValueError(f"expected 60 frozen Phase-B rows, got {len(rows)}")
    replicate_a = [row for row in rows if row["replicate"] == "0"]
    if len(replicate_a) != 30:
        raise ValueError(f"expected 30 replicate-A rows, got {len(replicate_a)}")
    tier_counts = {
        tier: sum(row["tier"] == tier for row in replicate_a)
        for tier in ("0", "1", "2")
    }
    if tier_counts != {"0": 10, "1": 10, "2": 10}:
        raise ValueError(f"unexpected replicate-A tier counts: {tier_counts}")
    if len({row["sha256"] for row in replicate_a}) != 30:
        raise ValueError("replicate-A validator contains duplicate PDDL hashes")
    return sha256(path)


def build_rows(
    source_path: Path,
    tuning_path: Path,
    validator_path: Path,
    module_path: Path = DEFAULT_MODULE,
) -> list[dict[str, str]]:
    source_rows = read_csv(source_path)
    tuning_rows = read_csv(tuning_path, "\t")
    if len(source_rows) != 28 or len(tuning_rows) != 28:
        raise ValueError(
            f"expected 28 source/tuning rows, got {len(source_rows)}/{len(tuning_rows)}"
        )
    source_by_id = {row["manifest_id"]: row for row in source_rows}
    tuning_by_id = {row["manifest_id"]: row for row in tuning_rows}
    if len(source_by_id) != 28 or source_by_id.keys() != tuning_by_id.keys():
        raise ValueError("source and tuning manifests do not contain the same 28 identities")

    validator_sha = validate_validator_manifest(validator_path)
    if not module_path.is_file():
        raise ValueError(f"missing frozen validator module: {module_path}")
    module_sha = sha256(module_path)
    rows: list[dict[str, str]] = []
    expected = {
        (vh, seed, anchor)
        for vh in VALUE_HEADS for seed in SEEDS for anchor in ANCHORS
    }
    observed = set()
    for index, manifest_id in enumerate(row["manifest_id"] for row in source_rows):
        source = source_by_id[manifest_id]
        tuning = tuning_by_id[manifest_id]
        identity = (source["value_head"], source["seed"], source["anchor"])
        observed.add(identity)
        if tuning["slurm_job_id"] != source["job_id"]:
            raise ValueError(
                f"{manifest_id}: tuning job {tuning['slurm_job_id']} != "
                f"source job {source['job_id']}"
            )
        if any(tuning[key] != source[key] for key in ("value_head", "seed", "anchor")):
            raise ValueError(f"{manifest_id}: identity disagreement between ledgers")
        rows.append({
            "array_index": str(index),
            "manifest_id": manifest_id,
            "training_job_id": source["job_id"],
            "value_head": source["value_head"],
            "seed": source["seed"],
            "anchor": source["anchor"],
            "source_training_log": source["source_log"],
            "expected_checkpoint_count": "21",
            "expected_validation_points": "630",
            "validation_module": VALIDATION_MODULE,
            "validation_replicate": "phase_b_a",
            "validation_instances": "30",
            "validation_manifest_sha256": validator_sha,
            "validation_module_sha256": module_sha,
            "remote_output_dir": f"{REMOTE_OUTPUT_ROOT}/{manifest_id}",
            "source_provenance": (
                "experiment_tracking/mprime_validation_ipc_scale_v1/"
                "anchor_tuning_submissions.tsv;"
                "experiment_tracking/mprime_validation_ipc_scale_v1/"
                "anchor_selection_invalid.csv"
            ),
        })
    if observed != expected:
        raise ValueError(f"anchor-grid mismatch: missing={expected-observed} extra={observed-expected}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tuning", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    args = parser.parse_args()
    rows = build_rows(args.source, args.tuning, args.validator, args.module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} points={len(rows)*21} output={args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one exact unclassified MPrime Stage-2 seed-instance identity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_recovery_record(
    *, temporary: Path, expected_number: int, expected_path: str,
) -> None:
    recovered = read_jsonl(temporary)
    if len(recovered) != 1:
        raise RuntimeError(
            f"exact recovery must produce one record, found {len(recovered)}: {temporary}"
        )
    record = recovered[0]
    if int(record["instance_number"]) != expected_number:
        raise RuntimeError(
            f"recovery instance {record['instance_number']} != {expected_number}"
        )
    if record["instance_path"] != expected_path:
        raise RuntimeError(
            f"recovery path {record['instance_path']!r} != {expected_path!r}"
        )
    if not record.get("evaluation_signature"):
        raise RuntimeError(f"recovery record lacks its own signature: {temporary}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--recovery-index", type=int, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    recovery = read_csv(args.recovery_manifest)
    matches = [
        row for row in recovery
        if row["recovery_index"] == str(args.recovery_index)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"recovery index {args.recovery_index}: expected one row, found {len(matches)}"
        )
    row = matches[0]
    manifest = read_csv(args.manifest)
    source = [
        item for item in manifest
        if item["array_index"] == row["array_index"]
    ]
    if len(source) != 1:
        raise ValueError(
            f"source array index {row['array_index']}: expected one row, found {len(source)}"
        )
    source = source[0]
    for field in (
        "manifest_id", "search_method", "value_head", "seed",
        "selected_epoch", "checkpoint_sha256",
    ):
        if row[field] != source[field]:
            raise ValueError(
                f"recovery/source mismatch {field}: {row[field]!r} != {source[field]!r}"
            )
    instance = int(row["instance_number"])
    if not 1 <= instance <= 20:
        raise ValueError(f"invalid recovery instance: {instance}")

    recovery_job_id = os.environ.get("SLURM_JOB_ID", "local")
    temporary = (
        args.campaign_root / source["search_method"] / source["value_head"] /
        source["seed"] / "recovery_completion" /
        f"instance_{instance}_{recovery_job_id}.jsonl"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()

    command = [
        sys.executable,
        str(args.checkout / "scripts" / "run_mprime_final_stage2_search_20260916.py"),
        "--manifest", str(args.manifest),
        "--index", row["array_index"],
        "--checkout", str(args.checkout),
        "--production", str(args.production),
        "--container", str(args.container),
        "--validator", str(args.validator),
        "--output-root", str(args.campaign_root),
        "--code-commit", args.code_commit,
        "--only-instance-number", str(instance),
        "--completion-file-override", str(temporary),
    ]
    print(
        "[MPRIME FINAL S2 EXACT RECOVERY] "
        f"recovery_index={args.recovery_index} manifest_id={row['manifest_id']} "
        f"instance={instance} path={row['instance_path']}"
    )
    returncode = subprocess.run(command).returncode
    if returncode:
        return returncode
    verify_recovery_record(
        temporary=temporary, expected_number=instance,
        expected_path=row["instance_path"],
    )
    print(
        "[MPRIME FINAL S2 EXACT RECOVERY] "
        f"instance={instance} disposition=verified_separate_ledger "
        f"completion={temporary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

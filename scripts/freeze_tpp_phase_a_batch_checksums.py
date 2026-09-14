#!/usr/bin/env python3
"""Freeze or verify the 120 Phase-A replay-batch files used by Phase B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


PHASE_A = Path(
    "/home/hersco/training_new_domains/2026-09-14/"
    "tpp_first_update_phase_a")
SOURCES = {
    "catastrophic_outlier": (
        PHASE_A / "catastrophic_outlier_1972442430"
        / "first_update_audit.jsonl.batches"),
    "stable_control": (
        PHASE_A / "stable_control_2082152039"
        / "first_update_audit.jsonl.batches"),
}
DEFAULT_MANIFEST = Path(
    "/home/hersco/training_new_domains/2026-09-14/"
    "tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows() -> list[dict[str, str]]:
    rows = []
    for role, directory in sorted(SOURCES.items()):
        files = sorted(directory.glob("optimizer_step_*.npz"))
        expected = [
            directory / f"optimizer_step_{step:03d}.npz"
            for step in range(60)
        ]
        if files != expected:
            raise RuntimeError(
                f"{role} must contain exactly the contiguous 60-step "
                f"schedule; found {len(files)} files")
        raw_rows = []
        for step, path in enumerate(files):
            raw_rows.append({
                "role": role,
                "step": str(step),
                "path": str(path.resolve()),
                "bytes": str(path.stat().st_size),
                "sha256": file_sha256(path),
            })
        encoded = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in raw_rows
        ).encode("utf-8")
        aggregate = hashlib.sha256(encoded).hexdigest()
        for row in raw_rows:
            row["schedule_sha256"] = aggregate
        rows.extend(raw_rows)
    return rows


def freeze(manifest: Path) -> None:
    if manifest.exists():
        raise RuntimeError(
            f"Refusing to overwrite frozen checksum manifest: {manifest}")
    rows = source_rows()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, manifest)
    print(
        f"froze {len(rows)} batch files in {manifest}; "
        f"aggregates={sorted({row['schedule_sha256'] for row in rows})}")


def verify(manifest: Path) -> None:
    with manifest.open(newline="", encoding="utf-8") as stream:
        frozen = list(csv.DictReader(stream))
    current = source_rows()
    if frozen != current:
        raise RuntimeError(
            "Frozen replay schedules no longer match the checksum manifest")
    print(
        f"verified {len(current)} frozen batch files; "
        f"aggregates={sorted({row['schedule_sha256'] for row in current})}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.manifest)
    else:
        verify(args.manifest)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Freeze exact checkpoint continuations for the interrupted KL pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


FIELDS = (
    "arm_index", "domain", "seed", "semantics", "training_module", "teacher",
    "source_checkpoint", "source_checkpoint_sha256", "start_epoch",
    "remaining_epochs", "segment",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--segment", default="segment_001")
    parser.add_argument("--output", default="recovery_manifest.csv")
    args = parser.parse_args()
    manifest = list(csv.DictReader((args.campaign / "manifest.csv").open()))
    rows = []
    for arm in manifest:
        output = args.campaign / "outputs" / (
            f"arm_{arm['arm_index']}_{arm['domain']}_{arm['seed']}_{arm['semantics']}"
        )
        if (output / "training_complete.json").exists():
            continue
        text = (output / "training.stdout").read_text(errors="replace")
        matches = re.findall(r"^Snapshot directory:\s*(.+?)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"arm {arm['arm_index']}: expected one original snapshot root")
        root = Path(matches[0])
        checkpoints = []
        for checkpoint in root.glob("snapshot_*_"):
            # Kept for compatibility with an empty suffix; normal names are
            # collected by the broader pass below.
            checkpoints.append(checkpoint)
        checkpoints = list(root.glob("snapshot_*"))
        parsed = []
        for checkpoint in checkpoints:
            try:
                epoch = int(checkpoint.name.split("_", 2)[1])
            except (IndexError, ValueError):
                continue
            required = ("weights.joblib", "optimizer.joblib", "trainer_state.joblib")
            if all((checkpoint / name).is_file() for name in required):
                parsed.append((epoch, checkpoint))
        if not parsed:
            raise RuntimeError(f"arm {arm['arm_index']}: no complete checkpoint")
        latest_epoch, latest = max(parsed)
        start = latest_epoch + 1
        if not 0 < start < 100:
            raise RuntimeError(f"arm {arm['arm_index']}: invalid recovery start {start}")
        rows.append({
            "arm_index": arm["arm_index"], "domain": arm["domain"],
            "seed": arm["seed"], "semantics": arm["semantics"],
            "training_module": arm["training_module"], "teacher": arm["teacher"],
            "source_checkpoint": str(latest),
            "source_checkpoint_sha256": sha256(latest / "weights.joblib"),
            "start_epoch": start, "remaining_epochs": 100 - start,
            "segment": args.segment,
        })
    if len(rows) != 8:
        raise RuntimeError(f"expected eight interrupted arms, found {len(rows)}")
    target = args.campaign / args.output
    if target.exists():
        old = list(csv.DictReader(target.open()))
        if old != rows:
            raise RuntimeError("existing recovery manifest differs")
    else:
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(
            f"arm={row['arm_index']} start={row['start_epoch']} "
            f"remaining={row['remaining_epochs']} source={row['source_checkpoint']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze source checkpoint and replay-file hashes before crossover submission."""

import csv
import hashlib
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
destination = root / "frozen_input_checksums"
destination.mkdir(parents=True, exist_ok=True)
summary = []

with manifest_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        pair = int(row["pair_index"])
        checkpoint = pathlib.Path(row["source_checkpoint"])
        checkpoint_file = checkpoint / "weights.joblib" if checkpoint.is_dir() else checkpoint
        replay_dir = pathlib.Path(row["frozen_replay_directory"])
        replay_files = sorted(replay_dir.glob("optimizer_step_*.npz"))
        if len(replay_files) != 60:
            raise RuntimeError(f"pair {pair}: expected 60 replay files, found {len(replay_files)}")
        inputs = [checkpoint_file, *replay_files]
        checksum_file = destination / f"pair_{pair}.sha256"
        with checksum_file.open("w", newline="\n") as output:
            for path in inputs:
                if not path.is_file():
                    raise FileNotFoundError(path)
                output.write(f"{sha256(path)}  {path}\n")
        summary.append(
            {
                "pair_index": pair,
                "domain": row["domain"],
                "seed": row["seed"],
                "checkpoint_file": str(checkpoint_file),
                "checkpoint_sha256": sha256(checkpoint_file),
                "replay_file_count": len(replay_files),
                "checksum_manifest": str(checksum_file),
            }
        )

with (root / "frozen_input_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
print(f"frozen {len(summary)} pairs and {len(summary) * 60} replay files")

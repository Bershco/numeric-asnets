#!/usr/bin/env python3
"""Build exact retries for terminal-led Delivery policy rows lacking a score."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiment_tracking" / "preserve3_terminal_led"
OUT = BASE / "delivery_policy_retry_manifest_20260903.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    retry_ids: set[str] = set()
    for vh in ("off", "on"):
        for row in read(BASE / f"preserve3_terminal_delivery_{vh}_results.csv"):
            if not row["score"]:
                retry_ids.add(row["manifest_id"])

    ready: dict[str, dict[str, str]] = {}
    for vh in ("off", "on"):
        for row in read(BASE / f"policy_ready_delivery_{vh}.csv"):
            ready[row["manifest_id"]] = row

    missing = sorted(retry_ids - ready.keys())
    if missing:
        raise RuntimeError(f"missing retry identities in source manifests: {missing}")
    rows = [ready[manifest_id] for manifest_id in sorted(retry_ids)]
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 scoreless Delivery rows, got {len(rows)}")
    selected = [row for row in rows if "validation_selected_policy" in row["analysis_roles"]]
    if len(selected) != 2:
        raise RuntimeError(f"expected two selected-endpoint retries, got {len(selected)}")

    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} selected_endpoints={len(selected)} output={OUT}")


if __name__ == "__main__":
    main()

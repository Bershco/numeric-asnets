#!/usr/bin/env python3
"""Summarize physical-state revisits versus full context-key reuse."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "experiment_tracking" / "mcts_safe_context"
    / "corrected_diagnostic_instance_results.csv"
)
OUTPUT = (
    ROOT / "experiment_tracking" / "mcts_safe_context"
    / "context_revisit_summary.csv"
)
FIELDS = [
    "scope", "domain", "instance", "observations", "physical_revisits",
    "different_context_revisits", "same_context_reuses",
    "different_context_share_of_physical_revisits", "node_multiplier",
    "source_results",
]


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        source = [row for row in csv.DictReader(stream) if row["physical_revisits"]]
    rows = []
    for row in source:
        physical = int(row["physical_revisits"])
        mismatch = int(row["context_mismatches"])
        rows.append({
            "scope": "instance",
            "domain": row["domain"],
            "instance": row["instance"],
            "observations": row["observations"],
            "physical_revisits": physical,
            "different_context_revisits": mismatch,
            "same_context_reuses": physical - mismatch,
            "different_context_share_of_physical_revisits": (
                f"{mismatch / physical:.6f}" if physical else "0.000000"
            ),
            "node_multiplier": row["node_multiplier"],
            "source_results": SOURCE.relative_to(ROOT).as_posix(),
        })
    physical = sum(int(row["physical_revisits"]) for row in source)
    mismatch = sum(int(row["context_mismatches"]) for row in source)
    rows.append({
        "scope": "aggregate",
        "domain": "all",
        "instance": "all measured instances",
        "observations": sum(int(row["observations"]) for row in source),
        "physical_revisits": physical,
        "different_context_revisits": mismatch,
        "same_context_reuses": physical - mismatch,
        "different_context_share_of_physical_revisits": f"{mismatch / physical:.6f}",
        "node_multiplier": "",
        "source_results": SOURCE.relative_to(ROOT).as_posix(),
    })
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

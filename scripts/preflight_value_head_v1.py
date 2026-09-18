#!/usr/bin/env python3
"""Strict local preflight for the V1 value-head audit candidate manifest."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "asnets"))

from asnets.value_head_audit_manifest import (  # noqa: E402
    validate_label_source_rows,
    validate_state_mixture_rows,
    validate_task_manifest_rows,
)


def main() -> int:
    base = ROOT / "experiment_tracking" / "value_head_quality_audit"
    def read(name: str) -> list[dict[str, str]]:
        with (base / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    domains = ("drone", "fo_counters", "rover", "mprime")
    tasks = read("v1_checkpoint_candidates.csv")
    errors = validate_task_manifest_rows(
        tasks,
        domains=domains,
        seeds=("534933607", "923500475"),
    )
    errors.extend(validate_state_mixture_rows(read("v1_state_mixture.csv"), domains=domains))
    errors.extend(validate_label_source_rows(read("v1_label_sources.csv")))
    if errors:
        print(f"BLOCKED: {len(errors)} submission preflight issue(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"READY: {len(tasks)} V1 checkpoint tasks passed strict preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

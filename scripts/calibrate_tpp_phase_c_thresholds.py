#!/usr/bin/env python3
"""Freeze Phase-C limits from the same deterministic KL statistic it guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path



def linear_quantile(values: list[float], q: float) -> float:
    """Match NumPy's default linear quantile without a host-side dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.audit_jsonl.read_text().splitlines() if line.strip()]
    steps = [row for row in rows if row.get("record_type") == "optimizer_step"]
    if len(steps) != 60:
        raise SystemExit(f"Expected 60 calibration steps, found {len(steps)}")
    if not all(row.get("kl_current_forward") == "deterministic" for row in steps):
        raise SystemExit("Calibration audit did not use deterministic current-policy KL")
    payload = {
        "calibration_role": "stable_control",
        "calibration_steps": 60,
        "statistic": "empirical 99th percentile across stable-control optimizer steps",
        "mean_kl_limit": linear_quantile([row["step_kl_mean"] for row in steps], 0.99),
        "p99_kl_limit": linear_quantile([row["step_kl_p99"] for row in steps], 0.99),
        "source_audit": str(args.audit_jsonl),
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

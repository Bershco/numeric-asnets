#!/usr/bin/env python3
"""Verify the frozen Phase-C limits from the 60 stable Phase-A steps."""

from __future__ import annotations

import argparse
import json
from math import floor, isclose
from pathlib import Path


EXPECTED_MEAN_LIMIT = 0.2123709661
EXPECTED_P99_LIMIT = 1.5138580917


def linear_quantile(values: list[float], probability: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("cannot calculate a quantile from no values")
    position = (len(values) - 1) * probability
    lower = floor(position)
    fraction = position - lower
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + fraction * (values[upper] - values[lower])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_jsonl", type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.audit_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [
        record for record in records
        if record.get("record_type") == "optimizer_step"
    ]
    if len(steps) != 60 or [row.get("step") for row in steps] != list(range(60)):
        raise SystemExit("Stable Phase-A audit must contain ordered steps 0..59")
    mean_limit = linear_quantile(
        [row["step_kl_mean"] for row in steps], 0.99)
    p99_limit = linear_quantile(
        [row["step_kl_p99"] for row in steps], 0.99)
    if not isclose(mean_limit, EXPECTED_MEAN_LIMIT, abs_tol=5e-11):
        raise SystemExit(
            f"Mean-KL threshold mismatch: {mean_limit:.10f} != "
            f"{EXPECTED_MEAN_LIMIT:.10f}")
    if not isclose(p99_limit, EXPECTED_P99_LIMIT, abs_tol=5e-11):
        raise SystemExit(
            f"p99-KL threshold mismatch: {p99_limit:.10f} != "
            f"{EXPECTED_P99_LIMIT:.10f}")
    print(json.dumps({
        "source": str(args.audit_jsonl.resolve()),
        "steps": len(steps),
        "stable_empirical_q99_step_kl_mean": round(mean_limit, 10),
        "stable_empirical_q99_step_kl_p99": round(p99_limit, 10),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

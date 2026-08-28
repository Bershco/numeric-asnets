#!/usr/bin/env python3
"""Build the two-seed, one-factor-at-a-time SAFE+PW sensitivity gate."""

from __future__ import annotations

import csv
from pathlib import Path


SOURCE = Path("experiment_tracking/mcts_progressive_widening_pilot/manifest.csv")
OUTPUT = Path("experiment_tracking/mcts_progressive_widening_sensitivity/manifest.csv")
SEEDS = (1963100312, 2011206605)
VARIANTS = (
    ("safe_pw_baseline", 2, 0.6, 0.5, 70),
    ("safe_pw_kmin3", 3, 0.6, 0.5, 70),
    ("safe_pw_alpha0p65", 2, 0.6, 0.65, 70),
    ("safe_pw_iter140", 2, 0.6, 0.5, 140),
)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sources = {}
    for row in rows:
        key = (row["value_head"], int(row["seed"]))
        sources.setdefault(key, row)

    output = []
    for value_head in ("off", "on"):
        for seed in SEEDS:
            source = sources[(value_head, seed)]
            for variant, k_min, c_value, alpha, iterations in VARIANTS:
                output.append({
                    "experiment_id": "mcts-pw-safe-sensitivity",
                    "manifest_id": f"pw-safe-drone-{value_head}-{seed}-{variant}",
                    "domain": "drone", "value_head": value_head,
                    "seed": seed, "variant": variant,
                    "source_checkpoint": source["source_checkpoint"],
                    "source_training_job_id": source["source_training_job_id"],
                    "width_or_max_width": 20, "iterations": iterations,
                    "pw_min_width": k_min, "pw_c": c_value,
                    "pw_alpha": alpha, "terminal_safe": "true",
                    "histogram_logging": "compact-v1", "workers": 3,
                    "cpus": 6, "memory": "120G",
                    "time_limit": "3-00:00:00", "status": "ready",
                })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(f"WROTE|{OUTPUT}|rows={len(output)}")


if __name__ == "__main__":
    main()

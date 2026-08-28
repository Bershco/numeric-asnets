#!/usr/bin/env python3
"""Build the four-additional-seed, two-VH Kmin=3 PW extension."""

from __future__ import annotations

import csv
from pathlib import Path


RESULTS = Path("experiment_tracking/experiment_results.csv")
OUTPUT = Path(
    "experiment_tracking/mcts_progressive_widening_sensitivity/"
    "kmin3_extension_manifest.csv"
)
SEEDS = (1073581256, 1239739722, 1472491096, 534933607)


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    sources = {}
    for row in rows:
        if (
            row["experiment_id"] == "MAIN-VAL"
            and row["task_type"] == "policy_eval"
            and row["domain"] == "drone"
            and row["stage"] == "stage1"
            and row["endpoint"] == "validation_selected"
            and int(row["seed"]) in SEEDS
        ):
            sources[(row["value_head"], int(row["seed"]))] = row

    expected = {(vh, seed) for vh in ("off", "on") for seed in SEEDS}
    missing = expected - set(sources)
    if missing:
        raise RuntimeError(f"Missing source checkpoints: {sorted(missing)}")

    output = []
    for value_head in ("off", "on"):
        for seed in SEEDS:
            source = sources[(value_head, seed)]
            output.append({
                "experiment_id": "mcts-pw-safe-kmin3-extension",
                "manifest_id": f"pw-safe-drone-{value_head}-{seed}-safe_pw_kmin3_ext",
                "domain": "drone",
                "value_head": value_head,
                "seed": seed,
                "variant": "safe_pw_kmin3",
                "source_checkpoint": source["checkpoint"],
                "source_training_job_id": source["source_training_job_id"],
                "policy_score": source["score"],
                "width_or_max_width": 20,
                "iterations": 70,
                "pw_min_width": 3,
                "pw_c": 0.6,
                "pw_alpha": 0.5,
                "terminal_safe": "true",
                "histogram_logging": "compact-v1",
                "workers": 3,
                "cpus": 6,
                "memory": "120G",
                "time_limit": "3-00:00:00",
                "status": "ready",
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"WROTE|{OUTPUT}|rows={len(output)}")


if __name__ == "__main__":
    main()

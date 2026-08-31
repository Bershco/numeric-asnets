#!/usr/bin/env python3
"""Build the ten-seed comparative Counters Stage-2 narrow result tables."""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT_DIR = TRACK / "mcts_counters_width_sensitivity"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        candidate = abs(statistics.mean(v * s for v, s in zip(values, signs)))
        extreme += candidate >= observed - 1e-12
    return extreme / (2 ** len(values))


def main() -> None:
    policies = {
        (row["value_head"], row["seed"]): int(float(row["score"]))
        for row in read(TRACK / "experiment_results.csv")
        if row["experiment_id"] == "MAIN-VAL"
        and row["domain"] == "counters"
        and row["stage"] == "stage2"
        and row["endpoint"] == "validation_selected"
        and row["task_type"] == "policy_eval"
    }
    cutoffs = read(TRACK / "mcts_runtime_cutoff_jobs_20260831.csv")
    cutoffs += read(OUT_DIR / "stage2_narrow_terminal3_cutoffs.csv")
    rows = []
    seen = set()
    for row in cutoffs:
        if row.get("domain") != "counters" or row.get("stage") != "stage2":
            continue
        if "narrow" not in row.get("arm", ""):
            continue
        key = (row["value_head"], row["seed"])
        if key in seen:
            continue
        seen.add(key)
        policy = policies[key]
        rows.append({
            "value_head": key[0], "seed": key[1], "policy": policy,
            "narrow_30m": int(row["successes_le_1800s"]),
            "narrow_2h": int(row["successes_le_7200s"]),
            "narrow_6h": int(row["successes_le_21600s"]),
            "recorded_instances": int(row["recorded_instances"]),
            "unclassified_instances": 59 - int(row["recorded_instances"]),
            "job_id": row["job_id"], "slurm_state": row["slurm_state"],
            "source_log": row["source_log"],
        })
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 matched seeds, found {len(rows)}")
    rows.sort(key=lambda row: (row["value_head"], int(row["seed"])))
    seed_path = OUT_DIR / "stage2_narrow_matched_10seed.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = []
    t95 = 2.262157
    for vh in ("off", "on"):
        selected = [row for row in rows if row["value_head"] == vh]
        for arm in ("narrow_30m", "narrow_2h", "narrow_6h"):
            differences = [row[arm] - row["policy"] for row in selected]
            mean = statistics.mean(differences)
            margin = t95 * statistics.stdev(differences) / math.sqrt(len(differences))
            summary.append({
                "value_head": vh, "n": len(selected),
                "policy_mean": statistics.mean(row["policy"] for row in selected),
                "arm": arm,
                "narrow_mean": statistics.mean(row[arm] for row in selected),
                "paired_change": mean, "ci_low": mean - margin,
                "ci_high": mean + margin, "raw_signflip_p": signflip(differences),
                "unclassified_instances": sum(row["unclassified_instances"] for row in selected),
            })
    summary_path = OUT_DIR / "stage2_narrow_summary_10seed.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(f"seed_rows={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()

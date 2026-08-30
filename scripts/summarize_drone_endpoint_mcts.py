#!/usr/bin/env python3
"""Join reconciled Drone endpoint MCTS results to matched policy endpoints."""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def signflip(values: list[float]) -> float:
    observed = abs(statistics.fmean(values))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        if abs(statistics.fmean(v * s for v, s in zip(values, signs))) >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(values))


def main() -> None:
    mcts = read(TRACKING / "drone_endpoint_mcts_results.csv")
    policy = read(TRACKING / "policy_endpoint_results.csv")
    policy_index = {
        (row["experiment_id"], row["value_head"], row["seed"]): row
        for row in policy
        if row["domain"] == "drone" and row["stage"] == "stage2"
        and row["endpoint"] == "validation_selected"
    }
    tag_experiment = {"SR10M": "MAIN-VAL", "SR10TCM": "MAIN-TERM"}
    pairs = []
    for row in mcts:
        experiment = tag_experiment.get(row["tag"])
        if not experiment:
            continue
        matched = policy_index.get((experiment, row["value_head"], row["seed"]))
        if not matched:
            continue
        policy_score = int(matched["score"])
        mcts_score = int(row["successes"])
        pairs.append({
            "experiment_id": experiment, "value_head": row["value_head"],
            "seed": row["seed"], "policy_score": policy_score,
            "mcts_score": mcts_score, "paired_change": mcts_score - policy_score,
            "policy_log": matched["source_evaluation_log"],
            "mcts_log": row["source_evaluation_log"],
            "mcts_val_valid": row["val_valid"],
            "mcts_val_invalid": row["val_invalid"],
        })
    with (TRACKING / "drone_endpoint_mcts_paired_results.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(pairs)
    summary = []
    for key in sorted({(row["experiment_id"], row["value_head"]) for row in pairs}):
        rows = [row for row in pairs if (row["experiment_id"], row["value_head"]) == key]
        differences = [float(row["paired_change"]) for row in rows]
        n = len(rows); mean = statistics.fmean(differences)
        if n >= 2:
            half = T975[n] * statistics.stdev(differences) / math.sqrt(n)
            low, high = mean - half, mean + half
            p_value = signflip(differences)
        else:
            low = high = mean; p_value = 1.0
        summary.append({
            "experiment_id": key[0], "value_head": key[1], "n": n,
            "policy_mean": f"{statistics.fmean(float(row['policy_score']) for row in rows):.3f}",
            "mcts_mean": f"{statistics.fmean(float(row['mcts_score']) for row in rows):.3f}",
            "paired_change": f"{mean:.3f}", "ci_low": f"{low:.3f}",
            "ci_high": f"{high:.3f}", "exact_signflip_p": f"{p_value:.6f}",
        })
    with (TRACKING / "drone_endpoint_mcts_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    print(f"pairs={len(pairs)} summaries={len(summary)}")


if __name__ == "__main__":
    main()

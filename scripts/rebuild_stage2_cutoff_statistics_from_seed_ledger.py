#!/usr/bin/env python3
"""Rebuild Stage-2 cutoff statistics from the complete local seed ledger.

This avoids re-reading remote logs after every seed row has already been
provenance-linked and frozen locally.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking" / "stage2_policy_mcts_seed_cutoffs_latest.csv"
OUTPUT = ROOT / "experiment_tracking" / "stage2_policy_mcts_all_cutoff_statistics_latest.csv"


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    return sum(
        abs(statistics.mean(value * sign for value, sign in zip(values, signs))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(values))
    ) / 2 ** len(values)


def interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = 2.262 * statistics.stdev(values) / math.sqrt(10) if statistics.stdev(values) else 0.0
    return mean - half, mean + half


with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
    seed_rows = list(csv.DictReader(stream))

groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
for row in seed_rows:
    groups[(row["stage2_branch"], row["domain"], row["value_head"])].append(row)

summary: list[dict[str, object]] = []
for (branch, domain, value_head), rows in sorted(groups.items()):
    if len(rows) != 10:
        raise RuntimeError(f"incomplete cell: {(branch, domain, value_head)} has {len(rows)} rows")
    output: dict[str, object] = {
        "stage2_branch": branch,
        "domain": domain,
        "value_head": value_head,
        "search": rows[0]["search"],
        "n": 10,
        "policy_mean": statistics.mean(float(row["policy_score"]) for row in rows),
    }
    for cutoff in ("30m", "2h", "6h"):
        policy = [float(row["policy_score"]) for row in rows]
        mcts = [float(row[f"mcts_{cutoff}"]) for row in rows]
        differences = [after - before for before, after in zip(policy, mcts)]
        low, high = interval(differences)
        output.update({
            f"mcts_mean_{cutoff}": statistics.mean(mcts),
            f"change_{cutoff}": statistics.mean(differences),
            f"ci95_low_{cutoff}": low,
            f"ci95_high_{cutoff}": high,
            f"raw_p_{cutoff}": signflip(differences),
        })
    output["row_level_provenance"] = "experiment_tracking/stage2_policy_mcts_seed_cutoffs_latest.csv"
    summary.append(output)

for cutoff in ("30m", "2h", "6h"):
    ranked = sorted(enumerate(summary), key=lambda item: float(item[1][f"raw_p_{cutoff}"]))
    running = 0.0
    for rank, (index, row) in enumerate(ranked):
        running = max(running, min(1.0, float(row[f"raw_p_{cutoff}"]) * (len(ranked) - rank)))
        summary[index][f"holm_p_{cutoff}"] = running

fields = [
    "stage2_branch", "domain", "value_head", "search", "n", "policy_mean",
    "mcts_mean_30m", "change_30m", "ci95_low_30m", "ci95_high_30m", "raw_p_30m",
    "mcts_mean_2h", "change_2h", "ci95_low_2h", "ci95_high_2h", "raw_p_2h",
    "mcts_mean_6h", "change_6h", "ci95_low_6h", "ci95_high_6h", "raw_p_6h",
    "row_level_provenance", "holm_p_30m", "holm_p_2h", "holm_p_6h",
]
with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print(f"wrote {len(summary)} complete Stage-2 cells to {OUTPUT}")

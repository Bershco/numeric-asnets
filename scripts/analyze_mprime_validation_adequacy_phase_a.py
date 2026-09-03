#!/usr/bin/env python3
"""Phase-A MPrime validation adequacy metrics from existing Stage-1 records."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1/validation_test_checkpoint_audit.csv"
SEEDS = ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1/validation_adequacy_phase_a_stage1_seeds_20260903.csv"
SUMMARY = ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1/validation_adequacy_phase_a_stage1_summary_20260903.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else float("nan")


def kendall_tau_b(left: list[float], right: list[float]) -> float:
    concordant = discordant = tied_left = tied_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = (left[i] > left[j]) - (left[i] < left[j])
            dy = (right[i] > right[j]) - (right[i] < right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_left += 1
            elif dy == 0:
                tied_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_left)
        * (concordant + discordant + tied_right)
    )
    return (concordant - discordant) / denominator if denominator else float("nan")


groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
for row in read(SOURCE):
    groups[(row["value_head"], row["seed"])].append(row)

seed_rows = []
for (value_head, seed), rows in sorted(groups.items()):
    rows.sort(key=lambda row: int(row["epoch"]))
    validation = [float(row["validation_success"]) for row in rows]
    test = [float(row["test_success"]) for row in rows]
    maximum = max(validation)
    at_max = [value == maximum for value in validation]
    longest = current = 0
    for flag in at_max:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    selected_candidates = [row for row in rows if "selected" in row["roles"]]
    if len(selected_candidates) != 1:
        raise RuntimeError(f"{value_head}/{seed}: selected rows={len(selected_candidates)}")
    selected = selected_candidates[0]
    test_best = max(rows, key=lambda row: (int(row["test_success"]), -int(row["epoch"])))
    seed_rows.append({
        "value_head": value_head,
        "seed": seed,
        "checkpoints": len(rows),
        "validation_unique_scores": len(set(validation)),
        "validation_max": f"{maximum:.6f}",
        "validation_max_fraction": f"{sum(at_max) / len(rows):.6f}",
        "validation_first_max_epoch": rows[at_max.index(True)]["epoch"],
        "validation_longest_observed_max_plateau": longest,
        "spearman_validation_test": f"{pearson(ranks(validation), ranks(test)):.6f}",
        "kendall_tau_b_validation_test": f"{kendall_tau_b(validation, test):.6f}",
        "selected_epoch": selected["epoch"],
        "selected_test_score": selected["test_success"],
        "observed_test_best_epoch": test_best["epoch"],
        "observed_test_best_score": test_best["test_success"],
        "selected_test_regret": int(test_best["test_success"]) - int(selected["test_success"]),
        "source_checkpoint_audit": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
    })

with SEEDS.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(seed_rows)

summary_rows = []
for value_head in ("off", "on"):
    subset = [row for row in seed_rows if row["value_head"] == value_head]
    mean = lambda field: sum(float(row[field]) for row in subset) / len(subset)
    summary_rows.append({
        "value_head": value_head,
        "lineages": len(subset),
        "checkpoints": sum(int(row["checkpoints"]) for row in subset),
        "mean_unique_validation_scores": f"{mean('validation_unique_scores'):.3f}",
        "mean_validation_max_fraction": f"{mean('validation_max_fraction'):.6f}",
        "positive_spearman_lineages": sum(float(row["spearman_validation_test"]) > 0 for row in subset),
        "mean_within_lineage_spearman": f"{mean('spearman_validation_test'):.6f}",
        "mean_within_lineage_kendall_tau_b": f"{mean('kendall_tau_b_validation_test'):.6f}",
        "mean_selected_test_regret": f"{mean('selected_test_regret'):.3f}",
        "seed_ledger": str(SEEDS.relative_to(ROOT)).replace("\\", "/"),
    })
with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(summary_rows)

print(f"wrote {len(seed_rows)} seed rows and {len(summary_rows)} summaries")

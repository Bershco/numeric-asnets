#!/usr/bin/env python3
"""Freeze declared-budget Counters terminal-led Stage-2 narrow MCTS results."""

from __future__ import annotations

import csv
import itertools
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "experiment_tracking"
SEEDS = {"534933607", "923500475", "1073581256", "1239739722", "1472491096",
         "1510771779", "1963100312", "1972442430", "2011206605", "2082152039"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = 2.262 * statistics.stdev(values) / math.sqrt(len(values)) if statistics.stdev(values) else 0.0
    return mean - half, mean + half


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    return sum(
        abs(statistics.mean(v * s for v, s in zip(values, signs))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(values))
    ) / (2 ** len(values))


def holm_pair(first: float, second: float) -> tuple[float, float]:
    ordered = sorted(enumerate((first, second)), key=lambda item: item[1])
    adjusted = [0.0, 0.0]
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (2 - rank)))
        adjusted[index] = running
    return adjusted[0], adjusted[1]


def main() -> None:
    policies = {}
    for row in read(T / "policy_paired_seed_results.csv"):
        if row["experiment_id"] == "MAIN-TERM" and row["domain"] == "counters":
            policies[(row["value_head"], row["seed"])] = row

    dynamic = []
    for row in read(T / "dynamic_experiment_jobs_latest.csv"):
        if row["experiment"] != "Stage-2 MCTS branch completion — Counters":
            continue
        seed_match = re.search(r"_s(\d+)_", row["job_name"])
        if not seed_match or seed_match.group(1) not in SEEDS:
            continue
        vh = "off" if "orig_novh" in row["job_name"] else "on"
        dynamic.append((vh, seed_match.group(1), row))
    assert len(dynamic) == 20, len(dynamic)

    seed_rows = []
    for vh, seed, row in sorted(dynamic):
        policy = policies[(vh, seed)]
        classified = int(float(row["classified_instances"] or 0))
        censored = row["state"] != "COMPLETED" or classified < 59
        seed_rows.append({
            "domain": "counters", "value_head": vh, "seed": seed,
            "policy_score": policy["after_score"],
            "mcts_30m": row["success_30m"], "mcts_2h": row["success_2h"],
            "mcts_6h": row["success_6h"], "classified_instances": classified,
            "scheduler_state": row["state"], "declared_budget_censored": str(censored).lower(),
            "mcts_job_id": row["job_id"], "source_training_job": policy["after_training_job"],
            "source_policy_job": policy["after_evaluation_job"],
            "source_policy_log": policy["after_log"],
            "source_mcts_log": row["source_evaluation_log"],
        })

    summaries = []
    for vh in ("off", "on"):
        cell = [row for row in seed_rows if row["value_head"] == vh]
        item = {
            "domain": "counters", "value_head": vh, "n": 10,
            "n_censored_allocations": sum(row["declared_budget_censored"] == "true" for row in cell),
            "policy_mean": statistics.mean(float(row["policy_score"]) for row in cell),
        }
        for cutoff in ("30m", "2h", "6h"):
            scores = [float(row[f"mcts_{cutoff}"]) for row in cell]
            diffs = [score - float(row["policy_score"]) for score, row in zip(scores, cell)]
            low, high = ci(diffs)
            item[f"mcts_mean_{cutoff}"] = statistics.mean(scores)
            item[f"delta_{cutoff}"] = statistics.mean(diffs)
            item[f"ci95_low_{cutoff}"] = low
            item[f"ci95_high_{cutoff}"] = high
            item[f"raw_p_{cutoff}"] = signflip(diffs)
        item["status"] = "terminal_declared_budget_with_censoring"
        item["row_level_provenance"] = "experiment_tracking/counters_terminal_stage2_narrow_seed_results_latest.csv"
        summaries.append(item)

    for cutoff in ("30m", "2h", "6h"):
        first, second = holm_pair(*(float(row[f"raw_p_{cutoff}"]) for row in summaries))
        summaries[0][f"holm_p_within_counters_{cutoff}"] = first
        summaries[1][f"holm_p_within_counters_{cutoff}"] = second

    for path, data in (
        (T / "counters_terminal_stage2_narrow_seed_results_latest.csv", seed_rows),
        (T / "counters_terminal_stage2_narrow_statistics_latest.csv", summaries),
    ):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)

    branch_path = T / "stage2_policy_mcts_comparison_by_branch_latest.csv"
    branch = read(branch_path)
    by_vh = {row["value_head"]: row for row in summaries}
    for row in branch:
        if row["domain"] != "counters" or row["stage2_branch"] != "terminal_led":
            continue
        result = by_vh[row["value_head"]]
        row.update({
            "n_scheduler_terminal": "10", "n_live": "0", "n_complete_fixed_budget": "10",
            "policy_mean": f'{result["policy_mean"]:.1f}',
            "mcts_30m": f'{result["mcts_mean_30m"]:.1f}',
            "mcts_2h": f'{result["mcts_mean_2h"]:.1f}',
            "mcts_6h": f'{result["mcts_mean_6h"]:.1f}',
            "change_6h": f'{result["delta_6h"]:.1f}',
            "ci95_low": f'{result["ci95_low_6h"]:.3f}', "ci95_high": f'{result["ci95_high_6h"]:.3f}',
            "raw_signflip_p": str(result["raw_p_6h"]),
            "holm_p": str(result["holm_p_within_counters_6h"]),
            "status": "terminal_declared_budget_with_censoring",
            "conclusion": (
                "Declared-budget mean includes OOM/timeout allocations as failures for unclassified instances; "
                + ("small positive mean" if row["value_head"] == "off" else "positive but variable mean")
            ),
            "row_level_provenance": "experiment_tracking/counters_terminal_stage2_narrow_seed_results_latest.csv",
        })
    with branch_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(branch[0])); writer.writeheader(); writer.writerows(branch)
    print("wrote 20 Counters seed rows, two cutoff-statistic rows, and refreshed Stage-2 branch table")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build matched five-seed PW70 comparisons for terminal confirmatory cells."""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiment_tracking" / "mcts_progressive_widening_cross_domain"
SCREEN = BASE / "terminal_results_20260901.csv"
MANIFEST = BASE / "pw70_confirmatory_expansion_manifest.csv"
CUTOFFS = BASE / "pw70_confirmatory_cutoffs_20260904.csv"
FIXED = ROOT / "experiment_tracking" / "stage1_mcts_results.csv"
SEEDS_OUT = BASE / "pw70_confirmatory_seed_results_20260904.csv"
SUMMARY_OUT = BASE / "pw70_confirmatory_statistics_20260904.csv"
T975 = {4: 2.776}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def signflip(values: list[float]) -> float:
    observed = abs(sum(values) / len(values))
    return sum(
        abs(sum(sign * value for sign, value in zip(signs, values)) / len(values)) >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ) / (2 ** len(values))


def interval(values: list[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = T975[len(values) - 1] * math.sqrt(variance / len(values))
    return mean, mean - half, mean + half


screen = [
    row for row in read(SCREEN)
    if row["domain"] in {"fo_counters", "rover"}
]
manifest = {(row["domain"], row["value_head"], row["seed"]): row for row in read(MANIFEST)}
cutoffs = {(row["domain"], row["value_head"], row["seed"]): row for row in read(CUTOFFS)}
fixed = {
    (row["domain"], row["value_head"], row["seed"]): row
    for row in read(FIXED)
    if row["domain"] in {"fo_counters", "rover"}
}

rows: list[dict[str, str | float]] = []
for row in screen:
    rows.append({
        "experiment_id": "MCTS-PW70-CONFIRMATORY",
        "domain": row["domain"], "stage": "stage1", "value_head": row["value_head"],
        "seed": row["seed"], "seed_role": "screen_reused",
        "policy_score": row["policy_score"], "fixed_20_70_score": row["fixed_score"],
        "pw70_30m_score": row["pw_30m"], "pw70_2h_score": row["pw_2h"],
        "pw70_6h_score": row["pw_6h"], "terminal": "true",
        "source_pw_log": row["source_pw_log"],
        "source_fixed_log": fixed[(row["domain"], row["value_head"], row["seed"])]["source_evaluation_log"],
    })
for key, man in manifest.items():
    domain, value_head, seed = key
    if domain not in {"fo_counters", "rover"}:
        continue
    result = cutoffs[key]
    rows.append({
        "experiment_id": "MCTS-PW70-CONFIRMATORY",
        "domain": domain, "stage": "stage1", "value_head": value_head,
        "seed": seed, "seed_role": "confirmation_new",
        "policy_score": man["policy_score"],
        "fixed_20_70_score": fixed[key]["successes"],
        "pw70_30m_score": result["successes_le_1800s"],
        "pw70_2h_score": result["successes_le_7200s"],
        "pw70_6h_score": result["successes_le_21600s"],
        "terminal": str(result["slurm_state"] not in {"RUNNING", "PENDING"}).lower(),
        "source_pw_log": result["source_log"],
        "source_fixed_log": fixed[key]["source_evaluation_log"],
    })
rows.sort(key=lambda row: (str(row["domain"]), str(row["value_head"]), int(str(row["seed"]))))

with SEEDS_OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)

summaries = []
for domain in ("fo_counters", "rover"):
    for value_head in ("off", "on"):
        group = [row for row in rows if row["domain"] == domain and row["value_head"] == value_head]
        if len(group) != 5 or any(row["terminal"] != "true" for row in group):
            continue
        policy = [float(row["policy_score"]) for row in group]
        fixed_scores = [float(row["fixed_20_70_score"]) for row in group]
        pw6 = [float(row["pw70_6h_score"]) for row in group]
        versus_policy = [right - left for left, right in zip(policy, pw6)]
        versus_fixed = [right - left for left, right in zip(fixed_scores, pw6)]
        pmean, plow, phigh = interval(versus_policy)
        fmean, flow, fhigh = interval(versus_fixed)
        summaries.append({
            "experiment_id": "MCTS-PW70-CONFIRMATORY",
            "domain": domain, "stage": "stage1", "value_head": value_head, "matched_n": 5,
            "policy_mean": sum(policy) / 5,
            "fixed_20_70_mean": sum(fixed_scores) / 5,
            "pw70_30m_mean": sum(float(row["pw70_30m_score"]) for row in group) / 5,
            "pw70_2h_mean": sum(float(row["pw70_2h_score"]) for row in group) / 5,
            "pw70_6h_mean": sum(pw6) / 5,
            "pw_minus_policy_mean": pmean, "pw_minus_policy_ci95_low": plow,
            "pw_minus_policy_ci95_high": phigh, "pw_minus_policy_raw_signflip_p": signflip(versus_policy),
            "pw_minus_fixed_mean": fmean, "pw_minus_fixed_ci95_low": flow,
            "pw_minus_fixed_ci95_high": fhigh, "pw_minus_fixed_raw_signflip_p": signflip(versus_fixed),
            "row_level_companion": "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_seed_results_20260904.csv",
        })
with SUMMARY_OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(summaries)
print(f"seed_rows={len(rows)} completed_cell_summaries={len(summaries)}")

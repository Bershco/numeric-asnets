#!/usr/bin/env python3
"""Summarize currently scored PRESERVE-3 terminal-led selected endpoints."""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiment_tracking" / "preserve3_terminal_led"
POLICY = ROOT / "experiment_tracking" / "experiment_results.csv"
OUT = BASE / "terminal_led_selected_seed_results_20260903.csv"
SUMMARY = BASE / "terminal_led_selected_summary_20260903.csv"
TUNING = {"1963100312", "2011206605"}
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def signflip(values: list[float]) -> float:
    observed = abs(sum(values) / len(values))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        value = abs(sum(sign * item for sign, item in zip(signs, values)) / len(values))
        extreme += value >= observed - 1e-12
    return extreme / (2 ** len(values))


def main() -> None:
    s1 = {
        (row["domain"], row["value_head"], row["seed"]): row
        for row in read(POLICY)
        if row["experiment_id"] == "PRESERVE-4"
        and row["stage"] == "stage1"
        and row["endpoint"] == "final"
    }
    stage2_checkpoints: dict[tuple[str, str, str, str], str] = {}
    manifest_paths = list(BASE.glob("*tpp*policy*.csv"))
    manifest_paths += list((ROOT / "experiment_tracking" / "four_domain_preservation").glob("*tpp*policy*.csv"))
    manifest_paths += list(BASE.glob("policy_ready_*.csv"))
    for path in sorted(set(manifest_paths)):
        for row in read(path):
            if row.get("source_checkpoint_ref"):
                key = (row["domain"], row["value_head"], row["seed"], row["snapshot_epoch"])
                stage2_checkpoints[key] = row["source_checkpoint_ref"]
    evidence: dict[tuple[str, str, str, str], dict[str, str]] = {}
    source_paths = list(BASE.glob("preserve3_terminal_*_results.csv"))
    source_paths += list(BASE.glob("delivery_policy_retry_results_*.csv"))
    source_paths += list(BASE.glob("tpp_*_results_*.csv"))
    source_paths += list(BASE.glob("terminal_stage2_tpp_*_policy_results_*.csv"))
    for path in sorted(set(source_paths)):
        for row in read(path):
            manifest_id = row.get("manifest_id", "")
            if not manifest_id:
                continue
            key = (row["domain"], row["value_head"], row["seed"], row["snapshot_epoch"])
            previous = evidence.get(key)
            if previous is None or (not previous.get("score") and row.get("score")):
                evidence[key] = row

    rows: list[dict[str, str]] = []
    for key, row in evidence.items():
            if "validation_selected_policy" not in row["analysis_roles"] or not row["score"]:
                continue
            before = s1[(row["domain"], row["value_head"], row["seed"])]
            rows.append({
                "experiment_id": "PRESERVE-3-TERM",
                "domain": row["domain"],
                "value_head": row["value_head"],
                "seed": row["seed"],
                "seed_role": "tuning" if row["seed"] in TUNING else "held_out",
                "stage1_final_score": before["score"],
                "stage2_selected_score": row["score"],
                "paired_change": str(float(row["score"]) - float(before["score"])),
                "stage2_selected_epoch": row["snapshot_epoch"],
                "stage1_checkpoint": before["checkpoint"].replace("\\", "/"),
                "stage2_checkpoint": stage2_checkpoints.get(key, "").replace("\\", "/"),
                "stage1_training_log": before["source_training_log"].replace("\\", "/"),
                "stage1_evaluation_log": before["source_evaluation_log"].replace("\\", "/"),
                "stage2_training_job_id": row["source_training_job_id"],
                "stage2_evaluation_job_id": row["slurm_job_id"],
                "stage2_evaluation_log": row["source_log"].replace("\\", "/"),
            })
    rows.sort(key=lambda row: (row["domain"], row["value_head"], int(row["seed"])))
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    summaries = []
    for domain in sorted({row["domain"] for row in rows}):
        for value_head in ("off", "on"):
            group = [row for row in rows if row["domain"] == domain and row["value_head"] == value_head]
            if not group:
                continue
            diffs = [float(row["paired_change"]) for row in group]
            mean = sum(diffs) / len(diffs)
            if len(diffs) > 1:
                variance = sum((value - mean) ** 2 for value in diffs) / (len(diffs) - 1)
                half = T975[len(diffs) - 1] * math.sqrt(variance / len(diffs))
                low, high = mean - half, mean + half
            else:
                low = high = mean
            held = [row for row in group if row["seed_role"] == "held_out"]
            tuning = [row for row in group if row["seed_role"] == "tuning"]
            summaries.append({
                "experiment_id": "PRESERVE-3-TERM",
                "domain": domain,
                "value_head": value_head,
                "matched_n": len(group),
                "stage1_final_mean": sum(float(row["stage1_final_score"]) for row in group) / len(group),
                "stage2_selected_mean": sum(float(row["stage2_selected_score"]) for row in group) / len(group),
                "stage2_held_out_mean": (sum(float(row["stage2_selected_score"]) for row in held) / len(held)) if held else "",
                "stage2_tuning_mean": (sum(float(row["stage2_selected_score"]) for row in tuning) / len(tuning)) if tuning else "",
                "mean_change": mean,
                "ci95_low": low,
                "ci95_high": high,
                "raw_signflip_p": signflip(diffs),
                "completion_status": "complete" if len(group) == 10 else "provisional_missing_endpoint_retries",
                "row_level_companion": "experiment_tracking/preserve3_terminal_led/terminal_led_selected_seed_results_20260903.csv",
            })
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summaries)
    print(f"seed_rows={len(rows)} summaries={len(summaries)}")


if __name__ == "__main__":
    main()

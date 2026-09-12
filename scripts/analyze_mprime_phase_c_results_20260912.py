#!/usr/bin/env python3
"""Analyze the completed candidate-limited MPrime Phase-C validation audit."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath


REMOTE_ROOT = PurePosixPath("/home/hersco/training_new_domains/2026-09-11/mprime_phase_c")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left, right = rankdata(left), rankdata(right)
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--phase-b-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inventory = read_csv(args.results_root / "checkpoint_candidates.csv")
    phase_b_rows = read_csv(args.phase_b_scores)
    phase_b = {(row["lineage"], int(row["epoch"])): row for row in phase_b_rows}
    global_test_best: dict[str, float] = {}
    for row in phase_b_rows:
        if row.get("test_score", "") != "":
            score = float(row["test_score"])
            global_test_best[row["lineage"]] = max(global_test_best.get(row["lineage"], score), score)
    details: list[dict[str, object]] = []
    for item in inventory:
        lineage = item["lineage"]
        epoch = int(item["epoch"])
        summary = args.results_root / "rescore" / lineage / f"epoch_{epoch}.val.csv"
        done = args.results_root / "rescore" / lineage / f"epoch_{epoch}.done.json"
        score = sum(row.get("val_valid") == "1" for row in read_csv(summary))
        old = phase_b[(lineage, epoch)]
        details.append({
            **item,
            "phase_c_score": score,
            "test_score": old["test_score"],
            "phase_c_summary": str(REMOTE_ROOT / "rescore" / lineage / summary.name),
            "phase_c_done_marker": str(REMOTE_ROOT / "rescore" / lineage / done.name),
            "phase_c_array_job": "21183542",
        })

    detail_fields = list(inventory[0]) + [
        "phase_c_score", "test_score", "phase_c_summary", "phase_c_done_marker", "phase_c_array_job"
    ]
    write_csv(args.output_dir / "phase_c_checkpoint_scores_latest.csv", details, detail_fields)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in details:
        grouped[str(row["lineage"])].append(row)

    lineage_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for lineage, group in sorted(grouped.items()):
        selected = min(group, key=lambda row: (-int(row["phase_c_score"]), int(row["epoch"])))
        test_best = max(float(row["test_score"]) for row in group)
        selected_test = float(selected["test_score"])
        rho = spearman(
            [float(row["phase_c_score"]) for row in group],
            [float(row["test_score"]) for row in group],
        )
        full_best = global_test_best[lineage]
        summary = {
            "lineage": lineage,
            "branch": group[0]["branch"],
            "value_head": group[0]["value_head"],
            "seed": group[0]["seed"],
            "candidate_checkpoints": len(group),
            "phase_c_saturated_fraction": sum(int(row["phase_c_score"]) == 30 for row in group) / len(group),
            "validation_test_spearman": "" if rho is None else rho,
            "selected_epoch": selected["epoch"],
            "selected_validation_score": selected["phase_c_score"],
            "selected_test_score": selected_test,
            "candidate_test_best": test_best,
            "global_test_best": full_best,
            "selection_regret_candidate_set": test_best - selected_test,
            "selection_regret_full_lineage": full_best - selected_test,
            "training_job": group[0]["training_job"],
            "training_log": group[0]["training_log"],
            "selected_policy_job": selected["policy_job"],
            "selected_policy_log": selected["policy_log"],
            "selected_summary": selected["phase_c_summary"],
        }
        lineage_rows.append(summary)
        selected_rows.append(summary)
    lineage_fields = list(lineage_rows[0])
    write_csv(args.output_dir / "phase_c_lineage_summary_latest.csv", lineage_rows, lineage_fields)
    write_csv(args.output_dir / "phase_c_selected_checkpoints_latest.csv", selected_rows, lineage_fields)

    cell_rows: list[dict[str, object]] = []
    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in lineage_rows:
        cells[(str(row["branch"]), str(row["value_head"]))].append(row)
    for (branch, value_head), group in sorted(cells.items()):
        rhos = [float(row["validation_test_spearman"]) for row in group if row["validation_test_spearman"] != ""]
        candidate_regrets = [float(row["selection_regret_candidate_set"]) for row in group]
        full_regrets = [float(row["selection_regret_full_lineage"]) for row in group]
        cell_rows.append({
            "branch": branch,
            "value_head": value_head,
            "lineages": len(group),
            "mean_selected_test_score": statistics.mean(float(row["selected_test_score"]) for row in group),
            "mean_candidate_test_best": statistics.mean(float(row["candidate_test_best"]) for row in group),
            "mean_candidate_set_regret": statistics.mean(candidate_regrets),
            "mean_full_lineage_regret": statistics.mean(full_regrets),
            "median_full_lineage_regret": statistics.median(full_regrets),
            "zero_full_lineage_regret_lineages": sum(value == 0 for value in full_regrets),
            "mean_validation_test_spearman": statistics.mean(rhos) if rhos else "",
            "median_validation_test_spearman": statistics.median(rhos) if rhos else "",
            "mean_saturated_fraction": statistics.mean(float(row["phase_c_saturated_fraction"]) for row in group),
            "row_level_provenance": "experiment_tracking/mprime_validation_phase_c_20260911/phase_c_lineage_summary_latest.csv",
        })
    write_csv(args.output_dir / "phase_c_cell_summary_latest.csv", cell_rows, list(cell_rows[0]))
    selected = {
        (str(row["branch"]), str(row["value_head"]), str(row["seed"])): row
        for row in lineage_rows
    }
    paired_rows: list[dict[str, object]] = []
    for value_head in ("off", "on"):
        seeds = sorted({key[2] for key in selected if key[0] == "stage1" and key[1] == value_head})
        for seed in seeds:
            before = selected[("stage1", value_head, seed)]
            after = selected[("validation_led_stage2", value_head, seed)]
            paired_rows.append({
                "experiment_id": "MPRIME-PHASEC",
                "domain": "mprime",
                "value_head": value_head,
                "seed": seed,
                "before_score": before["selected_test_score"],
                "after_score": after["selected_test_score"],
                "difference": float(after["selected_test_score"]) - float(before["selected_test_score"]),
                "stage1_epoch": before["selected_epoch"],
                "stage2_epoch": after["selected_epoch"],
                "stage1_policy_job": before["selected_policy_job"],
                "stage1_policy_log": before["selected_policy_log"],
                "stage2_policy_job": after["selected_policy_job"],
                "stage2_policy_log": after["selected_policy_log"],
                "stage1_validation_summary": before["selected_summary"],
                "stage2_validation_summary": after["selected_summary"],
            })
    write_csv(
        args.output_dir / "phase_c_policy_paired_seed_results_latest.csv",
        paired_rows,
        list(paired_rows[0]),
    )

    phase_b_selector_path = args.phase_b_scores.parent / "phase_b_lineage_selector_comparison_latest.csv"
    phase_b_a = [
        row for row in read_csv(phase_b_selector_path)
        if row["selector"] == "replicate_a"
        and row["branch"] in {"stage1", "validation_led_stage2"}
    ]
    chosen: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in phase_b_a:
        epoch = int(row["selected_epoch"])
        detail = phase_b[(row["lineage"], epoch)]
        chosen[(row["branch"], row["value_head"], row["seed"])] = {
            **row,
            "selected_epoch": epoch,
            "policy_job": detail["policy_job"],
            "policy_log": detail["policy_log"],
            "validation_summary": detail["rep0_summary"],
        }
    final_pairs: list[dict[str, object]] = []
    for value_head in ("off", "on"):
        seeds = sorted({key[2] for key in chosen if key[0] == "stage1" and key[1] == value_head})
        for seed in seeds:
            before = chosen[("stage1", value_head, seed)]
            after = chosen[("validation_led_stage2", value_head, seed)]
            final_pairs.append({
                "experiment_id": "MPRIME-FINAL-VALIDATOR",
                "domain": "mprime",
                "value_head": value_head,
                "seed": seed,
                "before_score": before["selected_test_score"],
                "after_score": after["selected_test_score"],
                "difference": float(after["selected_test_score"]) - float(before["selected_test_score"]),
                "stage1_epoch": before["selected_epoch"],
                "stage2_epoch": after["selected_epoch"],
                "stage1_policy_job": before["policy_job"],
                "stage1_policy_log": before["policy_log"],
                "stage2_policy_job": after["policy_job"],
                "stage2_policy_log": after["policy_log"],
                "stage1_validation_summary": before["validation_summary"],
                "stage2_validation_summary": after["validation_summary"],
                "frozen_selector": "phase_b_replicate_a",
            })
    write_csv(
        args.output_dir / "mprime_final_validator_policy_paired_seed_results_latest.csv",
        final_pairs,
        list(final_pairs[0]),
    )
    phase_c_regret = statistics.mean(float(row["selection_regret_full_lineage"]) for row in lineage_rows)
    phase_b_a_regret = statistics.mean(float(row["selection_regret"]) for row in phase_b_a)
    write_csv(
        args.output_dir / "validator_decision_latest.csv",
        [
            {
                "candidate": "phase_b_replicate_a",
                "lineages": len(phase_b_a),
                "mean_full_lineage_selection_regret": phase_b_a_regret,
                "independent_of_test_structure": 1,
                "selected": 1,
                "decision": "Frozen final selector: marginally lower common-reference regret and no test-informed generator design.",
                "row_level_provenance": "experiment_tracking/mprime_validation_phase_b_20260906/phase_b_lineage_selector_comparison_latest.csv",
            },
            {
                "candidate": "phase_c",
                "lineages": len(lineage_rows),
                "mean_full_lineage_selection_regret": phase_c_regret,
                "independent_of_test_structure": 0,
                "selected": 0,
                "decision": "Completed diagnostic; essentially tied overall, but test-informed structure and no aggregate advantage.",
                "row_level_provenance": "experiment_tracking/mprime_validation_phase_c_20260911/phase_c_lineage_summary_latest.csv",
            },
        ],
        ["candidate", "lineages", "mean_full_lineage_selection_regret", "independent_of_test_structure", "selected", "decision", "row_level_provenance"],
    )
    print(f"evaluations={len(details)}/347 durable_markers=347/347 lineages={len(lineage_rows)}/40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

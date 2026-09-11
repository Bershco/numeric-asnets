#!/usr/bin/env python3
"""Summarize MPrime Phase-B checkpoint scores without rereading them locally."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath


FINAL_RE = re.compile(r"\[EVAL FINAL\].*?success=(\d+(?:\.\d+)?)/(?:20(?:\.0)?)")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, values: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rankdata(left), rankdata(right))


def branch(lineage: str) -> str:
    if lineage.startswith("stage1-"):
        return "stage1"
    if lineage.startswith("validation_led-"):
        return "validation_led_stage2"
    if lineage.startswith("terminal_led-"):
        return "terminal_led_stage2"
    return "unknown"


def test_score(path: str) -> float | None:
    candidate = Path(path)
    if not path or not candidate.exists():
        return None
    matches = FINAL_RE.findall(candidate.read_text(errors="replace", encoding="utf-8"))
    return float(matches[-1]) if matches else None


def validation_score(path: Path) -> int | None:
    # VAL summaries contain one row per printed candidate plan, not one row per
    # evaluation instance.  A 21/30 policy can therefore have only 21 rows even
    # when the evaluation is complete.  Completion is certified by the adjacent
    # identity-bound .done.json written only after the 30-instance final summary
    # and post-hoc validation both succeed.
    done = path.with_name(path.name.removesuffix(".val.csv") + ".done.json")
    if not path.exists() or not done.exists():
        return None
    data = rows(path)
    return sum(row.get("val_valid") == "1" for row in data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reported-root",
        help=(
            "Optional durable path written into provenance columns. The local "
            "--root is still used to parse files, but this prevents a temporary "
            "download directory from leaking into canonical result CSVs."
        ),
    )
    parser.add_argument(
        "--existing-checkpoint-scores",
        type=Path,
        help=(
            "Optional prior checkpoint_scores CSV used only to recover cached "
            "test scores when policy logs are not available on this machine."
        ),
    )
    args = parser.parse_args()

    cached_test_scores: dict[tuple[str, int], float] = {}
    if args.existing_checkpoint_scores:
        for item in rows(args.existing_checkpoint_scores):
            if item.get("test_score", "") != "":
                cached_test_scores[(item["lineage"], int(item["epoch"]))] = float(item["test_score"])

    inventory = rows(args.root / "checkpoints.csv")
    details: list[dict[str, object]] = []
    for row in inventory:
        lineage = row["lineage"]
        epoch = int(row["epoch"])
        item: dict[str, object] = {
            "lineage": lineage,
            "branch": branch(lineage),
            "value_head": row["value_head"],
            "seed": row["seed"],
            "epoch": epoch,
            "checkpoint": row["checkpoint"],
            "training_job": row["training_job"],
            "training_log": row["training_log"],
            "policy_job": row["policy_job"],
            "policy_log": row["policy_log"],
            "test_score": (
                test_score(row["policy_log"])
                if Path(row["policy_log"]).exists()
                else cached_test_scores.get((lineage, epoch))
            ),
        }
        scores = []
        for rep in range(2):
            summary = args.root / "rescore" / lineage / f"epoch_{epoch}_rep{rep}.val.csv"
            score = validation_score(summary)
            item[f"rep{rep}_score"] = "" if score is None else score
            item[f"rep{rep}_summary"] = (
                str(PurePosixPath(args.reported_root) / "rescore" / lineage / summary.name)
                if args.reported_root else str(summary)
            )
            if score is not None:
                scores.append(score)
        item["mean_validation_score"] = statistics.mean(scores) if scores else ""
        item["replicates_complete"] = len(scores)
        details.append(item)

    detail_fields = [
        "lineage", "branch", "value_head", "seed", "epoch", "rep0_score",
        "rep1_score", "mean_validation_score", "replicates_complete", "test_score",
        "checkpoint", "training_job", "training_log", "policy_job", "policy_log",
        "rep0_summary", "rep1_summary", "phase_b_stdout_glob",
    ]
    for item in details:
        item["phase_b_stdout_glob"] = (
            str(PurePosixPath(args.reported_root) / "rescore_%A_%a.log")
            if args.reported_root else str(args.root / "rescore_%A_%a.log")
        )
    write(args.output_dir / "checkpoint_scores.csv", details, detail_fields)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in details:
        grouped[str(item["lineage"])].append(item)
    lineage_rows: list[dict[str, object]] = []
    for lineage, group in sorted(grouped.items()):
        complete = [item for item in group if item["replicates_complete"] == 2]
        rep0 = [float(item["rep0_score"]) for item in complete]
        rep1 = [float(item["rep1_score"]) for item in complete]
        mean_scores = [float(item["mean_validation_score"]) for item in complete]
        test_pairs = [item for item in complete if item["test_score"] != "" and item["test_score"] is not None]
        consensus = min(
            complete,
            key=lambda item: (-float(item["mean_validation_score"]), int(item["epoch"])),
        ) if complete else None
        test_best = max((float(item["test_score"]) for item in test_pairs), default=None)
        selected_test = float(consensus["test_score"]) if consensus and consensus["test_score"] not in {"", None} else None
        lineage_rows.append({
            "lineage": lineage,
            "branch": branch(lineage),
            "value_head": group[0]["value_head"],
            "seed": group[0]["seed"],
            "checkpoints_expected": len(group),
            "checkpoints_complete_both_replicates": len(complete),
            "complete": int(len(complete) == len(group)),
            "replicate_spearman": "" if (rho := spearman(rep0, rep1)) is None else rho,
            "rep0_saturated_fraction": sum(score == 30 for score in rep0) / len(rep0) if rep0 else "",
            "rep1_saturated_fraction": sum(score == 30 for score in rep1) / len(rep1) if rep1 else "",
            "consensus_selected_epoch": "" if consensus is None else consensus["epoch"],
            "consensus_validation_mean": "" if consensus is None else consensus["mean_validation_score"],
            "consensus_test_score": "" if selected_test is None else selected_test,
            "retrospective_test_best": "" if test_best is None else test_best,
            "selection_regret": "" if selected_test is None or test_best is None else test_best - selected_test,
            "validation_test_spearman": "" if (rho := spearman(
                [float(item["mean_validation_score"]) for item in test_pairs],
                [float(item["test_score"]) for item in test_pairs],
            )) is None else rho,
            "training_log": group[0]["training_log"],
        })
    lineage_fields = list(lineage_rows[0])
    write(args.output_dir / "lineage_summary.csv", lineage_rows, lineage_fields)

    selector_rows: list[dict[str, object]] = []
    for lineage, group in sorted(grouped.items()):
        complete = [
            item for item in group
            if item["replicates_complete"] == 2
            and item["test_score"] not in {"", None}
        ]
        if not complete:
            continue
        test_best = max(float(item["test_score"]) for item in complete)
        for selector, score_field in (
            ("replicate_a", "rep0_score"),
            ("replicate_b", "rep1_score"),
            ("consensus_mean", "mean_validation_score"),
        ):
            selected = min(
                complete,
                key=lambda item: (-float(item[score_field]), int(item["epoch"])),
            )
            selected_test = float(selected["test_score"])
            selector_rows.append({
                "lineage": lineage,
                "branch": branch(lineage),
                "value_head": group[0]["value_head"],
                "seed": group[0]["seed"],
                "selector": selector,
                "selected_epoch": selected["epoch"],
                "selected_validation_score": selected[score_field],
                "selected_test_score": selected_test,
                "retrospective_test_best": test_best,
                "selection_regret": test_best - selected_test,
                "training_log": group[0]["training_log"],
            })
    write(
        args.output_dir / "lineage_selector_comparison.csv",
        selector_rows,
        list(selector_rows[0]),
    )

    selector_cell_rows: list[dict[str, object]] = []
    selector_cells: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in selector_rows:
        selector_cells[(str(item["branch"]), str(item["value_head"]), str(item["selector"]))].append(item)
    for (cell_branch, value_head, selector), group in sorted(selector_cells.items()):
        regrets = [float(item["selection_regret"]) for item in group]
        scores = [float(item["selected_test_score"]) for item in group]
        selector_cell_rows.append({
            "branch": cell_branch,
            "value_head": value_head,
            "selector": selector,
            "lineages": len(group),
            "mean_selected_test_score": statistics.mean(scores),
            "mean_selection_regret": statistics.mean(regrets),
            "median_selection_regret": statistics.median(regrets),
            "zero_regret_lineages": sum(value == 0 for value in regrets),
            "seed_level_source": "phase_b_lineage_selector_comparison_latest.csv",
            "job_log_columns": "training_log",
        })
    write(
        args.output_dir / "cell_selector_comparison.csv",
        selector_cell_rows,
        list(selector_cell_rows[0]),
    )

    cell_rows: list[dict[str, object]] = []
    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in lineage_rows:
        cells[(str(item["branch"]), str(item["value_head"]))].append(item)
    for (cell_branch, value_head), group in sorted(cells.items()):
        complete = [item for item in group if item["complete"] == 1]
        regrets = [float(item["selection_regret"]) for item in complete if item["selection_regret"] != ""]
        replicate_rhos = [float(item["replicate_spearman"]) for item in complete if item["replicate_spearman"] != ""]
        test_rhos = [float(item["validation_test_spearman"]) for item in complete if item["validation_test_spearman"] != ""]
        cell_rows.append({
            "branch": cell_branch,
            "value_head": value_head,
            "lineages_expected": len(group),
            "lineages_complete": len(complete),
            "mean_replicate_spearman": statistics.mean(replicate_rhos) if replicate_rhos else "",
            "median_replicate_spearman": statistics.median(replicate_rhos) if replicate_rhos else "",
            "mean_validation_test_spearman": statistics.mean(test_rhos) if test_rhos else "",
            "median_validation_test_spearman": statistics.median(test_rhos) if test_rhos else "",
            "mean_selection_regret": statistics.mean(regrets) if regrets else "",
            "median_selection_regret": statistics.median(regrets) if regrets else "",
            "mean_rep0_saturated_fraction": statistics.mean(float(item["rep0_saturated_fraction"]) for item in complete) if complete else "",
            "mean_rep1_saturated_fraction": statistics.mean(float(item["rep1_saturated_fraction"]) for item in complete) if complete else "",
        })
    write(args.output_dir / "cell_summary.csv", cell_rows, list(cell_rows[0]))
    print(
        f"checkpoint_replicates={sum(int(item['replicates_complete']) for item in details)} "
        f"of={len(details) * 2} complete_lineages={sum(int(item['complete']) for item in lineage_rows)}/60"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

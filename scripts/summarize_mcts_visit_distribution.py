#!/usr/bin/env python3
"""Summarize opt-in MCTS root traces at 30m, 2h, and 6h milestones."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PREFIX = "[MCTS DETERMINISM] "
MILESTONES = (("30m", 1800.0), ("2h", 7200.0), ("6h", 21600.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(errors="replace", encoding="utf-8") as stream:
            for line in stream:
                if PREFIX not in line:
                    continue
                payload = line.split(PREFIX, 1)[1].strip()
                record = json.loads(payload)
                record["source_log"] = str(path)
                records.append(record)
    return records


def normalized(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values] if total > 0 else [0.0 for value in values]


def js_divergence(first: list[float], second: list[float]) -> float:
    midpoint = [(a + b) / 2 for a, b in zip(first, second)]

    def kl(left: list[float], right: list[float]) -> float:
        return sum(a * math.log(a / b) for a, b in zip(left, right) if a > 0 and b > 0)

    return (kl(first, midpoint) + kl(second, midpoint)) / 2


def flatten(record: dict) -> dict[str, object]:
    children = record.get("children", [])
    raw = normalized([float(child["raw_network_probability"]) for child in children])
    visits = normalized([float(child["N"]) for child in children])
    selected = int(record["selected_action"])
    selected_child = next((child for child in children if int(child["action"]) == selected), None)
    ranked_policy = sorted(children, key=lambda child: float(child["raw_network_probability"]), reverse=True)
    policy_rank = next((index + 1 for index, child in enumerate(ranked_policy)
                        if int(child["action"]) == selected), None)
    return {
        "source_log": record["source_log"],
        "instance": record["instance"],
        "step": int(record["step"]),
        "elapsed_seconds": float(record["elapsed_seconds"]),
        "child_count": len(children),
        "total_edge_visits": int(record.get("total_edge_visits", sum(int(child["N"]) for child in children))),
        "visit_entropy": float(record.get("visit_entropy", 0.0)),
        "top1_top2_visit_margin": int(record.get("top1_top2_visit_margin", 0)),
        "selected_action": selected,
        "network_argmax_action": int(record["network_argmax_action"]),
        "selected_differs_from_network_argmax": int(selected != int(record["network_argmax_action"])),
        "selected_policy_rank_among_children": policy_rank,
        "selected_visit_share": float(selected_child["visit_share"]) if selected_child else "",
        "selected_raw_network_probability": float(record["selected_action_network_probability"]),
        "policy_visit_js_divergence_expanded_children": js_divergence(raw, visits),
    }


def mean_or_blank(rows: list[dict], field: str) -> object:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return statistics.mean(values) if values else ""


def summary_rows(rows: list[dict]) -> list[dict[str, object]]:
    output = []
    for source in sorted(set(str(row["source_log"]) for row in rows)):
        for instance in sorted(set(str(row["instance"]) for row in rows if row["source_log"] == source)):
            instance_rows = [row for row in rows if row["source_log"] == source and row["instance"] == instance]
            for label, limit in MILESTONES:
                selected = [row for row in instance_rows if float(row["elapsed_seconds"]) <= limit]
                output.append({
                    "source_log": source,
                    "instance": instance,
                    "cumulative_milestone": label,
                    "root_searches_completed": len(selected),
                    "last_external_step": max((int(row["step"]) for row in selected), default=-1),
                    "mean_child_count": mean_or_blank(selected, "child_count"),
                    "mean_total_edge_visits": mean_or_blank(selected, "total_edge_visits"),
                    "mean_visit_entropy": mean_or_blank(selected, "visit_entropy"),
                    "mean_top1_top2_visit_margin": mean_or_blank(selected, "top1_top2_visit_margin"),
                    "policy_mcts_disagreement_rate": mean_or_blank(selected, "selected_differs_from_network_argmax"),
                    "mean_selected_policy_rank": mean_or_blank(selected, "selected_policy_rank_among_children"),
                    "mean_selected_visit_share": mean_or_blank(selected, "selected_visit_share"),
                    "mean_policy_visit_js_divergence": mean_or_blank(selected, "policy_visit_js_divergence_expanded_children"),
                })
    return output


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    steps = [flatten(record) for record in read_records(args.logs)]
    write(args.output_prefix.with_name(args.output_prefix.name + "_steps.csv"), steps)
    write(args.output_prefix.with_name(args.output_prefix.name + "_milestones.csv"), summary_rows(steps))


if __name__ == "__main__":
    main()

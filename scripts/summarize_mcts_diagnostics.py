#!/usr/bin/env python3
"""Aggregate compact MCTS width and phase-time summaries from evaluation logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


WIDTH_RE = re.compile(
    r"\[MCTS WIDTH SUMMARY\].*?final_widths=\(count=(?P<node_n>\d+) .*?"
    r"mean=(?P<node_mean>[0-9.]+).*?\).*?root_widths=\(count=(?P<root_n>\d+) .*?"
    r"mean=(?P<root_mean>[0-9.]+).*?\)"
)
TIME_RE = re.compile(
    r"\[MCTS TIME SUMMARY\] selection=(?P<selection>[0-9.]+)s "
    r"expansion=(?P<expansion>[0-9.]+)s "
    r"successor_generation=(?P<successor>[0-9.]+)s "
    r"network_inference=(?P<network>[0-9.]+)s "
    r"evaluation=(?P<evaluation>[0-9.]+)s "
    r"backpropagation=(?P<backprop>[0-9.]+)s"
)


def summarize(path: Path) -> dict[str, object]:
    node_n = root_n = summaries = 0
    node_weighted = root_weighted = 0.0
    times = {name: 0.0 for name in (
        "selection", "expansion", "successor", "network", "evaluation", "backprop"
    )}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            width = WIDTH_RE.search(line)
            if width:
                current_node_n = int(width["node_n"])
                current_root_n = int(width["root_n"])
                node_n += current_node_n
                root_n += current_root_n
                node_weighted += current_node_n * float(width["node_mean"])
                root_weighted += current_root_n * float(width["root_mean"])
                summaries += 1
            timing = TIME_RE.search(line)
            if timing:
                for name in times:
                    times[name] += float(timing[name])
    return {
        "source_log": str(path),
        "completed_instance_summaries": summaries,
        "node_observations": node_n,
        "weighted_mean_children_all_nodes": node_weighted / node_n if node_n else "",
        "root_observations": root_n,
        "weighted_mean_root_children": root_weighted / root_n if root_n else "",
        **{f"{name}_seconds": value for name, value in times.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.logs]
    total_nodes = sum(int(row["node_observations"]) for row in rows)
    total_roots = sum(int(row["root_observations"]) for row in rows)
    aggregate = {
        "source_log": "__aggregate__",
        "completed_instance_summaries": sum(int(row["completed_instance_summaries"]) for row in rows),
        "node_observations": total_nodes,
        "weighted_mean_children_all_nodes": (
            sum(float(row["weighted_mean_children_all_nodes"]) * int(row["node_observations"])
                for row in rows if row["node_observations"])
            / total_nodes if total_nodes else ""
        ),
        "root_observations": total_roots,
        "weighted_mean_root_children": (
            sum(float(row["weighted_mean_root_children"]) * int(row["root_observations"])
                for row in rows if row["root_observations"])
            / total_roots if total_roots else ""
        ),
    }
    for name in ("selection", "expansion", "successor", "network", "evaluation", "backprop"):
        aggregate[f"{name}_seconds"] = sum(float(row[f"{name}_seconds"]) for row in rows)
    rows.append(aggregate)
    fields = list(rows[0])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader(); writer.writerow(aggregate)


if __name__ == "__main__":
    main()

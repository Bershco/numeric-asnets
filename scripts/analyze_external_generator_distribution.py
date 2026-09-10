#!/usr/bin/env python3
"""Compare test, thesis-validation, and frozen external-generator distributions."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


TYPE_RE = re.compile(
    r"^\s*([a-z0-9_\- ]+?)\s+-\s+(counter|rover|waypoint|objective|camera)\s*$",
    re.I | re.M,
)


def features(path: Path, domain: str) -> dict[str, object]:
    text = path.read_text(errors="replace").lower()
    row: dict[str, object] = {"path": str(path), "file": path.name}
    if domain == "fo_counters":
        counters = set(re.findall(r"\bc\d+\b", re.search(r"\(:objects(.*?)\)\s*\(:init", text, re.S).group(1)))
        values = [int(value) for value in re.findall(r"\(=\s*\(value\s+c\d+\)\s*(\d+)\)", text)]
        row.update(objects=len(counters), counters=len(counters), max_int=int(re.search(r"\(=\s*\(max_int\)\s*(\d+)\)", text).group(1)),
                   init_nonzero_fraction=sum(value != 0 for value in values) / len(values),
                   goals=len(re.findall(r"\(<=\s*\(\+\s*\(value", text)))
    else:
        counts = {kind: 0 for kind in ("rover", "waypoint", "objective", "camera")}
        obj = re.search(r"\(:objects(.*?)\)\s*\(:init", text, re.S).group(1)
        for names, kind in TYPE_RE.findall(obj):
            counts[kind] += len(names.split())
        row.update(objects=sum(counts.values()), **counts,
                   visible_edges=len(re.findall(r"\(visible\s+waypoint", text)),
                   traverse_edges=len(re.findall(r"\(can_traverse\s+rover", text)),
                   goals=len(re.findall(r"\((?:communicated_|at_lander|at rover|have_)", re.search(r"\(:goal(.*)", text, re.S).group(1))))
    return row


def summarize(rows: list[dict[str, object]], domain: str, distribution: str) -> list[dict[str, object]]:
    metrics = ["counters", "max_int", "init_nonzero_fraction", "goals"] if domain == "fo_counters" else ["rover", "waypoint", "objective", "camera", "visible_edges", "traverse_edges", "goals"]
    output = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        output.append({"domain": domain, "distribution": distribution, "n": len(values), "metric": metric,
                       "min": min(values), "mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    specs = {
        "fo_counters": {
            "paper_test": sorted((args.repo_root / "problems/numeric/fo-counters/instances").glob("instance_*.pddl"), key=lambda p: int(re.search(r"\d+", p.stem).group())),
            "thesis_validation": sorted((args.repo_root / "problem_generator/generated_validation_instances/fo-counters").glob("valid_*/*.pddl")),
            "yarin_external": sorted((args.external_root / "fo-counters/external").glob("*.pddl")),
        },
        "rover": {
            "paper_test": sorted((args.repo_root / "problems/numeric/rover/problems").glob("pfile*.pddl"), key=lambda p: int(re.search(r"\d+", p.stem).group())),
            "thesis_validation": sorted((args.repo_root / "problem_generator/generated_validation_instances/rover").glob("valid_*/*.pddl")),
            "yarin_external": sorted((args.external_root / "rover/external").glob("*.pddl")),
        },
    }
    summaries = []
    for domain, distributions in specs.items():
        for distribution, paths in distributions.items():
            if not paths:
                continue
            rows = [features(path, domain) | {"domain": domain, "distribution": distribution} for path in paths]
            all_rows.extend(rows); summaries.extend(summarize(rows, domain, distribution))
    for name, rows in (("generator_distribution_instances.csv", all_rows), ("generator_distribution_summary.csv", summaries)):
        fields = []
        for row in rows:
            fields.extend(field for field in row if field not in fields)
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(all_rows)} instance rows and {len(summaries)} summaries")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Join post-first-update Stage-2 epoch-0 scores to their Stage-1 sources."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


RAW_RE = re.compile(
    r"^(?P<path>.*?/(?P<eval_job>\d+)_Ev_(?P<stem>.+?)_src(?P<s2_job>\d+)_e0000\.txt):"
    r"\[EVAL FINAL\] success=(?P<score>[0-9.]+)/(?P<total>\d+)="
)

DOMAINS = ("block_grouping", "fo_counters", "counters", "drone", "rover")


def signflip_p(deltas: list[float]) -> float:
    observed = abs(sum(deltas) / len(deltas))
    extreme = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(deltas)):
        permuted = abs(sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas))
        extreme += permuted >= observed - 1e-12
        total += 1
    return extreme / total


def holm(pairs: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[index] = running
    return adjusted


def domain_from_stem(stem: str) -> str:
    for domain in DOMAINS:
        if stem.startswith(domain + "_"):
            return domain
    raise ValueError(f"Cannot identify domain in {stem!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.results.open(newline="", encoding="utf-8-sig") as handle:
        results = list(csv.DictReader(handle))

    s2_lookup = {}
    s1_lookup = {}
    for row in results:
        exp = row["experiment_id"]
        if exp not in {"MAIN-VAL", "MAIN-TERM"}:
            continue
        key = (exp, row["domain"], row["value_head"], row["seed"])
        if row["stage"] == "stage2" and row["endpoint"] == "validation_selected":
            s2_lookup[str(row["source_training_job_id"])] = row
        elif row["stage"] == "stage1":
            wanted = "validation_selected" if exp == "MAIN-VAL" else "final"
            if row["endpoint"] == wanted:
                s1_lookup[key] = row

    joined = []
    for line in args.raw.read_text(encoding="utf-8").splitlines():
        match = RAW_RE.match(line)
        if not match:
            continue
        gd = match.groupdict()
        s2 = s2_lookup.get(gd["s2_job"])
        if not s2:
            continue
        branch = s2["experiment_id"]
        domain = domain_from_stem(gd["stem"])
        vh = "off" if "_novh_" in gd["stem"] else "on"
        seed = re.search(r"_s(\d+)_", gd["stem"]).group(1)
        key = (branch, domain, vh, seed)
        s1 = s1_lookup.get(key)
        if not s1:
            raise KeyError(f"Missing Stage-1 source row for {key}")
        s2_score = float(gd["score"])
        s1_score = float(s1["score"])
        joined.append(
            {
                "branch": branch,
                "domain": domain,
                "value_head": vh,
                "seed": seed,
                "stage1_endpoint": s1["endpoint"],
                "stage1_epoch": s1["epoch"],
                "stage1_score": f"{s1_score:g}",
                "stage2_epoch": "0",
                "stage2_epoch0_score": f"{s2_score:g}",
                "change_after_first_update": f"{s2_score - s1_score:g}",
                "total": gd["total"],
                "stage1_training_job_id": s1["source_training_job_id"],
                "stage2_training_job_id": gd["s2_job"],
                "epoch0_evaluation_job_id": gd["eval_job"],
                "source_stage1_training_log": s1["source_training_log"],
                "source_stage1_evaluation_log": s1["source_evaluation_log"],
                "source_stage2_training_log": s2["source_training_log"],
                "source_epoch0_evaluation_log": gd["path"],
            }
        )

    joined.sort(key=lambda r: (r["branch"], r["domain"], r["value_head"], int(r["seed"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    groups = defaultdict(list)
    for row in joined:
        groups[(row["branch"], row["domain"], row["value_head"])].append(row)
    summaries = []
    for (branch, domain, vh), rows in sorted(groups.items()):
        s1_vals = [float(r["stage1_score"]) for r in rows]
        s2_vals = [float(r["stage2_epoch0_score"]) for r in rows]
        deltas = [b - a for a, b in zip(s1_vals, s2_vals)]
        mean_delta = sum(deltas) / len(rows)
        half_width = (
            2.262 * statistics.stdev(deltas) / math.sqrt(len(deltas))
            if len(deltas) == 10 else float("nan")
        )
        summaries.append(
            {
                "branch": branch,
                "domain": domain,
                "value_head": vh,
                "n": len(rows),
                "stage1_mean": f"{sum(s1_vals) / len(rows):.3f}",
                "stage2_epoch0_mean": f"{sum(s2_vals) / len(rows):.3f}",
                "mean_change_after_first_update": f"{mean_delta:.3f}",
                "ci95_low": f"{mean_delta - half_width:.3f}" if math.isfinite(half_width) else "",
                "ci95_high": f"{mean_delta + half_width:.3f}" if math.isfinite(half_width) else "",
                "raw_signflip_p": f"{signflip_p(deltas):.6f}" if len(deltas) == 10 else "",
                "holm_p_across_10_terminal_cells": "",
                "seeds_losing_5_or_more": sum(delta <= -5 for delta in deltas),
                "source_seed_ledger": str(args.output).replace("\\", "/"),
            }
        )
    complete_terminal = [
        (index, float(row["raw_signflip_p"]))
        for index, row in enumerate(summaries)
        if row["branch"] == "MAIN-TERM" and row["n"] == 10
    ]
    for index, adjusted in holm(complete_terminal).items():
        summaries[index]["holm_p_across_10_terminal_cells"] = f"{adjusted:.6f}"

    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(joined)} seed rows and {len(summaries)} summaries")


if __name__ == "__main__":
    main()

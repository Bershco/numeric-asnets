#!/usr/bin/env python3
"""Summarize Phase-B-A anchor curves and freeze coefficients; never submit jobs."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


ANCHORS = ("0", "0.03", "0.3", "1", "3", "10", "30")
EPOCH = re.compile(r"epoch_(\d+)_phase_b_a\.val\.csv$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def score(summary: Path) -> float:
    rows = read_csv(summary)
    if len(rows) > 30:
        raise RuntimeError(f"{summary}: more than 30 candidate plans")
    return sum(int(row["val_valid"]) for row in rows) / 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rescore-root", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--frozen-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    if len(manifest) != 28:
        raise RuntimeError(f"expected 28 lineages, got {len(manifest)}")
    curves: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in manifest:
        directory = args.rescore_root / row["manifest_id"]
        for summary in directory.glob("epoch_*_phase_b_a.val.csv"):
            match = EPOCH.search(summary.name)
            if match:
                curves[(row["value_head"], row["anchor"], row["seed"])].append(
                    (int(match.group(1)), score(summary))
                )
    expected_keys = {
        (vh, anchor, seed)
        for vh in ("off", "on")
        for anchor in ANCHORS
        for seed in ("1963100312", "2011206605")
    }
    if set(curves) != expected_keys:
        raise RuntimeError(f"incomplete curve identities: {expected_keys-set(curves)}")
    for key, values in curves.items():
        if len(values) != 21:
            raise RuntimeError(f"{key}: expected 21 points, got {len(values)}")

    evidence = []
    winners = {}
    for vh in ("off", "on"):
        for anchor in ANCHORS:
            seed_curves = [sorted(curves[(vh, anchor, seed)]) for seed in ("1963100312", "2011206605")]
            flattened = [value for curve in seed_curves for _, value in curve]
            evidence.append({
                "value_head": vh,
                "anchor": anchor,
                "points": "42",
                "mean_auc": f"{statistics.mean(flattened):.9f}",
                "mean_peak": f"{statistics.mean(max(v for _, v in curve) for curve in seed_curves):.9f}",
                "mean_final": f"{statistics.mean(curve[-1][1] for curve in seed_curves):.9f}",
                "selection_rule": "mean_auc_then_peak_then_final_then_smallest_anchor",
                "rescore_root": str(args.rescore_root),
            })
        candidates = [row for row in evidence if row["value_head"] == vh]
        winner = max(
            candidates,
            key=lambda row: (
                float(row["mean_auc"]),
                float(row["mean_peak"]),
                float(row["mean_final"]),
                -float(row["anchor"]),
            ),
        )
        winners[vh] = winner

    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(evidence)
    frozen = []
    for vh in ("off", "on"):
        winner = winners[vh]
        frozen.append({
            **winner,
            "status": "frozen_after_complete_phase_b_a_rescore",
            "manual_curve_review_required_before_training": "1",
        })
    with args.frozen_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frozen[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(frozen)
    print(
        "COMPLETE " + " ".join(
            f"{vh}={winners[vh]['anchor']}" for vh in ("off", "on")
        )
    )


if __name__ == "__main__":
    main()

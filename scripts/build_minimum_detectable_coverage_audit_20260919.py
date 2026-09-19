#!/usr/bin/env python3
"""Find the least uniformly distributed coverage gain that can pass Holm.

This is a resolution counterfactual.  For each direct coverage estimand, it
adds exactly one solved instance to ``k`` distinct non-ceiling seed pairs,
leaves the other pairs unchanged, and recomputes the exact two-sided sign-flip
p-value and the target's Holm value in the declared six-domain family.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiment_tracking" / "ceiling_significance_audit_20260919"
PRIMARY = ROOT / "experiment_tracking" / "advisor_followup_20260910" / "rq_primary_validation_led.csv"
PERFECT = AUDIT / "perfect_coverage_counterfactual.csv"
OUTPUT = AUDIT / "minimum_consistent_gain_for_holm.csv"

LABEL_TO_KEY = {
    "Block Grouping": "block_grouping",
    "Drone": "drone",
    "FO Counters": "fo_counters",
    "Rover": "rover",
    "Counters": "counters",
    "MPrime": "mprime",
}

ESTIMAND = {
    "RQ1": "VH-off: Stage 2 policy - Stage 1 policy",
    "RQ2": "VH-off direct: MCTS - same-checkpoint policy",
    "RQ3": "VH-on direct: Stage 2 policy - Stage 1 policy",
    "RQ4": "VH-on direct: MCTS - same-checkpoint policy",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def holm_target(pvalues: dict[str, float], target: str) -> float:
    ordered = sorted(pvalues, key=lambda name: pvalues[name])
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * pvalues[name])
        adjusted[name] = min(1.0, running)
    return adjusted[target]


def main() -> None:
    primary = read(PRIMARY)
    output: list[dict[str, object]] = []
    for source in read(PERFECT):
        rq = source["rq"]
        stage = "Stage 1" if source["stage"] == "stage1" else "Stage 2"
        domain = LABEL_TO_KEY[source["domain"]]
        family_estimand = ESTIMAND[rq]
        for cutoff in source["cutoff"].split("/"):
            family = [
                row for row in primary
                if row["rq"] == rq and row["stage"] == stage
                and row["cutoff"] == cutoff and row["estimand"] == family_estimand
                and row["raw_p"] != ""
            ]
            target = next(row for row in family if row["domain"] == domain)
            pvalues = {row["domain"]: float(row["raw_p"]) for row in family}
            max_pairs = int(source["effective_nonzero_pairs"])
            required = None
            required_raw = None
            required_holm = None
            for k in range(1, max_pairs + 1):
                pvalues[domain] = 2 ** (1 - k)
                adjusted = holm_target(pvalues, domain)
                if adjusted <= 0.05 + 1e-12:
                    required = k
                    required_raw = pvalues[domain]
                    required_holm = adjusted
                    break
            baseline = float(source["baseline_mean"])
            n = int(target["n"])
            output.append({
                "rq": rq,
                "stage": source["stage"],
                "value_head": source["value_head"],
                "domain": source["domain"],
                "cutoff": cutoff,
                "n": n,
                "capacity": int(source["ceiling"]),
                "baseline_mean": baseline,
                "observed_comparison_mean": target["comparison_mean"],
                "observed_raw_p": target["raw_p"],
                "observed_holm_p": target["holm_p"],
                "available_non_ceiling_pairs": max_pairs,
                "required_positive_pairs": "" if required is None else required,
                "minimum_total_additional_solves": "" if required is None else required,
                "minimum_mean_gain": "" if required is None else required / n,
                "minimum_target_mean": "" if required is None else baseline + required / n,
                "counterfactual_raw_p": "" if required_raw is None else required_raw,
                "counterfactual_holm_p": "" if required_holm is None else required_holm,
                "attainable_with_current_pairs": "no" if required is None else "yes",
                "assumption": "one additional solve in each of k distinct seed pairs; no regressions",
            })
    fields = list(output[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

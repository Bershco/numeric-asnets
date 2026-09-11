#!/usr/bin/env python3
"""Audit root visit ties and a visits -> Q -> prior tie-break rule."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PREFIX = "[MCTS DETERMINISM] "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--q-tolerance", default=1e-8, type=float)
    return parser.parse_args()


def records(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if PREFIX not in line:
                    continue
                payload = line.split(PREFIX, 1)[1].strip()
                row = json.loads(payload)
                row["source_log"] = str(path)
                yield row


def preferred_action(tied: list[dict], q_tolerance: float) -> tuple[int, str]:
    q_values = [float(child["Q"]) for child in tied]
    if max(q_values) - min(q_values) > q_tolerance:
        best_q = max(q_values)
        tied = [child for child in tied if best_q - float(child["Q"]) <= q_tolerance]
        reason = "q"
    else:
        reason = "policy_prior"
    if len(tied) > 1:
        best_prior = max(float(child["raw_network_probability"]) for child in tied)
        tied = [child for child in tied
                if abs(float(child["raw_network_probability"]) - best_prior) <= 1e-12]
        reason = "policy_prior" if reason == "policy_prior" else "q_then_policy_prior"
    return min(int(child["action"]) for child in tied), reason


def main() -> None:
    args = parse_args()
    output = []
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in records(args.logs):
        children = row.get("children", [])
        if not children:
            continue
        max_visits = max(int(child["N"]) for child in children)
        tied = [child for child in children if int(child["N"]) == max_visits]
        current = int(row["selected_action"])
        proposed, reason = preferred_action(tied, args.q_tolerance)
        source = str(row["source_log"])
        instance = str(row["instance"])
        key = (source, instance)
        counts[key]["roots"] += 1
        if len(tied) > 1:
            counts[key]["max_visit_ties"] += 1
        current_is_max = any(int(child["action"]) == current for child in tied)
        if len(tied) > 1 and current_is_max:
            counts[key]["auditable_ties"] += 1
            if proposed != current:
                counts[key]["would_change"] += 1
                output.append({
                    "source_log": source,
                    "instance": instance,
                    "step": int(row["step"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "current_action": current,
                    "proposed_action": proposed,
                    "network_argmax_action": int(row["network_argmax_action"]),
                    "max_visit_count": max_visits,
                    "number_tied_at_max_visits": len(tied),
                    "tie_break_reason": reason,
                    "current_policy_rank": next(
                        index + 1 for index, child in enumerate(sorted(
                            children,
                            key=lambda item: float(item["raw_network_probability"]),
                            reverse=True,
                        )) if int(child["action"]) == current
                    ),
                    "proposed_policy_rank": next(
                        index + 1 for index, child in enumerate(sorted(
                            children,
                            key=lambda item: float(item["raw_network_probability"]),
                            reverse=True,
                        )) if int(child["action"]) == proposed
                    ),
                })
        if not current_is_max:
            counts[key]["selected_outside_max_visits"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else ["source_log", "instance", "step"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary_fields = [
        "source_log", "instance", "roots", "max_visit_ties", "auditable_ties",
        "would_change", "selected_outside_max_visits", "tie_rate",
        "change_rate_among_auditable_ties", "first_changed_step", "scope",
        "interpretation",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for (source, instance), counter in sorted(counts.items()):
            first = next((row["step"] for row in output
                          if row["source_log"] == source and row["instance"] == instance), "")
            writer.writerow({
                "source_log": source,
                "instance": instance,
                **counter,
                "tie_rate": counter["max_visit_ties"] / counter["roots"],
                "change_rate_among_auditable_ties": (
                    counter["would_change"] / counter["auditable_ties"]
                    if counter["auditable_ties"] else 0.0
                ),
                "first_changed_step": first,
                "scope": "raw_logged_children_before_unlogged_goal_chase_and_eligibility_filters",
                "interpretation": "diagnostic_upper_bound_not_causal_pipeline_change_rate",
            })


if __name__ == "__main__":
    main()

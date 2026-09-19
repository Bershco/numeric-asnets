#!/usr/bin/env python3
"""Deduplicate divergence result artifacts and summarize first selectors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def selector(first: dict | None) -> str:
    if not first:
        return "no_divergence"
    selection = first.get("selection") or {}
    if selection.get("final_stage"):
        return str(selection["final_stage"])
    path = first.get("override_path") or []
    for stage in path:
        if stage.get("stage") == "goal_chase" and stage.get("applied") is True:
            return "goal_chase"
    for stage in reversed(path):
        if stage.get("selected_action") == first.get("selected_action"):
            return str(stage.get("stage") or "unknown")
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="+")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    by_key: dict[tuple[str, int], dict] = {}
    sources: dict[tuple[str, int], list[str]] = {}
    for root in args.root:
        for path in root.rglob("*.result.json") if root.exists() else []:
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (str(record["task_id"]), int(record["candidate_index"]))
            essential = (
                record.get("first_divergence_observed"),
                record.get("first_divergence"),
                record.get("outcome"),
            )
            if key in by_key:
                prior = by_key[key]
                prior_essential = (
                    prior.get("first_divergence_observed"),
                    prior.get("first_divergence"),
                    prior.get("outcome"),
                )
                if essential != prior_essential:
                    raise RuntimeError(f"conflicting duplicate result: {key}")
            else:
                by_key[key] = record
            sources.setdefault(key, []).append(str(path))

    rows = []
    for key, record in sorted(by_key.items()):
        first = record.get("first_divergence")
        summary = (first or {}).get("summary") or {}
        outcome = str((record.get("outcome") or {}).get("classification") or "unknown")
        rows.append({
            "task_id": key[0],
            "candidate_index": key[1],
            "missing_stratum": record.get("missing_stratum", ""),
            "outcome": outcome,
            "success": outcome == "success",
            "first_divergence_observed": bool(record.get("first_divergence_observed")),
            "selector": selector(first),
            "q_argmax": summary.get("selected_is_signed_q_argmax", ""),
            "u_argmax": summary.get("selected_is_u_argmax", ""),
            "q_plus_u_argmax": summary.get("selected_is_signed_q_plus_u_argmax", ""),
            "max_visit_tie_count": summary.get("max_visit_tie_count", ""),
            "source_count": len(sources[key]),
        })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)

    selector_outcome = Counter((row["selector"], row["outcome"]) for row in rows)
    payload = {
        "unique_results": len(rows),
        "divergence_observed": sum(row["first_divergence_observed"] for row in rows),
        "no_divergence": sum(not row["first_divergence_observed"] for row in rows),
        "selector_counts": dict(Counter(row["selector"] for row in rows)),
        "selector_outcomes": {
            f"{selector}|{outcome}": count
            for (selector, outcome), count in sorted(selector_outcome.items())
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

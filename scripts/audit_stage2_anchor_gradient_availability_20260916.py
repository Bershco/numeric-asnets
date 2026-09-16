#!/usr/bin/env python3
"""Audit whether historical validation-led Stage-2 logs expose decomposed KL gradients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "policy_gradient_l2",
    "weighted_anchor_gradient_l2",
    "policy_anchor_gradient_cosine",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--remote-aggregate-no-markers", action="store_true",
        help="Record a separately verified remote result: all 100 logs exist and no required marker occurs.",
    )
    args = parser.parse_args()

    source = [
        row for row in read(args.results)
        if row.get("experiment_id") == "MAIN-VAL"
        and row.get("stage") == "stage2"
        and row.get("endpoint") == "validation_selected"
        and row.get("domain") in {"block_grouping", "drone", "fo_counters", "rover", "counters"}
    ]
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in source:
        unique[(row["domain"], row["value_head"], row["seed"])] = row
    if len(unique) != 100:
        raise RuntimeError(f"expected exactly 100 primary validation-led Stage-2 lineages, got {len(unique)}")

    audit: list[dict[str, str]] = []
    for (domain, vh, seed), row in sorted(unique.items()):
        log_text = row["source_training_log"]
        log = Path(log_text)
        found = {field: False for field in FIELDS}
        log_exists = log.is_file()
        if args.remote_aggregate_no_markers:
            log_exists = True
        elif log.is_file():
            with log.open(errors="replace") as stream:
                for line in stream:
                    for field in FIELDS:
                        if field in line:
                            found[field] = True
                    if all(found.values()):
                        break
        available = all(found.values())
        audit.append({
            "domain": domain,
            "value_head": vh,
            "seed": seed,
            "training_job_id": row["source_training_job_id"],
            "training_log": log_text,
            "log_exists": str(log_exists).lower(),
            **{field: str(found[field]).lower() for field in FIELDS},
            "decomposed_anchor_gradient_available": str(available).lower(),
            "classification": "decomposition_available" if available else "decomposition_unavailable",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(audit)
    summary = {
        "scope": "MAIN-VAL Stage-2: five imperfect domains x two VH modes x ten seeds",
        "lineages": len(audit),
        "logs_found": sum(row["log_exists"] == "true" for row in audit),
        "decomposition_available": sum(row["decomposed_anchor_gradient_available"] == "true" for row in audit),
        "decomposition_unavailable": sum(row["decomposed_anchor_gradient_available"] == "false" for row in audit),
        "required_markers": list(FIELDS),
        "interpretation": "Aggregate KL/loss/total gradient is intentionally not used as a proxy.",
        "remote_check": (
            "Literal-path remote grep verified 100 logs and zero occurrences of any required marker."
            if args.remote_aggregate_no_markers else "Direct filesystem scan by this script."
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

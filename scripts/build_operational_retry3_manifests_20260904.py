#!/usr/bin/env python3
"""Build exact third-wave retries for post-outage node compatibility failures."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
PRESERVE = TRACKING / "four_domain_preservation"

EXCLUDED_NODES = ",".join(
    [
        "ise-cpu128-03",
        "ise-cpu128-04",
        "ise-cpu-intl-08",
        "ise-cpu-intl-09",
        "ise-cpu-intl-10",
        "ise-cpu-intl-13",
        "ise-cpu-intl-14",
        "ise-cpu-intl-26",
        "ise-cpu-intl-28",
    ]
)


def build(
    source: Path,
    output: Path,
    targets: dict[str, str],
    old_suffix: str,
    new_suffix: str,
) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["manifest_id"] in targets]
    if len(rows) != len(targets):
        present = {row["manifest_id"] for row in rows}
        raise RuntimeError(
            f"{output.name}: expected {len(targets)} rows, found {len(rows)}; "
            f"missing={sorted(set(targets) - present)}"
        )
    for row in rows:
        previous = row["manifest_id"]
        if old_suffix and not previous.endswith(old_suffix):
            raise RuntimeError(f"unexpected manifest suffix: {previous}")
        base = previous[: -len(old_suffix)] if old_suffix else previous
        row["manifest_id"] = base + new_suffix
        row["excluded_nodes"] = EXCLUDED_NODES
        row["notes"] += (
            f"; exact operational retry of job {targets[previous]}, which failed "
            "before inference with native exit -4 (SIGILL); exclusions cover only "
            "nodes with directly observed semaphore ENOSPC or SIGILL failures"
        )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


build(
    TRACKING / "stage2_mcts_rover_node_retries_20260904.csv",
    TRACKING / "stage2_mcts_rover_node_retry2_20260904.csv",
    {
        "stage2-gap-validation-rover-off-534933607-w20_i70-node-retry1": "20944963",
        "stage2-gap-validation-rover-off-1239739722-w20_i70-node-retry1": "20944965",
        "stage2-gap-validation-rover-off-1472491096-w20_i70-node-retry1": "20944966",
        "stage2-gap-validation-rover-off-1510771779-w20_i70-node-retry1": "20944967",
        "stage2-gap-validation-rover-off-1963100312-w20_i70-node-retry1": "20944968",
        "stage2-gap-validation-rover-off-2011206605-w20_i70-node-retry1": "20944969",
        "stage2-gap-validation-rover-on-923500475-w20_i70-node-retry1": "20944970",
    },
    "-node-retry1",
    "-node-retry2",
)

build(
    TRACKING / "stage2_mcts_rover_completion_approved_20260904.csv",
    TRACKING / "stage2_mcts_rover_late_node_retries_20260904.csv",
    {
        "stage2-gap-validation-rover-on-1510771779-w20_i70": "20944000",
        "stage2-gap-validation-rover-on-1972442430-w20_i70": "20944002",
        "stage2-gap-validation-rover-on-2011206605-w20_i70": "20944003",
    },
    "",
    "-node-retry1",
)

build(
    TRACKING / "stage2_mcts_branch_completion_fo_node_retry2_20260904.csv",
    TRACKING / "stage2_mcts_branch_completion_fo_node_retry3_20260904.csv",
    {
        "stage2-gap-validation-fo_counters-off-923500475-w20_i70-node-retry2": "20944884",
        "stage2-gap-validation-fo_counters-on-2011206605-w20_i70-node-retry2": "20944886",
    },
    "-node-retry2",
    "-node-retry3",
)

build(
    PRESERVE / "terminal_stage2_tpp_on_policy_node_retry2_20260904.csv",
    PRESERVE / "terminal_stage2_tpp_on_policy_node_retry3_20260904.csv",
    {
        "preserve3-term-tpp-on-2082152039-a10-policy-e0055-node-retry2": "20945014",
        "preserve3-term-tpp-on-2082152039-a10-policy-e0065-node-retry2": "20945015",
        "preserve3-term-tpp-on-2082152039-a10-policy-e0070-node-retry2": "20945016",
    },
    "-node-retry2",
    "-node-retry3",
)

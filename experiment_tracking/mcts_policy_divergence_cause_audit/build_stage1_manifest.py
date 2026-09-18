#!/usr/bin/env python3
"""Build the frozen grouped Stage-1 compact-recorder task manifest.

This script performs local file transforms only.  It never invokes Slurm or
contacts the cluster.  One task may run several exact checkpoint/seed/instance
commands sequentially, which is why grouping by domain/VH/search family does
not imply that candidates share a checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stage0_missing_strata_manifest.csv"
OUTPUT = HERE / "stage1_grouped_tasks.csv"
FREEZE = HERE / "stage1_grouped_tasks.freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
    candidates = list(csv.DictReader(handle))

groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for candidate in candidates:
    groups[candidate["submission_group"]].append(candidate)

expected_fixed = {
    f"{domain}:{value_head}:fixed"
    for domain in ("block_grouping", "counters", "drone", "fo_counters", "rover")
    for value_head in ("off", "on")
}
expected_optional = {
    f"fo_counters:{value_head}:pw" for value_head in ("off", "on")
}
if set(groups) != expected_fixed | expected_optional:
    raise ValueError(
        "Stage-0 task groups changed; review rather than silently refreezing: "
        f"found={sorted(groups)}")

fields = [
    "task_id", "release_class", "submitted", "domain", "value_head",
    "search_family", "search_config", "candidate_count",
    "candidate_runs_json", "recorder_flag", "cpus", "memory_gib",
    "per_instance_timeout_seconds", "allocation_hours", "tie_break",
    "terminal_safe", "release_gate", "source_manifest",
    "source_manifest_sha256",
]
rows = []
source_hash = sha256(SOURCE)
for group_name in sorted(groups):
    group = sorted(
        groups[group_name],
        key=lambda row: (
            row["missing_stratum"], row["candidate_seed"],
            row["candidate_checkpoint_identity"], row["candidate_instance"],
        ),
    )
    domain, value_head, family = group_name.split(":")
    if family == "fixed":
        release_class = "primary_fixed"
        if domain in {"block_grouping", "counters"}:
            # These primary cells are the narrow comparator.
            config = {
                "mcts_expansion_k": 5,
                "mcts_iterations": 20,
                "mcts_exploration_weight": 0.1,
                "estimator_coeff": 0.5,
                "progressive_widening": False,
            }
        else:
            # Drone, FO Counters and Rover primary outcomes came from the
            # historical normal fixed 20-width/70-simulation arm.  Replaying
            # them as 5/20 would not explain the frozen outcome strata.
            config = {
                "mcts_expansion_k": 20,
                "mcts_iterations": 70,
                "mcts_exploration_weight": 0.1,
                "estimator_coeff": 0.5,
                "progressive_widening": False,
            }
        terminal_safe = False
    else:
        release_class = "optional_fo_pw"
        config = {
            "mcts_expansion_k": 20,
            "mcts_iterations": 70,
            "mcts_exploration_weight": 0.1,
            "estimator_coeff": 0.5,
            "progressive_widening": True,
            "pw_min_width": 3,
            "pw_c": 0.6,
            "pw_alpha": 0.5,
        }
        # Match the completed FO-PW70 arm whose result motivated this trace.
        terminal_safe = True

    task_candidates = [
        {
            "missing_stratum": row["missing_stratum"],
            "seed": int(row["candidate_seed"]),
            "checkpoint_identity": row["candidate_checkpoint_identity"],
            "instance": row["candidate_instance"],
            "policy_log": row["candidate_policy_log"],
            "search_log": row["candidate_search_log"],
            "outcome_source": row["candidate_source_artifact"],
        }
        for row in group
    ]
    rows.append({
        "task_id": f"root-trace-{domain}-{value_head}-{family}",
        "release_class": release_class,
        "submitted": "false",
        "domain": domain,
        "value_head": value_head,
        "search_family": family,
        "search_config": json.dumps(
            config, sort_keys=True, separators=(",", ":")),
        "candidate_count": len(task_candidates),
        "candidate_runs_json": json.dumps(
            task_candidates, sort_keys=True, separators=(",", ":")),
        "recorder_flag": "--eval-mcts-first-divergence-record",
        "cpus": 2,
        "memory_gib": 120,
        "per_instance_timeout_seconds": 21600,
        "allocation_hours": 26,
        "tie_break": "action_id",
        "terminal_safe": str(terminal_safe).lower(),
        "release_gate": "known_counters_root_compute_smoke_passed",
        "source_manifest": SOURCE.name,
        "source_manifest_sha256": source_hash,
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

freeze_payload = {
    "schema_version": "mcts-divergence-stage1-task-freeze-v1",
    "source_manifest": SOURCE.name,
    "source_manifest_sha256": source_hash,
    "grouped_manifest": OUTPUT.name,
    "grouped_manifest_sha256": sha256(OUTPUT),
    "primary_fixed_tasks": sum(
        row["release_class"] == "primary_fixed" for row in rows),
    "optional_fo_pw_tasks": sum(
        row["release_class"] == "optional_fo_pw" for row in rows),
    "candidate_runs": sum(int(row["candidate_count"]) for row in rows),
    "submitted": False,
}
with FREEZE.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(freeze_payload, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(json.dumps(freeze_payload, sort_keys=True))

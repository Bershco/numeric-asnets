#!/usr/bin/env python3
"""Build the approved Rover and gated Counters Stage-2 MCTS completion scopes."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiment_tracking" / "stage2_mcts_historical_log_audit_20260902.csv"
RESULTS = ROOT / "experiment_tracking" / "experiment_results.csv"
ROVER_OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_rover_completion_approved_20260904.csv"
COUNTERS_OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_counters_terminal_completion_ready_20260904.csv"

FIELDS = [
    "manifest_id", "cohort", "seed", "domain", "domain_label", "value_head",
    "rq_scope", "task_type", "stage", "variant", "architecture", "teacher",
    "source_checkpoint_ref", "supervised_lr", "max_epochs",
    "original_training_set", "estimator", "puct", "tree_sampling", "anchor",
    "width", "iterations", "workers", "jpddl_heap", "cpus", "memory",
    "time_limit", "instance_timeout", "completion_mode", "dependency_ref",
    "checkpoint_selection", "status", "notes", "source_training_job_id",
    "snapshot_epoch", "analysis_roles", "excluded_nodes",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize_checkpoint(value: str) -> str:
    checkpoint = value.replace("\\", "/")
    return "/" + checkpoint.lstrip("/")


audit = read(AUDIT)
results = read(RESULTS)

endpoints: dict[tuple[str, str, str, str], dict[str, str]] = {}
for row in results:
    if row.get("stage") != "stage2" or row.get("endpoint") != "validation_selected":
        continue
    key = (row["experiment_id"], row["domain"], row["value_head"], row["seed"])
    if key in endpoints:
        raise RuntimeError(f"duplicate selected policy endpoint: {key}")
    endpoints[key] = row


def build(scope: str) -> list[dict[str, str]]:
    if scope == "rover":
        gaps = [
            row for row in audit
            if row["experiment_id"] == "MAIN-VAL"
            and row["domain"] == "rover"
            and row["audit_state"] == "no_candidate_log_found"
        ]
        expected = 19
        experiment_id = "MAIN-VAL"
        branch = "validation"
        teacher = "hmrp-ha-gbfs"
        width, iterations = "20", "70"
        label = "Rover"
        configuration = "normal20/70"
    elif scope == "counters":
        gaps = [
            row for row in audit
            if row["experiment_id"] == "MAIN-TERM"
            and row["domain"] == "counters"
            and row["audit_state"] == "no_candidate_log_found"
        ]
        expected = 20
        experiment_id = "MAIN-TERM"
        branch = "terminal"
        teacher = "hmrmax-astar"
        width, iterations = "5", "20"
        label = "Counters"
        configuration = "narrow5/20"
    else:
        raise ValueError(scope)
    if len(gaps) != expected:
        raise RuntimeError(f"{scope}: expected {expected} gaps, found {len(gaps)}")

    output: list[dict[str, str]] = []
    for gap in gaps:
        key = (experiment_id, scope, gap["value_head"], gap["seed"])
        endpoint = endpoints.get(key)
        if endpoint is None:
            raise RuntimeError(f"missing selected policy endpoint: {key}")
        if endpoint["source_training_job_id"] != gap["source_training_job_id"]:
            raise RuntimeError(f"training-job mismatch: {key}")
        if endpoint["epoch"] != gap["selected_epoch"]:
            raise RuntimeError(f"selected-epoch mismatch: {key}")
        output.append({
            "manifest_id": (
                f"stage2-gap-{branch}-{scope}-{gap['value_head']}-{gap['seed']}-"
                f"w{width}_i{iterations}"
            ),
            "cohort": "stage2-branch-completion",
            "seed": gap["seed"],
            "domain": scope,
            "domain_label": label,
            "value_head": gap["value_head"],
            "rq_scope": "RQ2/RQ4",
            "task_type": "mcts_eval",
            "stage": "stage2",
            "variant": f"branch_completion_{configuration}",
            "architecture": f"experiments_numeric.architecture_2.{scope}_mcts",
            "teacher": teacher,
            "source_checkpoint_ref": normalize_checkpoint(endpoint["checkpoint"]),
            "supervised_lr": "", "max_epochs": "", "original_training_set": "True",
            "estimator": "0.5", "puct": "0.1", "tree_sampling": "", "anchor": "",
            "width": width, "iterations": iterations, "workers": "3",
            "jpddl_heap": "4g", "cpus": "6", "memory": "120G",
            "time_limit": "3-00:00:00", "instance_timeout": "21600",
            "completion_mode": "rolling+VAL",
            "dependency_ref": f"training:{gap['source_training_job_id']}",
            "checkpoint_selection": "validation_selected",
            "status": "ready",
            "notes": (
                f"Approved 2026-09-04 after exhaustive historical-log audit; {branch}-led; "
                f"configuration frozen as {configuration}; submit Counters only after the separate "
                "Rover running-state and memory gate is satisfied"
            ),
            "source_training_job_id": gap["source_training_job_id"],
            "snapshot_epoch": gap["selected_epoch"],
            "analysis_roles": f"stage2_{branch}_selected_policy_vs_mcts_branch_completion",
            "excluded_nodes": "ise-cpu128-03,ise-cpu-intl-13",
        })
    return sorted(output, key=lambda r: (r["value_head"], int(r["seed"])))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


write(ROVER_OUTPUT, build("rover"))
write(COUNTERS_OUTPUT, build("counters"))

#!/usr/bin/env python3
"""Build the explicitly approved BG/FO Stage-2 MCTS branch-completion scope."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiment_tracking" / "stage2_mcts_historical_log_audit_20260902.csv"
RESULTS = ROOT / "experiment_tracking" / "experiment_results.csv"
OUTPUT = ROOT / "experiment_tracking" / "stage2_mcts_branch_completion_approved_20260903.csv"

FIELDS = [
    "manifest_id", "cohort", "seed", "domain", "domain_label", "value_head",
    "rq_scope", "task_type", "stage", "variant", "architecture", "teacher",
    "source_checkpoint_ref", "supervised_lr", "max_epochs",
    "original_training_set", "estimator", "puct", "tree_sampling", "anchor",
    "width", "iterations", "workers", "jpddl_heap", "cpus", "memory",
    "time_limit", "instance_timeout", "completion_mode", "dependency_ref",
    "checkpoint_selection", "status", "notes", "source_training_job_id",
    "snapshot_epoch", "analysis_roles",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


audit = read(AUDIT)
results = read(RESULTS)


def approved(row: dict[str, str]) -> bool:
    if row["audit_state"] != "no_candidate_log_found":
        return False
    if row["domain"] == "block_grouping":
        return row["experiment_id"] == "MAIN-VAL" or (
            row["experiment_id"] == "MAIN-TERM" and row["value_head"] == "on"
        )
    return row["experiment_id"] == "MAIN-VAL" and row["domain"] == "fo_counters"


missing = [row for row in audit if approved(row)]
if len(missing) != 18:
    raise RuntimeError(f"expected 18 approved gaps, found {len(missing)}")

index: dict[tuple[str, str, str, str], dict[str, str]] = {}
for row in results:
    if row.get("stage") != "stage2" or row.get("endpoint") != "validation_selected":
        continue
    key = (row["experiment_id"], row["domain"], row["value_head"], row["seed"])
    if key in index:
        raise RuntimeError(f"duplicate selected policy endpoint: {key}")
    index[key] = row

output = []
for gap in missing:
    key = (gap["experiment_id"], gap["domain"], gap["value_head"], gap["seed"])
    endpoint = index.get(key)
    if endpoint is None:
        raise RuntimeError(f"missing policy endpoint: {key}")
    if endpoint["source_training_job_id"] != gap["source_training_job_id"]:
        raise RuntimeError(f"training-job mismatch: {key}")
    if endpoint["epoch"] != gap["selected_epoch"]:
        raise RuntimeError(f"selected-epoch mismatch: {key}")
    domain = gap["domain"]
    narrow = domain == "block_grouping"
    width, iterations = ("5", "20") if narrow else ("20", "70")
    search = f"w{width}_i{iterations}"
    branch = "validation" if gap["experiment_id"] == "MAIN-VAL" else "terminal"
    checkpoint = endpoint["checkpoint"].replace("\\", "/")
    if not checkpoint.startswith("/"):
        checkpoint = "/" + checkpoint.lstrip("/")
    output.append({
        "manifest_id": (
            f"stage2-gap-{branch}-{domain}-{gap['value_head']}-{gap['seed']}-{search}"
        ),
        "cohort": "stage2-branch-completion",
        "seed": gap["seed"],
        "domain": domain,
        "domain_label": "Block Grouping" if narrow else "FO Counters",
        "value_head": gap["value_head"],
        "rq_scope": "RQ2/RQ4",
        "task_type": "mcts_eval",
        "stage": "stage2",
        "variant": f"branch_completion_{'narrow' if narrow else 'normal'}_{search}",
        "architecture": f"experiments_numeric.architecture_2.{domain}_mcts",
        "teacher": "hadd-gbfs" if narrow else "hmrmax-astar",
        "source_checkpoint_ref": checkpoint,
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
            "Explicitly approved 2026-09-03 after exhaustive historical-log audit; "
            f"{branch}-led branch; Block Grouping is always narrow5/20 and FO is normal20/70"
        ),
        "source_training_job_id": gap["source_training_job_id"],
        "snapshot_epoch": gap["selected_epoch"],
        "analysis_roles": f"stage2_{branch}_selected_policy_vs_mcts_branch_completion",
    })

counts = {}
for row in output:
    key = (row["domain"], row["value_head"], row["width"], row["iterations"])
    counts[key] = counts.get(key, 0) + 1
expected = {
    ("block_grouping", "off", "5", "20"): 10,
    ("block_grouping", "on", "5", "20"): 3,
    ("fo_counters", "off", "20", "70"): 4,
    ("fo_counters", "on", "20", "70"): 1,
}
if counts != expected:
    raise RuntimeError(f"unexpected approved scope: {counts}")

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
print(f"wrote {len(output)} rows to {OUTPUT}")
print(counts)

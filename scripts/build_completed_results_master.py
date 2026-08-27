#!/usr/bin/env python3
"""Combine static policy endpoints and Stage-1 MCTS outcomes into one ledger."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
TMP = ROOT / ".codex-tmp"
SNAPSHOT_EPOCH_RE = re.compile(r"(?:^|[\\/])snapshot_(\d+)(?:_|$)")


def checkpoint_epoch(checkpoint: str) -> str:
    """Extract the saved epoch from a checkpoint path without reopening logs."""
    match = SNAPSHOT_EPOCH_RE.search(checkpoint or "")
    return str(int(match.group(1))) if match else ""


def read(path, delimiter=","):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main():
    policy = read(TRACK / "policy_endpoint_results.csv")
    mcts = read(TRACK / "stage1_mcts_results.csv")
    ledger = read(TMP / "health_20260826" / "main_ledger_current.tsv", "\t")
    source = {row["slurm_job_id"]: row for row in ledger if row.get("slurm_job_id")}
    stage1_sources = {}
    for item in ledger:
        manifest_id = item.get("manifest_id", "")
        if (
            item.get("task_type") != "mcts_eval"
            or "stage1" not in manifest_id
            or "primary" not in manifest_id
            or "terminal" in manifest_id
        ):
            continue
        key = (item.get("domain"), item.get("value_head"), item.get("seed"))
        stage1_sources.setdefault(key, []).append(item)
    rows = []
    fields = [
        "experiment_id", "task_type", "domain", "value_head", "stage", "endpoint",
        "seed", "source_training_job_id", "checkpoint", "epoch", "job_id", "score", "total",
        "slurm_state", "exit_code", "elapsed", "width", "iterations", "puct", "estimator",
        "ordinary_failures", "instance_timeouts", "unprocessed_after_job_end",
        "val_valid", "val_invalid", "source_training_log", "source_evaluation_log",
        "source_validation_log", "alternate_valid_job_ids", "alternate_valid_log_paths", "notes"
    ]
    for row in policy:
        rows.append({
            "experiment_id": row["experiment_id"], "task_type": "policy_eval",
            "domain": row["domain"], "value_head": row["value_head"],
            "stage": row["stage"], "endpoint": row["endpoint"], "seed": row["seed"],
            "source_training_job_id": row["source_training_job_id"],
            "checkpoint": row["checkpoint"],
            "epoch": checkpoint_epoch(row["checkpoint"]),
            "job_id": row["evaluation_job_id"],
            "score": row["score"], "total": row["total"],
            "slurm_state": row["evaluation_state"], "exit_code": row["evaluation_exit_code"],
            "elapsed": row["evaluation_elapsed"], "val_valid": row["val_valid"],
            "val_invalid": row["val_invalid"], "source_training_log": row["source_training_log"],
            "source_evaluation_log": row["source_evaluation_log"],
            "source_validation_log": row["source_validation_log"],
            "alternate_valid_job_ids": row["alternate_valid_job_ids"],
            "alternate_valid_log_paths": row["alternate_valid_log_paths"],
            "notes": "Earliest VAL-valid original endpoint evaluation; alternate valid retries retained",
        })
    for row in mcts:
        src = source.get(row["mcts_job_id"])
        if src is None:
            # Corrected/retry MCTS jobs may deliberately replace an obsolete
            # evaluation while reusing the exact same Stage-1 checkpoint.  The
            # controller ledger predates those retry Slurm IDs, so resolve the
            # immutable checkpoint by the experiment key and refuse ambiguity.
            key = (row["domain"], row["value_head"], row["seed"])
            candidates = stage1_sources.get(key, [])
            checkpoints = {item.get("source_checkpoint") for item in candidates}
            if len(checkpoints) != 1:
                raise KeyError(
                    f"Cannot resolve one Stage-1 checkpoint for retry job "
                    f"{row['mcts_job_id']} key={key}: {sorted(checkpoints)}"
                )
            src = candidates[0]
        validation_log = row["source_evaluation_log"] if row["inline_val_agrees"] == "true" else row["posthoc_val_summary"]
        rows.append({
            "experiment_id": "MAIN-VAL", "task_type": "mcts_eval",
            "domain": row["domain"], "value_head": row["value_head"],
            "stage": "stage1", "endpoint": "validation_selected", "seed": row["seed"],
            "source_training_job_id": src["source_training_job_id"],
            "checkpoint": src["source_checkpoint"],
            "epoch": checkpoint_epoch(src["source_checkpoint"]),
            "job_id": row["mcts_job_id"],
            "score": row["successes"], "total": row["total_instances"],
            "slurm_state": row["slurm_state"], "exit_code": row["exit_code"],
            "elapsed": row["elapsed"], "width": row["width"],
            "iterations": row["iterations"], "puct": row["puct"],
            "estimator": row["estimator"], "ordinary_failures": row["ordinary_failures"],
            "instance_timeouts": row["instance_timeouts"],
            "unprocessed_after_job_end": row["unprocessed_after_job_end"],
            "val_valid": row["inline_val_valid"] or row["posthoc_val_valid"],
            "val_invalid": row["inline_val_invalid"] or row["posthoc_val_invalid"],
            "source_evaluation_log": row["source_evaluation_log"],
            "source_validation_log": validation_log,
            "notes": row["notes"],
        })
    with (TRACK / "experiment_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"policy={len(policy)} mcts={len(mcts)} total={len(rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize downstream manifests for completed four-domain stage-1 jobs."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


DOMAINS = {
    "delivery": "hadd-astar",
    "mprime": "hmrp-ha-gbfs",
    "tpp": "hadd-gbfs",
    "zenotravel": "hadd-gbfs",
}
FINAL_RE = re.compile(r"Last valid checkpoint is (.+/snapshot_(\d+)_[^\s]+)")
BEST_RE = re.compile(
    r"\[VALIDATION\] New best reached! .*?iteration (\d+) .*?snapshot name: (snapshot_\d+_[^\]]+)"
)
FIELDS = [
    "manifest_id", "cohort", "seed", "domain", "domain_label", "value_head",
    "rq_scope", "task_type", "stage", "variant", "architecture", "teacher",
    "source_checkpoint_ref", "supervised_lr", "max_epochs", "original_training_set",
    "estimator", "puct", "tree_sampling", "anchor", "width", "iterations",
    "workers", "jpddl_heap", "cpus", "memory", "time_limit",
    "instance_timeout", "completion_mode", "dependency_ref",
    "checkpoint_selection", "status", "notes", "source_training_job_id",
    "snapshot_epoch", "analysis_roles",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def completed(ids: list[str]) -> dict[str, str]:
    output = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
         "--format=JobIDRaw,State,StdOut"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    result = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, state, stdout, *_ = line.split("|")
        if state == "COMPLETED":
            result[job_id] = stdout.replace("%j", job_id)
    return result


def checkpoints(log: Path) -> tuple[Path, int, Path]:
    final_match = best_match = None
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            final_match = FINAL_RE.search(line) or final_match
            best_match = BEST_RE.search(line) or best_match
    if final_match is None or best_match is None:
        raise RuntimeError(f"Missing final/validation markers in {log}")
    final = Path(final_match.group(1))
    selected_epoch = int(best_match.group(1))
    selected = final.parent / best_match.group(2)
    if not final.is_dir() or not selected.is_dir():
        raise RuntimeError(f"Missing checkpoint for {log}: final={final} selected={selected}")
    return final, selected_epoch, selected


def snapshots(final: Path) -> dict[int, Path]:
    found = {}
    for path in final.parent.glob("snapshot_*_*"):
        match = re.match(r"snapshot_(\d+)_", path.name)
        if match:
            found[int(match.group(1))] = path
    final_epoch = int(re.match(r"snapshot_(\d+)_", final.name).group(1))
    wanted = {epoch for epoch in found if epoch % 5 == 0}
    wanted.add(final_epoch)
    return {epoch: found[epoch] for epoch in sorted(wanted)}


def base(train: dict[str, str], task: str, checkpoint: Path, epoch: int) -> dict[str, str]:
    domain = train["domain"]
    vh = train["value_head"]
    is_mcts = task == "mcts_eval"
    return {
        "cohort": "completed-domains-ten-seed", "seed": train["seed"],
        "domain": domain, "domain_label": domain.replace("_", " ").title(),
        "value_head": vh, "rq_scope": "preservation-followup", "task_type": task,
        "stage": "stage1", "architecture": f"experiments_numeric.architecture_2.{domain}" + ("_mcts" if is_mcts else ""),
        "teacher": DOMAINS[domain], "source_checkpoint_ref": str(checkpoint),
        "supervised_lr": "", "max_epochs": "", "original_training_set": "True",
        "estimator": "0.5" if is_mcts else "", "puct": "0.1" if is_mcts else "",
        "tree_sampling": "", "anchor": "", "width": "20" if is_mcts else "",
        "iterations": "70" if is_mcts else "", "workers": "3" if is_mcts else "10",
        "jpddl_heap": "4g", "cpus": "6", "memory": "120G" if is_mcts else "20G",
        "time_limit": "3-00:00:00" if is_mcts else "04:00:00",
        "instance_timeout": "21600" if is_mcts else "", "completion_mode": "rolling+VAL" if is_mcts else "VAL",
        "dependency_ref": train["manifest_id"], "status": "ready",
        "source_training_job_id": train["slurm_job_id"], "snapshot_epoch": str(epoch),
    }


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [row["manifest_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate manifest IDs in {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trains = read_tsv(args.training_ledger)
    states = completed([row["slurm_job_id"] for row in trains])
    policies, mcts, stage2_plan = [], [], []
    for train in trains:
        stdout = states.get(train["slurm_job_id"])
        if not stdout:
            continue
        final, selected_epoch, selected = checkpoints(Path(stdout))
        curve = snapshots(final)
        prefix = f"cd4-{train['domain']}-{train['value_head']}-{train['seed']}-stage1"
        for epoch, checkpoint in curve.items():
            row = base(train, "policy_eval", checkpoint, epoch)
            row.update({
                "manifest_id": f"{prefix}-policy-curve-e{epoch:04d}",
                "variant": f"curve_epoch_{epoch:04d}",
                "checkpoint_selection": "every-five stage-1 learning curve; never test-selected",
                "notes": "Stage-1 policy learning-curve evaluation; final epoch included",
                "analysis_roles": "stage1_learning_curve" + (";stage1_final_policy" if checkpoint == final else ""),
            })
            policies.append(row)
        if selected_epoch not in curve:
            row = base(train, "policy_eval", selected, selected_epoch)
            row.update({
                "manifest_id": f"{prefix}-policy-validation-selected",
                "variant": "validation_selected", "checkpoint_selection": "validation-selected stage-1 checkpoint",
                "notes": "Exact policy counterpart of validation-selected MCTS",
                "analysis_roles": "stage1_validation_selected_policy",
            })
            policies.append(row)
        endpoints = [("validation-selected", selected_epoch, selected)]
        final_epoch = int(re.match(r"snapshot_(\d+)_", final.name).group(1))
        if final != selected:
            endpoints.append(("final", final_epoch, final))
        for variant, epoch, checkpoint in endpoints:
            row = base(train, "mcts_eval", checkpoint, epoch)
            row.update({
                "manifest_id": f"{prefix}-mcts-{variant}", "variant": variant,
                "checkpoint_selection": "validation-selected stage-1 checkpoint" if variant == "validation-selected" else "terminal stage-1 checkpoint",
                "notes": "Held MCTS endpoint; not submitted by the policy controller",
                "analysis_roles": f"stage1_{variant}_mcts",
            })
            mcts.append(row)
        for branch, checkpoint, epoch in (
            ("validation-led", selected, selected_epoch), ("terminal-led", final, final_epoch)
        ):
            stage2_plan.append({
                "manifest_id": f"cd4-{train['domain']}-{train['value_head']}-{train['seed']}-stage2-train-{branch}",
                "cohort": "completed-domains-ten-seed", "seed": train["seed"], "domain": train["domain"],
                "domain_label": train["domain"].replace("_", " ").title(), "value_head": train["value_head"],
                "rq_scope": "preservation-followup", "task_type": "train", "stage": "stage2", "variant": branch,
                "architecture": f"experiments_numeric.architecture_2.{train['domain']}_mcts", "teacher": DOMAINS[train["domain"]],
                "source_checkpoint_ref": str(checkpoint), "supervised_lr": "0.0003", "max_epochs": "100",
                "original_training_set": "True", "estimator": "0.5", "puct": "0.1", "tree_sampling": "0",
                "anchor": "", "width": "20", "iterations": "0", "workers": "3", "jpddl_heap": "4g",
                "cpus": "6", "memory": "48G", "time_limit": "3-00:00:00", "instance_timeout": "",
                "completion_mode": "validation+terminal", "dependency_ref": train["manifest_id"],
                "checkpoint_selection": branch, "status": "blocked_anchor_selection",
                "notes": "Tree sampling and estimator fixed; anchor coefficient must be declared before submission",
                "source_training_job_id": train["slurm_job_id"], "snapshot_epoch": str(epoch),
                "analysis_roles": f"stage2_{branch}_training",
            })
    write(args.output_dir / "completed_domains_stage1_policy_ready.csv", policies)
    write(args.output_dir / "completed_domains_stage1_mcts_held.csv", mcts)
    write(args.output_dir / "completed_domains_stage2_training_blocked.csv", stage2_plan)
    print(f"completed_lineages={len(states)} policy_rows={len(policies)} mcts_rows={len(mcts)} stage2_plan_rows={len(stage2_plan)}")


if __name__ == "__main__":
    main()

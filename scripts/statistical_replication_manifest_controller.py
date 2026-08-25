#!/usr/bin/env python3
"""Submit statistical-replication manifests in evaluation-then-training order."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")
SUBMITTER = ROOT / "submit_training.sh"
JOB_ID_RE = re.compile(r"\[OK \] job=\s*(\d+)")
DOMAINS = {"block_grouping", "counters", "drone", "fo_counters", "rover"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_ledger(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["manifest_id"]: row
            for row in csv.DictReader(stream, delimiter="\t")
            if row.get("slurm_job_id")
        }


def append_ledger(path: Path, row: dict[str, str], job_id: str) -> None:
    fields = [
        "manifest_id", "task_type", "domain", "value_head", "seed",
        "source_training_job_id", "snapshot_epoch", "slurm_job_id",
        "submitted_at", "source_checkpoint",
    ]
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({
            "manifest_id": row["manifest_id"],
            "task_type": row["task_type"],
            "domain": row["domain"],
            "value_head": row["value_head"],
            "seed": row["seed"],
            "source_training_job_id": row["source_training_job_id"],
            "snapshot_epoch": row["snapshot_epoch"],
            "slurm_job_id": job_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": row["source_checkpoint_ref"],
        })
        stream.flush()
        os.fsync(stream.fileno())


def seed_retry_completion(row: dict[str, str], job_id: str) -> None:
    old_job_id = row.get("retry_of_eval_job_id", "").strip()
    if not old_job_id:
        return
    matches = list(ROOT.glob(
        f"*/statistical_replication_stage1_mcts_eval/.resume_state/"
        f"{old_job_id}.eval_completed.jsonl"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"{row['manifest_id']}: expected one completion file for "
            f"{old_job_id}, found {matches}"
        )
    target = (
        ROOT / datetime.now().astimezone().date().isoformat()
        / output_subdir(row) / ".resume_state"
        / f"{job_id}.eval_completed.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], target)
    print(
        f"[CONTROLLER] seeded_completion={target} source={matches[0]}",
        flush=True,
    )


def integer(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{row.get('manifest_id')}: invalid integer {field}={row.get(field)!r}") from exc


def number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{row.get('manifest_id')}: invalid number {field}={row.get(field)!r}") from exc


def validate_row(row: dict[str, str], expected_task: str) -> None:
    mid = row.get("manifest_id", "")
    if not mid:
        raise ValueError("Manifest row lacks manifest_id")
    if row.get("task_type") != expected_task:
        raise ValueError(f"{mid}: expected task_type={expected_task}, got {row.get('task_type')}")
    if row.get("status") != "ready":
        raise ValueError(f"{mid}: status must be ready")
    if row.get("domain") not in DOMAINS:
        raise ValueError(f"{mid}: invalid domain {row.get('domain')}")
    if row.get("value_head") not in {"on", "off"}:
        raise ValueError(f"{mid}: invalid value_head")
    integer(row, "seed")
    integer(row, "source_training_job_id")
    integer(row, "snapshot_epoch")
    checkpoint = Path(row["source_checkpoint_ref"])
    if not checkpoint.exists():
        raise ValueError(f"{mid}: missing checkpoint {checkpoint}")

    policy_arch = f"experiments_numeric.architecture_2.{row['domain']}"
    mcts_arch = policy_arch + "_mcts"
    if expected_task == "policy_eval":
        expected_arch = policy_arch if row.get("stage") == "stage1" else mcts_arch
        if row.get("stage") not in {"stage1", "stage2"}:
            raise ValueError(f"{mid}: invalid policy-evaluation stage")
        if row["architecture"] != expected_arch or integer(row, "workers") != 10:
            raise ValueError(f"{mid}: invalid policy architecture/workers")
        if row["memory"] != "20G" or row["time_limit"] != "04:00:00":
            raise ValueError(f"{mid}: invalid policy resources")
    elif expected_task == "mcts_eval":
        if row.get("stage") not in {"stage1", "stage2"}:
            raise ValueError(f"{mid}: invalid MCTS-evaluation stage")
        if row["architecture"] != mcts_arch or integer(row, "workers") != 3:
            raise ValueError(f"{mid}: invalid MCTS architecture/workers")
        if row["memory"] not in {"120G", "160G"} or row["time_limit"] != "3-00:00:00":
            raise ValueError(f"{mid}: invalid MCTS resources")
        if number(row, "estimator") != 0.5 or number(row, "puct") <= 0:
            raise ValueError(f"{mid}: invalid MCTS estimator/PUCT")
        if integer(row, "width") <= 0 or integer(row, "iterations") <= 0:
            raise ValueError(f"{mid}: invalid MCTS width/iterations")
        if integer(row, "instance_timeout") != 21600:
            raise ValueError(f"{mid}: MCTS timeout must be six hours")
    elif expected_task == "train":
        if row.get("stage") != "stage2" or row["architecture"] != mcts_arch:
            raise ValueError(f"{mid}: invalid stage-2 architecture")
        if integer(row, "workers") != 3 or row["memory"] != "48G":
            raise ValueError(f"{mid}: invalid stage-2 workers/resources")
        if row["time_limit"] != "3-00:00:00" or integer(row, "max_epochs") != 100:
            raise ValueError(f"{mid}: invalid stage-2 time/epochs")
        if number(row, "supervised_lr") != 0.0003:
            raise ValueError(f"{mid}: invalid stage-2 learning rate")
        if number(row, "estimator") != 0.5 or number(row, "puct") != 0.1:
            raise ValueError(f"{mid}: invalid stage-2 estimator/PUCT")
        if integer(row, "tree_sampling") != 0 or integer(row, "width") != 20:
            raise ValueError(f"{mid}: invalid stage-2 tree/width")
        if integer(row, "iterations") != 0:
            raise ValueError(f"{mid}: stage-2 iterations must use dynamic default (0)")


def suffix(row: dict[str, str]) -> str:
    task = {"policy_eval": "P", "mcts_eval": "M", "train": "T"}[row["task_type"]]
    cohort = {
        "terminal-checkpoint-ten": "TC",
        "long-training-side": "LONG",
    }.get(row.get("cohort", ""), "")
    return f"SR10{cohort}{task}_src{row['source_training_job_id']}_e{int(row['snapshot_epoch']):04d}"


def output_subdir(row: dict[str, str]) -> str:
    cohort = row.get("cohort", "")
    if cohort == "terminal-checkpoint-ten":
        return f"statistical_replication_terminal_{row['stage']}_{row['task_type']}"
    if cohort == "long-training-side":
        return f"statistical_replication_stage1_long_{row['task_type']}"
    return f"statistical_replication_{row['stage']}_{row['task_type']}"


def command(row: dict[str, str], dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
    task = row["task_type"]
    cmd = [
        str(SUBMITTER), f"--dom-{row['domain']}", "--original-only",
        "--seed", row["seed"], "--workers", row["workers"],
        "--jpddl-max-heap", row["jpddl_heap"], "--time", row["time_limit"],
        "--mem", row["memory"], "--cpus", row["cpus"],
        "--job-suffix", suffix(row),
    ]
    if row["value_head"] == "off":
        cmd.append("--vh-off")
    if task == "policy_eval":
        cmd += [
            "--domain-architecture", "policy",
            "--eval-from", row["source_checkpoint_ref"],
            "--output-subdir", output_subdir(row),
        ]
    elif task == "mcts_eval":
        cmd += [
            "--domain-architecture", "mcts",
            "--eval-from", row["source_checkpoint_ref"], "--eval-with-mcts",
            "--use-estimator", row["estimator"],
            "--exploration-weight", row["puct"],
            "--mcts-expansion-size", row["width"],
            "--mcts-iterations", row["iterations"],
            "--eval-scheduling", "rolling",
            "--eval-instance-timeout", row["instance_timeout"],
            "--output-subdir", output_subdir(row),
        ]
    else:
        cmd += [
            "--domain-architecture", "mcts",
            "--train-from", row["source_checkpoint_ref"],
            "--use-estimator", row["estimator"],
            "--exploration-weight", row["puct"],
            "--override-tree-sampling", row["tree_sampling"],
            "--mcts-expansion-size", row["width"],
            "--mcts-iterations", row["iterations"],
            "--policy-anchor-kl-coeff", row["anchor"],
            "--max-opt-epochs", row["max_epochs"],
            "--supervised-lr", row["supervised_lr"],
            "--output-subdir", output_subdir(row),
        ]
    if dry_run:
        cmd.append("--dry-run")
    env = os.environ.copy()
    env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
    return cmd, env


def queued_count() -> int:
    result = subprocess.run(
        ["squeue", "-u", "hersco", "-h", "-t", "RUNNING,PENDING"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return sum(bool(line.strip()) for line in result.stdout.splitlines())


def run_wrapper(row: dict[str, str], dry_run: bool = False) -> str:
    cmd, env = command(row, dry_run=dry_run)
    result = subprocess.run(
        cmd, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(f"{row['manifest_id']}: wrapper failed:\n{result.stdout}")
    if dry_run:
        if "Expected jobs: 1" not in result.stdout or "[DRY]" not in result.stdout:
            raise RuntimeError(f"{row['manifest_id']}: invalid dry-run output:\n{result.stdout}")
        return "DRY"
    ids = JOB_ID_RE.findall(result.stdout)
    if len(ids) != 1:
        raise RuntimeError(f"{row['manifest_id']}: expected one job ID, got {ids}:\n{result.stdout}")
    return ids[0]


def signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in (
        "task_type", "domain", "value_head", "architecture", "teacher",
        "workers", "jpddl_heap", "memory", "time_limit", "estimator",
        "puct", "tree_sampling", "anchor", "width", "iterations",
        "instance_timeout",
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mcts-manifest", type=Path, action="append", required=True)
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue-cap", type=int, default=1999)
    parser.add_argument("--max-per-cycle", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    phases = [
        ("policy_eval", args.policy_manifest),
        ("mcts_eval", args.mcts_manifest),
    ]
    if args.training_manifest:
        phases.append(("train", args.training_manifest))
    rows_by_phase: list[tuple[str, Path, list[dict[str, str]]]] = []
    all_ids: list[str] = []
    for task, paths in phases:
        rows: list[dict[str, str]] = []
        for path in paths:
            manifest_rows = read_csv(path)
            if not manifest_rows:
                raise ValueError(f"Empty manifest: {path}")
            for row in manifest_rows:
                validate_row(row, task)
            rows.extend(manifest_rows)
        all_ids.extend(row["manifest_id"] for row in rows)
        rows_by_phase.append((task, paths[0], rows))
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate manifest_id across phases")

    print(
        "[VALID] " + " ".join(f"{task}={len(rows)}" for task, _, rows in rows_by_phase)
        + f" total={len(all_ids)} unique={len(set(all_ids))}", flush=True,
    )
    representatives: dict[tuple[str, ...], dict[str, str]] = {}
    for _, _, rows in rows_by_phase:
        for row in rows:
            representatives.setdefault(signature(row), row)
    print(f"[VALID] wrapper dry-run signatures={len(representatives)}", flush=True)
    for row in representatives.values():
        run_wrapper(row, dry_run=True)
    print("[VALID] all representative wrapper dry-runs passed", flush=True)
    if args.validate_only:
        return 0

    while True:
        submitted = read_ledger(args.ledger)
        remaining_total = 0
        active_phase: tuple[str, list[dict[str, str]]] | None = None
        for task, _, rows in rows_by_phase:
            remaining = [row for row in rows if row["manifest_id"] not in submitted]
            remaining_total += len(remaining)
            if active_phase is None and remaining:
                active_phase = (task, remaining)
        if active_phase is None:
            print(f"[CONTROLLER] complete submitted={len(submitted)}", flush=True)
            return 0

        task, remaining = active_phase
        queue = queued_count()
        allowance = min(
            max(0, args.queue_cap - queue), args.max_per_cycle, len(remaining)
        )
        print(
            f"[CONTROLLER] phase={task} queue={queue} submitted={len(submitted)} "
            f"remaining_total={remaining_total} phase_remaining={len(remaining)} "
            f"allowance={allowance}", flush=True,
        )
        for row in remaining[:allowance]:
            try:
                job_id = run_wrapper(row)
            except RuntimeError as exc:
                # The queue count and sbatch admission are not atomic.  Another
                # controller can fill the final slot after queued_count() but
                # before this submission.  Do not fail or record the row: wait
                # for the next cycle and retry the identical manifest entry.
                if "QOSMaxSubmitJobPerUserLimit" not in str(exc):
                    raise
                print(
                    "[CONTROLLER] transient QOS submit cap; "
                    f"will retry manifest_id={row['manifest_id']}",
                    flush=True,
                )
                break
            append_ledger(args.ledger, row, job_id)
            seed_retry_completion(row, job_id)
            print(
                f"[CONTROLLER] submitted={job_id} manifest_id={row['manifest_id']}",
                flush=True,
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        raise

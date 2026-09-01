#!/usr/bin/env python3
"""Read-only Slurm and artifact audit for recovery after a cluster outage.

The script deliberately cannot submit, cancel, release, requeue, move, or edit
jobs.  It emits evidence and recommended actions for a later reviewed recovery.
Run it from a cluster login node after Slurm and the experiment filesystem have
returned.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "experiment_tracking" / "pre_outage_job_inventory_20260901_1037.csv"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EVAL_FINAL_RE = re.compile(r"\[EVAL FINAL\].*?success(?:es)?\s*[=:]\s*(\d+)(?:\s*/\s*(\d+))?", re.I)
PLAN_RE = re.compile(r"\[EVAL\]\[PLAN\].*?(?:instance|problem)\s*[=:]\s*([^\s,;]+)", re.I)
EPOCH_RE = re.compile(r"\bepoch\s*:\s*.*?\b(\d+)\s*/\s*(\d+)\b", re.I)
SNAPSHOT_DIR_RE = re.compile(r"Snapshot directory:\s*(.+)")
LAST_CHECKPOINT_RE = re.compile(r"Last valid checkpoint is\s+(.+)")
COMPLETION_FILE_RE = re.compile(r"--eval-completion-file(?:=|\s+)([^\s]+)")
OUTAGE_STATES = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED"}
ABNORMAL_STATES = OUTAGE_STATES | {"CANCELLED", "FAILED", "DEADLINE"}

SACCT_FIELDS = [
    "JobIDRaw", "JobName", "State", "ExitCode", "Elapsed", "Timelimit",
    "AllocCPUS", "ReqMem", "Start", "End", "NodeList", "Reason", "StdOut", "WorkDir",
]
OUTPUT_FIELDS = [
    "audit_time", "experiment_id", "component", "job_id", "manifest_id",
    "expected_pre_outage_state", "current_state", "reason", "exit_code",
    "elapsed", "time_limit", "remaining_seconds", "recommended_continuation_time",
    "cpus", "requested_memory", "node_list", "start", "end", "stdout",
    "stdout_exists", "latest_epoch", "target_epoch", "latest_checkpoint",
    "eval_final_success", "eval_final_total", "printed_plan_instances",
    "completion_records", "classification", "recommended_action", "notes",
]


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def chunks(values: list[str], size: int = 80) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def parse_duration(value: str) -> int | None:
    value = (value or "").strip()
    if not value or value.upper() in {"UNLIMITED", "NOT_SET", "N/A", "UNKNOWN"}:
        return None
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def tail_text(path: Path, head_bytes: int = 512_000, tail_bytes: int = 4_000_000) -> str:
    """Read bounded evidence from a possibly multi-gigabyte log."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(min(head_bytes, size))
        if size > head_bytes:
            stream.seek(max(head_bytes, size - tail_bytes))
            tail = stream.read(tail_bytes)
        else:
            tail = b""
    return ANSI_RE.sub("", (head + b"\n" + tail).decode("utf-8", errors="replace"))


def resolve_stdout(value: str, job_id: str) -> Path | None:
    if not value or value in {"Unknown", "None", "(null)"}:
        return None
    return Path(value.replace("%j", job_id).replace("%A", job_id))


def highest_checkpoint(snapshot_dir: Path | None) -> tuple[str, int | None]:
    if snapshot_dir is None or not snapshot_dir.is_dir():
        return "", None
    candidates: list[tuple[int, Path]] = []
    for path in snapshot_dir.glob("snapshot_*"):
        match = re.match(r"snapshot_(\d+)(?:_|$)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return "", None
    epoch, path = max(candidates, key=lambda item: item[0])
    return str(path), epoch


def parse_artifacts(stdout: Path | None) -> dict[str, object]:
    result: dict[str, object] = {
        "stdout_exists": False,
        "latest_epoch": None,
        "target_epoch": None,
        "latest_checkpoint": "",
        "eval_final_success": None,
        "eval_final_total": None,
        "printed_plan_instances": 0,
        "completion_records": 0,
    }
    if stdout is None or not stdout.is_file():
        return result
    result["stdout_exists"] = True
    text = tail_text(stdout)

    epochs = EPOCH_RE.findall(text)
    if epochs:
        result["latest_epoch"] = int(epochs[-1][0])
        result["target_epoch"] = int(epochs[-1][1])

    checkpoint_markers = LAST_CHECKPOINT_RE.findall(text)
    if checkpoint_markers:
        result["latest_checkpoint"] = checkpoint_markers[-1].strip()
    snapshot_markers = SNAPSHOT_DIR_RE.findall(text)
    snapshot_dir = Path(snapshot_markers[-1].strip()) if snapshot_markers else None
    checkpoint, checkpoint_epoch = highest_checkpoint(snapshot_dir)
    if checkpoint and (
        not result["latest_checkpoint"]
        or checkpoint_epoch is not None and checkpoint_epoch >= (result["latest_epoch"] or -1)
    ):
        result["latest_checkpoint"] = checkpoint
        if result["latest_epoch"] is None:
            result["latest_epoch"] = checkpoint_epoch

    final_matches = EVAL_FINAL_RE.findall(text)
    if final_matches:
        success, total = final_matches[-1]
        result["eval_final_success"] = int(success)
        result["eval_final_total"] = int(total) if total else None
    result["printed_plan_instances"] = len(set(PLAN_RE.findall(text)))

    completion_markers = COMPLETION_FILE_RE.findall(text)
    if completion_markers:
        completion_path = Path(completion_markers[-1].strip("\"'"))
        if completion_path.is_file():
            seen: set[str] = set()
            with completion_path.open(encoding="utf-8", errors="replace") as stream:
                for raw in stream:
                    try:
                        payload = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    identity = str(payload.get("instance") or payload.get("problem") or raw)
                    seen.add(identity)
            result["completion_records"] = len(seen)
    return result


def classify(
    *, expected: str, artifact_kind: str, state: str, reason: str,
    has_checkpoint: bool, eval_final: bool, partial_eval: bool,
) -> tuple[str, str]:
    state = re.split(r"[ +]", state.upper().strip(), maxsplit=1)[0]
    expected = expected.lower()
    reason_lower = reason.lower()

    if "deliberately_held" in expected:
        if state == "PENDING":
            return "held_intact", "keep_held_and_review_state"
        if state == "":
            return "held_missing_from_scheduler", "recreate_as_held_only_after_accounting_review"
        return "held_state_changed", "keep_held_and_review_state"
    if "operationally_held" in expected:
        return "operational_hold", "inspect_environment_failure_then_release_or_resubmit"
    if state in {"RUNNING", "COMPLETING"}:
        return "survived_or_restarted", "continue_monitoring_without_mutation"
    if state == "PENDING":
        if "dependency" in reason_lower or artifact_kind == "controller":
            return "dependency_pending", "verify_dependency_ids_and_leave_pending"
        return "scheduler_pending", "leave_pending_unless_scheduler_lost_the_job"
    if state == "COMPLETED":
        if artifact_kind == "training" and has_checkpoint:
            return "completed_with_checkpoint", "preserve_and_run_missing_downstream_controller"
        if artifact_kind.endswith("evaluation") and eval_final:
            return "completed_with_final_result", "preserve_static_result_and_validate_provenance"
        if artifact_kind == "controller":
            return "controller_completed", "verify_idempotent_outputs"
        return "completed_needs_artifact_check", "inspect_log_and_output_before_any_rerun"
    if "terminal" in expected and state == "":
        return "preoutage_terminal_accounting_missing", "preserve_local_static_result_and_recover_accounting"
    if state in {"TIMEOUT", "OUT_OF_MEMORY"}:
        if artifact_kind == "training" and has_checkpoint:
            return "declared_training_endpoint", "preserve_checkpoint_as_protocol_terminal_endpoint"
        if artifact_kind.endswith("evaluation") and (eval_final or partial_eval):
            return "fixed_budget_terminal_result", "preserve_partial_score_and_posthoc_val_printed_plans"
        return "resource_terminal_without_evidence", "inspect_before_resource_sensitivity_rerun"
    if state in ABNORMAL_STATES or state == "":
        if artifact_kind == "training" and has_checkpoint:
            return "interrupted_resumable_training", "resume_from_latest_checkpoint_after_review"
        if artifact_kind == "mcts_evaluation" and (eval_final or partial_eval):
            return "interrupted_resumable_mcts", "resume_from_completion_jsonl_then_posthoc_val"
        if artifact_kind == "policy_evaluation":
            return "interrupted_policy_evaluation", "idempotently_resubmit_policy_evaluation"
        if artifact_kind == "controller":
            return "interrupted_controller", "idempotently_rerun_controller_after_dependency_audit"
        return "interrupted_no_recovery_artifact", "manual_log_review_before_clean_rerun"
    return "unclassified", "manual_review"


def collect_squeue(job_ids: list[str]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    result: dict[str, dict[str, str]] = {}
    all_rows: list[dict[str, str]] = []
    wanted = set(job_ids)
    # Querying old IDs via ``squeue -j`` can fail the whole request as soon as
    # one ID has left the live queue.  Query the current user's queue once and
    # filter locally instead.
    output = run([
        "squeue", "-h", "-u", getpass.getuser(),
        "-o", "%i|%T|%r|%M|%l|%C|%m|%N",
    ])
    for raw in output.splitlines():
        fields = raw.split("|", 7)
        if len(fields) != 8:
            continue
        job_id, state, reason, elapsed, limit, cpus, memory, nodes = fields
        row = {
            "JobIDRaw": job_id, "State": state, "Reason": reason,
            "Elapsed": elapsed, "Timelimit": limit, "AllocCPUS": cpus,
            "ReqMem": memory, "NodeList": nodes,
        }
        all_rows.append(row)
        if job_id in wanted:
            result[job_id] = row
    return result, all_rows


def collect_sacct(job_ids: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for group in chunks(job_ids):
        output = run([
            "sacct", "-X", "-n", "-P", "-j", ",".join(group),
            "--format=" + ",".join(SACCT_FIELDS),
        ])
        for raw in output.splitlines():
            values = raw.split("|")
            if len(values) < len(SACCT_FIELDS):
                values.extend([""] * (len(SACCT_FIELDS) - len(values)))
            row = dict(zip(SACCT_FIELDS, values))
            job_id = row["JobIDRaw"]
            if job_id in job_ids:
                result[job_id] = row
    return result


def render_markdown(
    rows: list[dict[str, str]], untracked: list[dict[str, str]],
    output: Path, audit_time: str,
) -> None:
    classes = Counter(row["classification"] for row in rows)
    experiments = Counter(row["experiment_id"] for row in rows)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# Post-outage cluster audit — {audit_time}\n\n")
        stream.write("This is a read-only audit. It did not submit, cancel, release, or edit jobs.\n\n")
        stream.write("## Recovery classification\n\n")
        stream.write("| Classification | Jobs |\n|---|---:|\n")
        for key, count in sorted(classes.items()):
            stream.write(f"| {key} | {count} |\n")
        stream.write("\n## Inventory coverage\n\n")
        stream.write("| Experiment | Jobs checked |\n|---|---:|\n")
        for key, count in sorted(experiments.items()):
            stream.write(f"| {key} | {count} |\n")
        stream.write("\n## Live jobs absent from the pre-outage inventory\n\n")
        if not untracked:
            stream.write("None.\n")
        else:
            stream.write("These may be scheduler-created replacements, later submissions, or unrelated work. Review them before mutation.\n\n")
            stream.write("| Job | State | Reason | Elapsed | Limit | CPUs | Memory | Nodes |\n")
            stream.write("|---:|---|---|---:|---:|---:|---:|---|\n")
            for row in untracked:
                stream.write(
                    f"| {row['JobIDRaw']} | {row['State']} | {row['Reason']} | "
                    f"{row['Elapsed']} | {row['Timelimit']} | {row['AllocCPUS']} | "
                    f"{row['ReqMem']} | {row['NodeList']} |\n"
                )
        stream.write("\n## Jobs requiring review\n\n")
        stream.write("| Job | Experiment | State | Classification | Remaining | Recommendation |\n")
        stream.write("|---:|---|---|---|---:|---|\n")
        for row in rows:
            if row["classification"] in {
                "completed_with_checkpoint", "completed_with_final_result",
                "controller_completed", "held_intact", "dependency_pending",
                "scheduler_pending", "survived_or_restarted",
            }:
                continue
            stream.write(
                f"| {row['job_id']} | {row['experiment_id']} | {row['current_state']} | "
                f"{row['classification']} | {row['recommended_continuation_time']} | "
                f"{row['recommended_action']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiment_tracking" / "post_outage_audit")
    args = parser.parse_args()

    with args.inventory.open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    job_ids = [row["job_id"] for row in inventory]
    squeue, live_queue = collect_squeue(job_ids)
    sacct = collect_sacct(job_ids)
    audit_time = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, str]] = []

    for expected in inventory:
        job_id = expected["job_id"]
        account = dict(sacct.get(job_id, {}))
        account.update({key: value for key, value in squeue.get(job_id, {}).items() if value})
        state = account.get("State", "")
        reason = account.get("Reason", "")
        elapsed = account.get("Elapsed", "")
        limit = account.get("Timelimit", "") or expected["time_limit"]
        elapsed_seconds = parse_duration(elapsed)
        limit_seconds = parse_duration(limit)
        remaining = None
        if elapsed_seconds is not None and limit_seconds is not None:
            remaining = max(0, limit_seconds - elapsed_seconds)
        continuation = None if remaining is None else min(limit_seconds or remaining, max(3600, remaining + 3600))

        stdout = resolve_stdout(account.get("StdOut", ""), job_id)
        artifacts = parse_artifacts(stdout)
        partial_eval = bool(
            artifacts["printed_plan_instances"] or artifacts["completion_records"]
        )
        classification, action = classify(
            expected=expected["expected_pre_outage_state"],
            artifact_kind=expected["artifact_kind"],
            state=state,
            reason=reason,
            has_checkpoint=bool(artifacts["latest_checkpoint"]),
            eval_final=artifacts["eval_final_success"] is not None,
            partial_eval=partial_eval,
        )
        rows.append({
            "audit_time": audit_time,
            "experiment_id": expected["experiment_id"],
            "component": expected["component"],
            "job_id": job_id,
            "manifest_id": expected["manifest_id"],
            "expected_pre_outage_state": expected["expected_pre_outage_state"],
            "current_state": state,
            "reason": reason,
            "exit_code": account.get("ExitCode", ""),
            "elapsed": elapsed,
            "time_limit": limit,
            "remaining_seconds": "" if remaining is None else str(remaining),
            "recommended_continuation_time": format_duration(continuation),
            "cpus": account.get("AllocCPUS", "") or expected["cpus"],
            "requested_memory": account.get("ReqMem", "") or expected["memory_gib"] + "G",
            "node_list": account.get("NodeList", ""),
            "start": account.get("Start", ""),
            "end": account.get("End", ""),
            "stdout": "" if stdout is None else str(stdout),
            "stdout_exists": str(artifacts["stdout_exists"]).lower(),
            "latest_epoch": "" if artifacts["latest_epoch"] is None else str(artifacts["latest_epoch"]),
            "target_epoch": "" if artifacts["target_epoch"] is None else str(artifacts["target_epoch"]),
            "latest_checkpoint": str(artifacts["latest_checkpoint"]),
            "eval_final_success": "" if artifacts["eval_final_success"] is None else str(artifacts["eval_final_success"]),
            "eval_final_total": "" if artifacts["eval_final_total"] is None else str(artifacts["eval_final_total"]),
            "printed_plan_instances": str(artifacts["printed_plan_instances"]),
            "completion_records": str(artifacts["completion_records"]),
            "classification": classification,
            "recommended_action": action,
            "notes": expected["notes"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"post_outage_audit_{stamp}.csv"
    md_path = args.output_dir / f"post_outage_audit_{stamp}.md"
    untracked_path = args.output_dir / f"post_outage_untracked_live_jobs_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    known_ids = set(job_ids)
    untracked = [row for row in live_queue if row["JobIDRaw"] not in known_ids]
    with untracked_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "JobIDRaw", "State", "Reason", "Elapsed", "Timelimit",
            "AllocCPUS", "ReqMem", "NodeList",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(untracked)
    render_markdown(rows, untracked, md_path, audit_time)
    print(csv_path)
    print(md_path)
    print(untracked_path)


if __name__ == "__main__":
    main()

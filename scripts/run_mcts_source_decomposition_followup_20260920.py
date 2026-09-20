#!/usr/bin/env python3
"""Render or run one frozen source-decomposed MCTS root diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking/mcts_policy_divergence_cause_audit"
SOURCE_RUNNER_PATH = ROOT / "scripts/run_mcts_policy_divergence_stage1_20260918.py"
SOURCE_SPEC = importlib.util.spec_from_file_location(
    "mcts_divergence_stage1_runner", SOURCE_RUNNER_PATH)
source_runner = importlib.util.module_from_spec(SOURCE_SPEC)
assert SOURCE_SPEC.loader is not None
SOURCE_SPEC.loader.exec_module(source_runner)

MANIFEST_SCHEMA = "mcts-source-decomposition-followup-v1"
FREEZE_SCHEMA = "mcts-source-decomposition-followup-freeze-v1"
RESULT_SCHEMA = "mcts-source-decomposition-followup-result-v1"
RECORD_SCHEMA = "mcts-root-source-decomposition-v1"
MARKER = "[MCTS SOURCE DECOMPOSITION] "
STOP_MARKER = "[MCTS SOURCE DECOMPOSITION STOP] "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_design(
    manifest_path: Path,
    freeze_path: Path,
    source_manifest_path: Path,
    reconciliation_path: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported source-decomposition manifest schema")
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise RuntimeError("unsupported source-decomposition freeze schema")
    if payload.get("submitted") is not False or freeze.get("submitted") is not False:
        raise RuntimeError("source-decomposition design is not frozen unsubmitted")
    checks = {
        "manifest_sha256": sha256_file(manifest_path),
        "source_grouped_manifest_sha256": sha256_file(source_manifest_path),
        "source_reconciliation_sha256": sha256_file(reconciliation_path),
    }
    mismatches = {
        key: (freeze.get(key), value)
        for key, value in checks.items()
        if freeze.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"source-decomposition freeze mismatch: {mismatches}")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 6:
        raise RuntimeError("source-decomposition design must contain six tasks")
    if [task.get("task_index") for task in tasks] != list(range(6)):
        raise RuntimeError("source-decomposition task indices must be 0..5")
    if len({task.get("diagnostic_id") for task in tasks}) != 6:
        raise RuntimeError("diagnostic ids must be unique")
    if {task.get("role") for task in tasks} != {"harmful", "control"}:
        raise RuntimeError("design must contain harmful and control roles")
    if sum(task["role"] == "harmful" for task in tasks) != 3:
        raise RuntimeError("design must contain exactly three harmful cases")
    if sum(task["role"] == "control" for task in tasks) != 3:
        raise RuntimeError("design must contain exactly three controls")
    if any("matched_pair" in task for task in tasks):
        raise RuntimeError("descriptive controls must not be labelled matched pairs")
    if any(not task.get("contrast_group") for task in tasks):
        raise RuntimeError("every task must name a descriptive contrast group")
    controls = [task for task in tasks if task["role"] == "control"]
    harmful_ids = {
        task["diagnostic_id"] for task in tasks if task["role"] == "harmful"}
    if {task.get("descriptive_control_for") for task in controls} != harmful_ids:
        raise RuntimeError("each harmful root requires one descriptive control")

    source_rows = {
        row["task_id"]: row for row in read_csv(source_manifest_path)
    }
    reconciliation = {
        (row["task_id"], int(row["candidate_index"])): row
        for row in read_csv(reconciliation_path)
    }
    for task in tasks:
        source = source_rows.get(task["source_task_id"])
        if source is None:
            raise RuntimeError(f"unknown source task: {task['source_task_id']}")
        candidates = json.loads(source["candidate_runs_json"])
        index = int(task["source_candidate_index"])
        if not 0 <= index < len(candidates):
            raise RuntimeError("source candidate index is out of range")
        candidate = candidates[index]
        expected = {
            "domain": source["domain"],
            "value_head": source["value_head"],
            "checkpoint_identity": candidate["checkpoint_identity"],
            "seed": int(candidate["seed"]),
            "instance": candidate["instance"],
            "tie_break": source["tie_break"],
            "terminal_safe": source["terminal_safe"].lower() == "true",
            "search_config": json.loads(source["search_config"]),
            "policy_log": candidate["policy_log"],
            "search_log": candidate["search_log"],
        }
        mismatches = {
            key: (task.get(key), value)
            for key, value in expected.items()
            if task.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"{task['diagnostic_id']}: source manifest mismatch: {mismatches}")
        joined = reconciliation.get((task["source_task_id"], index))
        if joined is None:
            raise RuntimeError(f"{task['diagnostic_id']}: missing reconciliation row")
        joined_expected = {
            "historical_outcome": joined["outcome"],
            "historical_stratum": joined["current_outcome_stratum"],
            "historical_mechanism": joined["selection_mechanism"],
        }
        mismatches = {
            key: (task.get(key), value)
            for key, value in joined_expected.items()
            if task.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"{task['diagnostic_id']}: reconciliation mismatch: {mismatches}")
        if int(task["target_step"]) < 0:
            raise RuntimeError("target step cannot be negative")
    return tasks


def diagnostic_records(log: Path) -> list[dict[str, Any]]:
    records = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(MARKER)
        if marker >= 0:
            records.append(json.loads(line[marker + len(MARKER):]))
    return records


def validate_record(record: dict[str, Any], task: dict[str, Any]) -> None:
    if record.get("schema_version") != RECORD_SCHEMA:
        raise RuntimeError("source diagnostic schema mismatch")
    if Path(record.get("instance", "")).name != task["instance"]:
        raise RuntimeError("source diagnostic instance mismatch")
    if int(record.get("step", -1)) != int(task["target_step"]):
        raise RuntimeError("source diagnostic step mismatch")
    if record.get("diagnostic_stop_after_record") is not True:
        raise RuntimeError("source diagnostic is not marked stop-after-record")
    if bool(record.get("actions_diverge")) != bool(
            task["expected_actions_diverge"]):
        raise RuntimeError("source diagnostic action-divergence mismatch")
    for key, record_key in (
        ("expected_policy_action", "policy_action"),
        ("expected_selected_action", "selected_action"),
    ):
        if task.get(key) is not None and int(record[record_key]) != int(task[key]):
            raise RuntimeError(f"source diagnostic {record_key} mismatch")
    root = record.get("root", {})
    act_dim = root.get("act_dim")
    sources = root.get("q_source_decomposition")
    if not isinstance(act_dim, int) or act_dim <= 0:
        raise RuntimeError("source diagnostic has invalid action dimension")
    if not isinstance(sources, list) or len(sources) != act_dim:
        raise RuntimeError("source decomposition vector length mismatch")
    expanded = root.get("expanded")
    visits = root.get("edge_visit_counts")
    if not isinstance(expanded, list) or not isinstance(visits, list):
        raise RuntimeError("source diagnostic is missing root vectors")
    populated = 0
    for action, row in enumerate(sources):
        if row is None:
            if int(visits[action]) > 0:
                raise RuntimeError(
                    f"visited action {action} lacks source decomposition")
            continue
        populated += 1
        if row.get("scope") != "node_global_matches_child_q":
            raise RuntimeError("source decomposition scope mismatch")
        if int(row.get("backups", 0)) <= 0:
            raise RuntimeError("source decomposition has no backups")
        if abs(float(row["reconstruction_residual"])) > 1e-6:
            raise RuntimeError("source decomposition does not reconstruct Q")
        if sum(int(value) for value in row["leaf_source_counts"].values()) != int(
                row["backups"]):
            raise RuntimeError("source decomposition count mismatch")
    if populated == 0:
        raise RuntimeError("source diagnostic contains no populated action source")


def build_command(
    *, repo: Path, task: dict[str, Any], checkpoint: str,
    completion: Path,
) -> list[str]:
    command = source_runner.build_command(
        repo=repo,
        domain=task["domain"],
        value_head=task["value_head"],
        seed=int(task["seed"]),
        checkpoint=checkpoint,
        instance_name=task["instance"],
        search_config=task["search_config"],
        tie_break=task["tie_break"],
        terminal_safe=bool(task["terminal_safe"]),
        completion=completion,
        instance_timeout=6600,
        max_actions=10000,
        first_divergence_record=False,
        source_decomposition_step=int(task["target_step"]),
    )
    # A generic evaluation ledger would classify this intentional diagnostic
    # stop as ``finished_unsolved``.  The dedicated validated result JSON must
    # be the only completion artifact for this observation-only campaign.
    completion_index = command.index("--eval-completion-file")
    del command[completion_index:completion_index + 2]
    return command


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_result_payload(
    *, task: dict[str, Any], task_index: int, record: dict[str, Any],
    checkpoint: str, log: Path, completion: Path, code_commit: str,
    manifest_sha256: str, registry_sha256: str,
) -> dict[str, Any]:
    """Build a diagnostic result that cannot masquerade as a full outcome."""
    return {
        "schema_version": RESULT_SCHEMA,
        "task_index": task_index,
        "diagnostic_id": task["diagnostic_id"],
        "role": task["role"],
        "contrast_group": task["contrast_group"],
        "checkpoint_identity": task["checkpoint_identity"],
        "checkpoint": checkpoint,
        "instance": task["instance"],
        "target_step": task["target_step"],
        "record": record,
        "diagnostic_stopped_after_record": True,
        "terminal_outcome_repeated": False,
        "historical_outcome": task["historical_outcome"],
        "historical_outcome_source": task["search_log"],
        "log": str(log),
        "generic_completion_file_disabled": True,
        "generic_completion_file_expected_absent": str(completion),
        "code_commit": code_commit,
        "manifest_sha256": manifest_sha256,
        "registry_sha256": registry_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    tasks = load_design(
        args.manifest, args.freeze, args.source_manifest, args.reconciliation)
    if not 0 <= args.task_index < len(tasks):
        raise RuntimeError("task index out of range")
    task = tasks[args.task_index]
    registry_rows = source_runner.read_csv(args.registry)
    checkpoint, registry_row = source_runner.resolve_checkpoint(
        registry_rows,
        task["checkpoint_identity"],
        policy_log=task["policy_log"],
    )
    task_dir = args.output_root / task["diagnostic_id"]
    log = task_dir / "diagnostic.txt"
    completion = task_dir / "diagnostic.completed.jsonl"
    result_path = task_dir / "diagnostic.result.json"
    command = build_command(
        repo=args.repo, task=task, checkpoint=checkpoint,
        completion=completion)
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "task": task,
            "checkpoint": checkpoint,
            "registry_job": registry_row["job_id"],
            "command": command,
            "result": str(result_path),
        }, indent=2))
        return

    if not Path(checkpoint).is_dir():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    if log.exists() or completion.exists() or result_path.exists():
        raise RuntimeError("refusing to overwrite diagnostic artifacts")
    task_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": MANIFEST_SCHEMA,
        "diagnostic_id": task["diagnostic_id"],
        "task_index": args.task_index,
        "code_commit": args.code_commit,
        "checkpoint": checkpoint,
        "checkpoint_identity": task["checkpoint_identity"],
        "manifest_sha256": sha256_file(args.manifest),
    }
    with log.open("w", encoding="utf-8") as stream:
        stream.write(
            "[MCTS SOURCE DECOMPOSITION RUN] "
            + json.dumps(header, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        stream.flush()
        result = subprocess.run(
            command,
            cwd=args.repo / "asnets",
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        raise SystemExit(result.returncode)
    records = diagnostic_records(log)
    if len(records) != 1:
        raise RuntimeError(
            f"expected one source diagnostic record, found {len(records)}")
    if STOP_MARKER not in log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("diagnostic stop marker is missing")
    validate_record(records[0], task)
    if completion.exists():
        raise RuntimeError(
            "generic completion ledger unexpectedly classified diagnostic stop")
    atomic_json(result_path, build_result_payload(
        task=task,
        task_index=args.task_index,
        record=records[0],
        checkpoint=checkpoint,
        log=log,
        completion=completion,
        code_commit=args.code_commit,
        manifest_sha256=sha256_file(args.manifest),
        registry_sha256=sha256_file(args.registry),
    ))


if __name__ == "__main__":
    main()

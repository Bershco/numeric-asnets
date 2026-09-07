#!/usr/bin/env python3
"""Inventory scored Drone MCTS jobs and audit interruption-lost instances.

The local ``inventory`` mode joins the authoritative/experiment-specific CSV
ledgers.  The cluster ``audit`` mode reads the original logs, asks Slurm for
terminal allocation state, and reports only scheduler-interrupted jobs.  An
instance is an *opportunity lost to interruption* only when it has no terminal
``[EVAL INSTANCE]`` record after joining the complete log for that allocation.
Per-instance timeouts and ordinary unsolved records are therefore not counted.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


TRACKING = Path(__file__).resolve().parents[1] / "experiment_tracking"
SOURCES = (
    "experiment_results.csv",
    "drone_endpoint_mcts_results.csv",
    "main_val_stage2_drone_mcts_terminal_20260831.csv",
    "mcts_progressive_widening_pilot/results.csv",
    "mcts_progressive_widening_sensitivity/results.csv",
    "mcts_horizon_pilot/results.csv",
    "mcts_horizon_binding/results.csv",
    "mcts_safe_context/live_reconciliation_20260831.csv",
    "mcts_safe_drone/targeted_results_20260828.csv",
    "mcts_determinism_audit/results.csv",
    "mcts_determinism_audit/followup_results.csv",
)
JOB_FIELDS = ("job_id", "slurm_job_id", "mcts_job_id")
LOG_FIELDS = ("source_evaluation_log", "source_log", "mcts_log", "evaluation_log")
BAD_STATES = {"OUT_OF_MEMORY", "OOM", "TIMEOUT", "FAILED", "NODE_FAIL", "PREEMPTED", "CANCELLED"}
RECORD_RE = re.compile(
    r"\[EVAL INSTANCE\] (?:completed|skip completed) "
    r"number=(?P<number>\d+) path=(?P<path>\S+) "
    r"status=(?P<status>\S+) elapsed=(?P<elapsed>[0-9.]+)s "
    r"success=(?P<success>\S+) steps=(?P<steps>-?\d+)"
)
TIMEOUT_RE = re.compile(
    r"\[EVAL INSTANCE\] timeout number=(?P<number>\d+) "
    r"path=(?P<path>\S+) limit=(?P<elapsed>[0-9.]+)s"
)
DRONE_INSTANCES = (
    "instances/problem_1_1_2.pddl", "instances/problem_1_1_4.pddl",
    "instances/problem_1_8_1.pddl", "instances/problem_1_9_3.pddl",
    "instances/problem_2_5_4.pddl", "instances/problem_2_8_3.pddl",
    "instances/problem_2_9_1.pddl", "instances/problem_3_3_4.pddl",
    "instances/problem_4_2_5.pddl", "instances/problem_4_9_5.pddl",
    "instances/problem_5_2_2.pddl", "instances/problem_5_10_1.pddl",
    "instances/problem_6_8_1.pddl", "instances/problem_7_6_4.pddl",
    "instances/problem_8_1_2.pddl", "instances/problem_8_1_4.pddl",
    "instances/problem_8_1_5.pddl", "instances/problem_8_7_1.pddl",
    "instances/problem_9_7_3.pddl", "instances/problem_10_10_3.pddl",
)


def first(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def inventory(output: Path) -> None:
    jobs: dict[str, dict[str, str]] = {}
    for relative in SOURCES:
        path = TRACKING / relative
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                domain = row.get("domain", "drone")
                if domain and domain != "drone":
                    continue
                if relative == "experiment_results.csv" and row.get("task_type") != "mcts_eval":
                    continue
                job_id = first(row, JOB_FIELDS)
                log = first(row, LOG_FIELDS)
                if not job_id or not log or not re.fullmatch(r"\d{7,8}", job_id):
                    continue
                current = jobs.setdefault(job_id, {
                    "job_id": job_id, "experiment_id": row.get("experiment_id", ""),
                    "stage": row.get("stage", ""), "value_head": row.get("value_head", ""),
                    "seed": row.get("seed", ""), "arm": row.get("arm", row.get("variant", "")),
                    "source_log": log, "source_ledgers": relative,
                })
                ledgers = set(current["source_ledgers"].split(";"))
                ledgers.add(relative)
                current["source_ledgers"] = ";".join(sorted(ledgers))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["job_id", "experiment_id", "stage", "value_head", "seed", "arm", "source_log", "source_ledgers"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sorted(jobs.values(), key=lambda row: int(row["job_id"])))
    print(f"inventory_jobs={len(jobs)} output={output}")


def slurm_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    states: dict[str, dict[str, str]] = {}
    for start in range(0, len(job_ids), 100):
        ids = ",".join(job_ids[start:start + 100])
        result = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", ids,
             "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList"],
            check=True, text=True, capture_output=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 7 or not re.fullmatch(r"\d{7,8}", parts[0]):
                continue
            states[parts[0]] = dict(zip(
                ("job_id", "slurm_state", "exit_code", "elapsed", "start", "end", "node"),
                parts[:7], strict=True,
            ))
    return states


def parse_records(path: Path) -> dict[int, dict[str, str]]:
    records: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = RECORD_RE.search(line)
            if match:
                records[int(match["number"])] = match.groupdict()
                continue
            match = TIMEOUT_RE.search(line)
            if match:
                records.setdefault(int(match["number"]), {
                    "number": match["number"], "path": match["path"], "status": "timeout",
                    "elapsed": match["elapsed"], "success": "False", "steps": "-1",
                })
    return records


def audit(input_path: Path, jobs_output: Path, instances_output: Path) -> None:
    rows = list(csv.DictReader(input_path.open(newline="", encoding="utf-8-sig")))
    states = slurm_states([row["job_id"] for row in rows])
    job_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    for row in rows:
        state = states.get(row["job_id"], {"slurm_state": "NOT_FOUND", "exit_code": "", "elapsed": "", "start": "", "end": "", "node": ""})
        normalized = state["slurm_state"].split("+")[0]
        if normalized not in BAD_STATES:
            continue
        log = Path(row["source_log"])
        records = parse_records(log) if log.is_file() else {}
        missing = [number for number in range(1, 21) if number not in records]
        successes = sum(str(record.get("success", "")).lower() in {"1", "1.0", "true"} for record in records.values())
        timeouts = sum(record.get("status") == "timeout" for record in records.values())
        joined = {**row, **state, "log_exists": int(log.is_file()), "classified_instances": len(records),
                  "recorded_successes": successes, "recorded_instance_timeouts": timeouts,
                  "opportunities_lost_to_interruption": len(missing),
                  "missing_instance_numbers": ";".join(map(str, missing)),
                  "missing_instance_paths": ";".join(DRONE_INSTANCES[number - 1] for number in missing)}
        job_rows.append(joined)
        for number in missing:
            instance_rows.append({
                "job_id": row["job_id"], "experiment_id": row["experiment_id"], "stage": row["stage"],
                "value_head": row["value_head"], "seed": row["seed"], "arm": row["arm"],
                "slurm_state": state["slurm_state"], "instance_number": number,
                "instance_path": DRONE_INSTANCES[number - 1], "source_log": row["source_log"],
                "source_ledgers": row["source_ledgers"],
            })
    jobs_output.parent.mkdir(parents=True, exist_ok=True)
    job_fields = list(job_rows[0]) if job_rows else [
        "job_id", "experiment_id", "stage", "value_head", "seed", "arm", "source_log",
        "source_ledgers", "slurm_state", "exit_code", "elapsed", "start", "end", "node",
        "log_exists", "classified_instances", "recorded_successes", "recorded_instance_timeouts",
        "opportunities_lost_to_interruption", "missing_instance_numbers", "missing_instance_paths",
    ]
    instance_fields = list(instance_rows[0]) if instance_rows else [
        "job_id", "experiment_id", "stage", "value_head", "seed", "arm", "slurm_state",
        "instance_number", "instance_path", "source_log", "source_ledgers",
    ]
    for path, fields, values in ((jobs_output, job_fields, job_rows), (instances_output, instance_fields, instance_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    print(f"interrupted_jobs={len(job_rows)} opportunities={len(instance_rows)} jobs_out={jobs_output} instances_out={instances_output}")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    inv = sub.add_parser("inventory"); inv.add_argument("--output", type=Path, required=True)
    aud = sub.add_parser("audit"); aud.add_argument("--input", type=Path, required=True)
    aud.add_argument("--jobs-output", type=Path, required=True); aud.add_argument("--instances-output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "inventory": inventory(args.output)
    else: audit(args.input, args.jobs_output, args.instances_output)


if __name__ == "__main__":
    main()

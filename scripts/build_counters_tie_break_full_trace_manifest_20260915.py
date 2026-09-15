#!/usr/bin/env python3
"""Build the exact Counters policy-success/action-ID-failure trace manifest.

The builder is intentionally fail-closed.  It refuses to emit a trace manifest
unless every Stage-1 VH-off action-ID completion ledger contains exactly one
terminal record for each of the 59 test instances and the parsed pure-policy
success count matches the frozen strict-campaign manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TOTAL_INSTANCES = 59
SOURCE_ARRAY_JOB_ID = "21233925"
RULES = ("action_id", "policy")
COMPLETED_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(?P<number>\d+) .*?"
    r"status=(?P<status>\S+)"
)
PLAN_RE = re.compile(
    r"\[EVAL\]\[PLAN\].*?fz_instance_(?P<pddl_number>\d+)\.pddl"
)


def parse_policy_successes(path: Path) -> set[int]:
    text = path.read_text(errors="replace")
    terminal = {
        int(match.group("number")): match.group("status")
        for match in COMPLETED_RE.finditer(text)
    }
    if terminal:
        return {number for number, status in terminal.items() if status == "success"}
    # The historical policy logs predate per-instance completion records.  The
    # Counters test files are fz_instance_2..fz_instance_60 while evaluator
    # identities are 1..59, hence the explicit minus one.
    return {
        int(match.group("pddl_number")) - 1
        for match in PLAN_RE.finditer(text)
    }


def load_complete_action_id_ledger(path: Path) -> dict[int, dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = Counter(int(record["instance_number"]) for record in records)
    duplicate = sorted(number for number, count in counts.items() if count != 1)
    if duplicate:
        raise RuntimeError(f"duplicate action-ID ledger identities in {path}: {duplicate}")
    expected = set(range(1, TOTAL_INSTANCES + 1))
    actual = set(counts)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"action-ID ledger is not complete: {path}; missing={missing}, extra={extra}"
        )
    allowed = {"success", "finished_unsolved"}
    bad_status = sorted(
        (int(record["instance_number"]), record.get("status"))
        for record in records
        if record.get("status") not in allowed
    )
    if bad_status:
        raise RuntimeError(f"non-terminal statuses in {path}: {bad_status}")
    return {int(record["instance_number"]): record for record in records}


def build_rows(base_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    action_rows = [row for row in base_rows if row["tie_break"] == "action_id"]
    if len(action_rows) != 10:
        raise RuntimeError(f"expected ten action-ID rows, found {len(action_rows)}")
    if len({row["seed"] for row in action_rows}) != 10:
        raise RuntimeError("action-ID strict rows do not contain ten unique seeds")

    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for base in sorted(action_rows, key=lambda row: int(row["array_index"])):
        source_task = int(base["array_index"])
        policy_log = Path(base["source_policy_log"])
        source_ledger = (
            Path(base["remote_output"])
            / f"{SOURCE_ARRAY_JOB_ID}_{source_task}.completed.jsonl"
        )
        policy_successes = parse_policy_successes(policy_log)
        expected_policy_score = int(base["policy_score"])
        if len(policy_successes) != expected_policy_score:
            raise RuntimeError(
                f"policy score mismatch for seed {base['seed']}: "
                f"parsed={len(policy_successes)}, expected={expected_policy_score}, "
                f"log={policy_log}"
            )
        invalid_policy_numbers = sorted(
            policy_successes - set(range(1, TOTAL_INSTANCES + 1))
        )
        if invalid_policy_numbers:
            raise RuntimeError(
                f"invalid policy instance numbers for seed {base['seed']}: "
                f"{invalid_policy_numbers}"
            )
        action_records = load_complete_action_id_ledger(source_ledger)
        wanted = sorted(
            number
            for number in policy_successes
            if action_records[number]["status"] != "success"
        )
        summary_rows.append({
            "seed": base["seed"],
            "source_task": source_task,
            "policy_successes": len(policy_successes),
            "action_id_successes": sum(
                record["status"] == "success" for record in action_records.values()
            ),
            "policy_success_action_id_failures": len(wanted),
            "instance_numbers": ";".join(map(str, wanted)),
            "source_policy_log": str(policy_log),
            "source_action_id_ledger": str(source_ledger),
        })
        for number in wanted:
            for rule in RULES:
                trace_rows.append({
                    "array_index": len(trace_rows),
                    "tie_break": rule,
                    "seed": base["seed"],
                    "instance_number": number,
                    "source_strict_task": source_task,
                    "source_training_job_id": base["source_training_job_id"],
                    "source_policy_job_id": base["source_policy_job_id"],
                    "checkpoint": base["checkpoint"],
                    "source_training_log": base["source_training_log"],
                    "source_policy_log": str(policy_log),
                    "source_action_id_ledger": str(source_ledger),
                    "source_action_id_status": action_records[number]["status"],
                    "source_action_id_steps": action_records[number].get("steps", ""),
                    "search": "narrow_5_children_20_simulations",
                    "workers": 1,
                    "cpus": 2,
                    "ram_gib": 120,
                    "instance_timeout": "6h",
                    "job_walltime": "8h",
                    "remote_output": (
                        "/home/hersco/training_new_domains/2026-09-15/"
                        "counters_tie_break_full_trace/"
                        f"{rule}/{base['seed']}/instance_{number}"
                    ),
                })
    if not trace_rows:
        raise RuntimeError("strict evidence contains no policy-success/action-ID failures")
    identities = [
        (row["tie_break"], row["seed"], row["instance_number"])
        for row in trace_rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate trace identities were generated")
    if len(trace_rows) != 2 * sum(
            row["policy_success_action_id_failures"] for row in summary_rows):
        raise RuntimeError("each wanted identity must occur under exactly two rules")
    return trace_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    with args.base_manifest.open(newline="", encoding="utf-8-sig") as stream:
        base_rows = list(csv.DictReader(stream))
    trace_rows, summary_rows = build_rows(base_rows)
    write_csv(args.output_manifest, trace_rows)
    write_csv(args.output_summary, summary_rows)
    print(json.dumps({
        "seeds": len(summary_rows),
        "wanted_identities": len(trace_rows) // 2,
        "trace_tasks": len(trace_rows),
        "action_id_tasks": sum(row["tie_break"] == "action_id" for row in trace_rows),
        "policy_tasks": sum(row["tie_break"] == "policy" for row in trace_rows),
        "manifest": str(args.output_manifest),
        "summary": str(args.output_summary),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

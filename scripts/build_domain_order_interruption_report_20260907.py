#!/usr/bin/env python3
"""Build the reproducible test-order and interrupted-MCTS opportunity audit.

This deliberately distinguishes an *unclassified instance* (the allocation
ended before that instance obtained a terminal record) from an ordinary
unsolved instance or an explicit per-instance timeout.  It reads only the
small ``.resume_state/*.eval_completed.jsonl`` ledgers on the cluster; it does
not rescan multi-gigabyte stdout files.
"""

from __future__ import annotations

import base64
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
SSH = Path(r"C:\Windows\System32\OpenSSH\ssh.exe")


DOMAIN_ROWS = [
    {
        "domain": "drone",
        "test_instances": 20,
        "order_class": "non_monotone",
        "include_primary_interruption_audit": "yes",
        "observed_structural_sequence": "objects/cells=2,4,8,27,40,48,18,36,40,180,20,50,48,168,16,32,40,56,189,300",
        "logical_basis": "Repeated large reversals (48->18, 180->20, 168->16); later index is not uniformly harder.",
    },
    {
        "domain": "fo_counters",
        "test_instances": 20,
        "order_class": "strictly_increasing_by_construction",
        "include_primary_interruption_audit": "no_supplementary_only",
        "observed_structural_sequence": "instance_2..instance_21; n counters; n-1 chain goals; max_int=2n",
        "logical_basis": "Every successive instance adds one counter, one chain requirement, and a larger numeric bound.",
    },
    {
        "domain": "rover",
        "test_instances": 20,
        "order_class": "non_monotone_with_increasing_scale_trend",
        "include_primary_interruption_audit": "yes",
        "observed_structural_sequence": "objects=13,14,16,18,18,19,20,25,27,29,27,28,30,31,32,33,44,50,51,60; goals=3,3,3,3,7,10,6,8,8,11,9,6,12,8,10,11,13,11,17,20",
        "logical_basis": "Overall scale rises, but adjacent reversals in both objects and goals make later unrun instances potentially easier.",
    },
    {
        "domain": "mprime",
        "test_instances": 20,
        "order_class": "non_monotone",
        "include_primary_interruption_audit": "yes",
        "observed_structural_sequence": "objects=10,31,18,27,24,75,74,41,27,38,37,33,52,55,42,55,8,21,16,13; goals vary 1..3",
        "logical_basis": "Large reversals throughout the ordered suite; file index is not a difficulty ordering.",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def completion_path(log: str, job_id: str) -> str:
    parent = str(Path(log).parent).replace("\\", "/")
    return f"{parent}/.resume_state/{job_id}.eval_completed.jsonl"


def rover_jobs() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for row in read_csv(TRACKING / "experiment_results.csv"):
        if row.get("domain") != "rover" or row.get("task_type") != "mcts_eval" or row.get("stage") != "stage1":
            continue
        if row.get("slurm_state") != "OUT_OF_MEMORY":
            continue
        jobs.append({
            "audit_scope": "primary_non_monotone",
            "domain": "rover", "experiment_scope": "stage1_validation_selected_fixed_20_70",
            "value_head": row["value_head"], "seed": row["seed"], "job_id": row["job_id"],
            "slurm_state": row["slurm_state"], "recorded_successes": row["score"],
            "recorded_instance_timeouts": row.get("instance_timeouts", ""),
            "source_log": row["source_evaluation_log"],
            "completion_ledger": completion_path(row["source_evaluation_log"], row["job_id"]),
            "source_result_ledger": "experiment_tracking/experiment_results.csv",
        })
    for row in read_csv(TRACKING / "fo_terminal_stage2_mcts_reconciliation_20260903.csv"):
        if row.get("domain") != "rover" or row.get("slurm_state") != "OUT_OF_MEMORY" or int(row.get("classified_instances") or 0) >= 20:
            continue
        jobs.append({
            "audit_scope": "primary_non_monotone",
            "domain": "rover", "experiment_scope": "stage2_terminal_led_fixed_20_70",
            "value_head": row["value_head"], "seed": row["seed"], "job_id": row["job_id"],
            "slurm_state": row["slurm_state"], "recorded_successes": row["successes"],
            "recorded_instance_timeouts": "",
            "source_log": row["source_evaluation_log"],
            "completion_ledger": completion_path(row["source_evaluation_log"], row["job_id"]),
            "source_result_ledger": "experiment_tracking/fo_terminal_stage2_mcts_reconciliation_20260903.csv",
        })
    for row in read_csv(TRACKING / "rover_validation_seed_results_20260906.csv"):
        if row.get("slurm_state") != "OUT_OF_MEMORY":
            continue
        jobs.append({
            "audit_scope": "primary_non_monotone",
            "domain": "rover", "experiment_scope": "stage2_validation_led_fixed_20_70",
            "value_head": row["value_head"], "seed": row["seed"], "job_id": row["job_id"],
            "slurm_state": row["slurm_state"], "recorded_successes": row["mcts_6h"],
            "recorded_instance_timeouts": "", "source_log": row["mcts_log"],
            "completion_ledger": row["completion_record"],
            "source_result_ledger": "experiment_tracking/rover_validation_seed_results_20260906.csv",
        })
    return jobs


def fetch_membership(jobs: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    specs = [
        {"job_id": row["job_id"], "ledger": row["completion_ledger"], "log": row["source_log"]}
        for row in jobs
    ]
    remote = r'''
import json
import re
from pathlib import Path
specs = json.loads(%r)
for spec in specs:
    ledger = Path(spec["ledger"])
    log = Path(spec["log"])
    terminal = set()
    started = set()
    crashed = set()
    if log.is_file():
        with log.open(errors="replace") as handle:
            for line in handle:
                match = re.search(r"\[EVAL INSTANCE\] (?:completed|skip completed|timeout) number=(\d+)", line)
                if match:
                    terminal.add(int(match.group(1)))
                match = re.search(r"\[EVAL INSTANCE\] started number=(\d+)", line)
                if match:
                    started.add(int(match.group(1)))
                match = re.search(r"\[EVAL INSTANCE\] crashed number=(\d+)", line)
                if match:
                    crashed.add(int(match.group(1)))
    print(json.dumps({"job_id": spec["job_id"], "ledger_exists": ledger.is_file(), "log_exists": log.is_file(), "terminal": sorted(terminal), "started": sorted(started), "crashed": sorted(crashed)}))
''' % json.dumps(specs)
    payload = base64.b64encode(remote.encode()).decode()
    command = f"python3 -c \"import base64;exec(base64.b64decode('{payload}'))\""
    result = subprocess.run([str(SSH), "uni-cluster", command], check=True, text=True, capture_output=True)
    return {row["job_id"]: row for row in map(json.loads, result.stdout.splitlines())}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    jobs = rover_jobs()
    membership = fetch_membership(jobs)
    rover_instances = [f"pfile{number}" for number in range(1, 21)]
    instance_rows: list[dict[str, object]] = []
    for row in jobs:
        found = membership[row["job_id"]]
        numbers = set(map(int, found["terminal"]))
        started = set(map(int, found["started"]))
        crashed = set(map(int, found["crashed"]))
        missing = [number for number in range(1, 21) if number not in numbers]
        row["completion_ledger_exists"] = "yes" if found["ledger_exists"] else "no"
        row["source_log_exists"] = "yes" if found["log_exists"] else "no"
        row["classified_instances"] = len(numbers)
        row["unclassified_opportunities"] = len(missing)
        row["started_but_unclassified"] = ";".join(map(str, sorted((started | crashed) - numbers)))
        row["never_started"] = ";".join(map(str, sorted(set(range(1, 21)) - started - numbers)))
        row["missing_instance_numbers"] = ";".join(map(str, missing))
        row["missing_instance_paths"] = ";".join(rover_instances[number - 1] for number in missing)
        row["interpretation"] = "Could still contain solvable instances; original fixed-budget result remains unchanged."
        for number in missing:
            instance_rows.append({
                **{key: row[key] for key in ("audit_scope", "domain", "experiment_scope", "value_head", "seed", "job_id", "slurm_state")},
                "instance_number": number, "instance_path": rover_instances[number - 1],
                "source_log": row["source_log"], "completion_ledger": row["completion_ledger"],
                "source_result_ledger": row["source_result_ledger"],
            })

    # FO Counters is structurally increasing and therefore excluded from the
    # primary non-monotone cohort, but retained as the motivating supplement.
    fo_rows = [
        {
            "audit_scope": "supplementary_increasing_order", "domain": "fo_counters",
            "experiment_scope": "stage1_validation_selected_fixed_20_70", "value_head": "off",
            "seed": "1963100312", "job_id": "20430072", "slurm_state": "OUT_OF_MEMORY",
            "recorded_successes": "9", "recorded_instance_timeouts": "10", "classified_instances": 19,
            "unclassified_opportunities": 1, "missing_instance_numbers": "20", "missing_instance_paths": "instance_21.pddl",
            "source_log": "/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_mcts_eval/20430072_Ev_fo_counters_fo_counters_mcts_orig_novh_e.5_c.1_s1963100312_K0_SR10M_src20401228_e0008.txt",
            "completion_ledger": "/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_mcts_eval/.resume_state/20430072.eval_completed.jsonl",
            "completion_ledger_exists": "yes", "source_result_ledger": "experiment_tracking/experiment_results.csv",
            "interpretation": "One last, structurally hardest instance lacked a terminal record; supplementary only.",
        },
        {
            "audit_scope": "supplementary_increasing_order", "domain": "fo_counters",
            "experiment_scope": "pw70_confirmation_stage1", "value_head": "off", "seed": "2082152039",
            "job_id": "21039205", "slurm_state": "OUT_OF_MEMORY", "recorded_successes": "7",
            "recorded_instance_timeouts": "9", "classified_instances": 19, "unclassified_opportunities": 1,
            "missing_instance_numbers": "14", "missing_instance_paths": "instance_15.pddl",
            "source_log": "/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/21039205_pw70-ten-fo_counters-off-2082152039-s1.txt",
            "completion_ledger": "/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/.resume_state/21039205.eval_completed.jsonl",
            "completion_ledger_exists": "yes", "source_result_ledger": "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_terminal_20260903.csv",
            "interpretation": "Only instance 14 was unclassified; the earlier ten-classified description was incorrect.",
        },
    ]
    jobs = [row for row in jobs if int(row["unclassified_opportunities"]) > 0]
    jobs.extend(fo_rows)
    for row in fo_rows:
        instance_rows.append({
            **{key: row[key] for key in ("audit_scope", "domain", "experiment_scope", "value_head", "seed", "job_id", "slurm_state")},
            "instance_number": row["missing_instance_numbers"], "instance_path": row["missing_instance_paths"],
            "source_log": row["source_log"], "completion_ledger": row["completion_ledger"],
            "source_result_ledger": row["source_result_ledger"],
        })

    jobs.sort(key=lambda row: (row["domain"], row["experiment_scope"], int(row["job_id"])))
    instance_rows.sort(key=lambda row: (row["domain"], row["experiment_scope"], int(row["job_id"]), int(row["instance_number"])))
    job_fields = [
        "audit_scope", "domain", "experiment_scope", "value_head", "seed", "job_id", "slurm_state",
        "recorded_successes", "recorded_instance_timeouts", "classified_instances", "unclassified_opportunities",
        "missing_instance_numbers", "missing_instance_paths", "source_log", "completion_ledger",
        "completion_ledger_exists", "source_log_exists", "started_but_unclassified", "never_started",
        "source_result_ledger", "interpretation",
    ]
    instance_fields = [
        "audit_scope", "domain", "experiment_scope", "value_head", "seed", "job_id", "slurm_state",
        "instance_number", "instance_path", "source_log", "completion_ledger", "source_result_ledger",
    ]
    write_csv(TRACKING / "domain_test_order_20260907.csv", DOMAIN_ROWS, list(DOMAIN_ROWS[0]))
    write_csv(TRACKING / "mcts_interruption_opportunity_jobs_20260907.csv", jobs, job_fields)
    write_csv(TRACKING / "mcts_interruption_opportunity_instances_20260907.csv", instance_rows, instance_fields)

    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"jobs": 0, "classified": 0, "missing": 0})
    for row in jobs:
        key = (str(row["domain"]), str(row["experiment_scope"]))
        grouped[key]["jobs"] += 1
        grouped[key]["classified"] += int(row["classified_instances"])
        grouped[key]["missing"] += int(row["unclassified_opportunities"])
    summary = [
        {"domain": domain, "experiment_scope": scope, "interrupted_jobs_with_missing_instances": values["jobs"],
         "classified_instances": values["classified"], "unclassified_opportunities": values["missing"]}
        for (domain, scope), values in sorted(grouped.items())
    ]
    write_csv(TRACKING / "mcts_interruption_opportunity_summary_20260907.csv", summary, list(summary[0]))
    print(json.dumps({"job_rows": len(jobs), "instance_rows": len(instance_rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect bounded live/terminal experiment evidence through canonical SSH."""

from __future__ import annotations

import base64
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
SSH = [
    r"C:\Windows\System32\OpenSSH\ssh.exe",
    "-F",
    r"C:\Users\roeeh\.ssh\config",
    "uni-cluster",
]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def build_specs() -> list[dict[str, str]]:
    specs: dict[str, dict[str, str]] = {}

    def add(job_id: str, experiment: str, log_path: str = "") -> None:
        specs[job_id] = {
            "job_id": job_id,
            "experiment": experiment,
            "log_path": log_path,
        }

    for row in read_csv(TRACKING / "cluster_workload_latest.csv"):
        if row["experiment"] != "MPrime validation adequacy Phase B full rescore":
            add(row["job_id"], row["experiment"])

    # Complete Counters branch-completion family, including jobs that left the
    # queue between snapshots.
    for job_id in (
        20974537, 20974538, 20974539, 20974540, 20974542,
        20974543, 20974544, 20974545, 20974546, 20974353, 20974361,
    ):
        add(str(job_id), "Stage-2 MCTS branch completion — Counters")

    for row in read_csv(
        TRACKING / "mcts_progressive_widening_cross_domain" / "pw70_ten_seed_jobs_20260907.csv"
    ):
        add(row["job_id"], "PW70 ten-seed FO/Rover expansion", row["source_log"])

    for row in read_csv(TRACKING / "tpp_catastrophic_mcts" / "submissions_20260907.csv"):
        add(row["job_id"], "TPP catastrophic-seed MCTS diagnostic", row["evaluation_log"])

    for row in read_csv(
        TRACKING / "mcts_progressive_widening_cross_domain" / "counters_divergence_status_20260907.csv"
    ):
        add(row["pw70_job_id"], "Counters PW divergence recovery", row["pw70_log"])

    # The recovery output name is deterministic by array index.
    for row in read_csv(TRACKING / "rover_interrupted_mcts_recovery_manifest_20260908.csv"):
        index = row["array_index"]
        add(
            f"21107687_{index}",
            "Rover interrupted-instance MCTS recovery",
            "/home/hersco/training_new_domains/2026-09-08/"
            f"rover_interrupted_mcts_recovery/21107687_{index}.sbatch.out",
        )
    add(
        "21114859_0",
        "Rover interrupted-instance MCTS recovery follow-up",
        "/home/hersco/training_new_domains/2026-09-08/"
        "rover_interrupted_mcts_recovery_followup/21114859_0.sbatch.out",
    )
    add(
        "21114871_0",
        "Rover interrupted-instance MCTS recovery follow-up",
        "/home/hersco/training_new_domains/2026-09-08/"
        "rover_interrupted_mcts_recovery_followup/21114871_0.sbatch.out",
    )
    return list(specs.values())


def collect(specs: list[dict[str, str]]) -> dict[str, object]:
    remote = r'''
import csv, glob, json, re, subprocess
from pathlib import Path

specs = json.loads(%r)
ids = [row["job_id"] for row in specs]
fields = ["JobID", "JobIDRaw", "JobName", "State", "ExitCode", "Elapsed", "Start", "End", "NodeList", "ReqCPUS", "ReqMem", "StdOut"]
account = {}
for offset in range(0, len(ids), 150):
    text = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(ids[offset:offset+150]),
        "-o", "JobID%%80,JobIDRaw,JobName%%180,State,ExitCode,Elapsed,Start,End,NodeList,ReqCPUS,ReqMem,StdOut%%1000",
    ], text=True)
    for values in csv.reader(text.splitlines(), delimiter="|"):
        if len(values) < len(fields):
            continue
        row = dict(zip(fields, values))
        account[row["JobID"]] = row

instance_re = re.compile(r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)")
timeout_re = re.compile(r"\[EVAL INSTANCE\] timeout number=(\d+)")
final_re = re.compile(r"\[EVAL FINAL\].*?success=(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
valid_re = re.compile(r"VAL-valid plans\s*:\s*(\d+)")
invalid_re = re.compile(r"VAL-invalid plans\s*:\s*(\d+)")

def resolved_log(spec, acct):
    explicit = spec.get("log_path", "")
    if explicit and Path(explicit).is_file():
        return explicit
    raw = acct.get("StdOut", "")
    if not raw:
        return explicit
    job_id = acct.get("JobID", spec["job_id"])
    job_raw = acct.get("JobIDRaw", job_id)
    name = acct.get("JobName", "")
    if "_" in job_id and job_id.split("_")[-1].isdigit():
        array, task = job_id.rsplit("_", 1)
    else:
        array, task = job_raw, "4294967294"
    return (raw.replace("%%j", job_raw).replace("%%x", name)
               .replace("%%A", array).replace("%%a", task))

rows = []
for spec in specs:
    acct = account.get(spec["job_id"], {})
    path = resolved_log(spec, acct)
    records = {}
    explicit_timeouts = set()
    completion_ledgers = []
    final = None
    valid = invalid = ""
    raw_id = acct.get("JobIDRaw", spec["job_id"])
    for pattern in (
        f"/home/hersco/training_new_domains/*/*/completion/{raw_id}.jsonl",
        f"/home/hersco/training_new_domains/*/*/.resume_state/{raw_id}.eval_completed.jsonl",
    ):
        completion_ledgers.extend(glob.glob(pattern))
    for ledger in sorted(set(completion_ledgers)):
        with Path(ledger).open(errors="replace") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                number = item.get("instance_number", item.get("number"))
                if number is None:
                    continue
                number = int(number)
                status = str(item.get("status", "completed"))
                hit_goal = item.get("hit_goal", item.get("success", False))
                elapsed = float(item.get("elapsed_seconds", item.get("elapsed", 0.0)) or 0.0)
                candidate = {
                    "number": number,
                    "path": str(item.get("instance_path", item.get("path", ""))),
                    "status": status,
                    "elapsed": elapsed,
                    "success": 1.0 if hit_goal else 0.0,
                    "steps": int(item.get("steps", item.get("plan_length", 0)) or 0),
                }
                prior = records.get(number)
                if prior is None or candidate["success"] > prior["success"] or (
                    candidate["success"] == prior["success"] and candidate["elapsed"] < prior["elapsed"]
                ):
                    records[number] = candidate
                if "timeout" in status.lower():
                    explicit_timeouts.add(number)
    if path and Path(path).is_file():
        with Path(path).open(errors="replace") as stream:
            for line in stream:
                match = instance_re.search(line)
                if match:
                    number = int(match.group(1))
                    candidate = {
                        "number": number, "path": match.group(2), "status": match.group(3),
                        "elapsed": float(match.group(4)), "success": float(match.group(5)),
                        "steps": int(match.group(6)),
                    }
                    prior = records.get(number)
                    if prior is None or candidate["success"] > prior["success"] or (
                        candidate["success"] == prior["success"] and candidate["elapsed"] < prior["elapsed"]
                    ):
                        records[number] = candidate
                match = timeout_re.search(line)
                if match:
                    explicit_timeouts.add(int(match.group(1)))
                match = final_re.search(line)
                if match:
                    final = (float(match.group(1)), float(match.group(2)))
                match = valid_re.search(line)
                if match:
                    valid = match.group(1)
                match = invalid_re.search(line)
                if match:
                    invalid = match.group(1)
    successes = [row for row in records.values() if row["success"] == 1.0]
    rows.append({
        "experiment": spec["experiment"], "job_id": spec["job_id"],
        "job_name": acct.get("JobName", ""), "state": acct.get("State", "UNKNOWN").split()[0],
        "exit_code": acct.get("ExitCode", ""), "elapsed": acct.get("Elapsed", ""),
        "start": acct.get("Start", ""), "end": acct.get("End", ""),
        "node": acct.get("NodeList", ""), "requested_cpus": acct.get("ReqCPUS", ""),
        "requested_memory": acct.get("ReqMem", ""), "classified_instances": len(records) + len(explicit_timeouts - set(records)),
        "success_30m": sum(row["elapsed"] <= 1800 for row in successes),
        "success_2h": sum(row["elapsed"] <= 7200 for row in successes),
        "success_6h": sum(row["elapsed"] <= 21600 for row in successes),
        "success_full_record": len(successes), "explicit_instance_timeouts": len(explicit_timeouts),
        "latest_final_success": "" if final is None else final[0],
        "latest_final_total": "" if final is None else final[1],
        "val_valid": valid, "val_invalid": invalid, "source_evaluation_log": path,
        "source_completion_ledgers": ";".join(sorted(set(completion_ledgers))),
    })

mroot = Path("/home/hersco/training_new_domains/2026-09-06/mprime_phase_b")
done = list((mroot / "rescore").glob("*/*.done.json"))
summaries = list((mroot / "rescore").glob("*/*.val.csv"))
logs = list((mroot / "rescore").glob("*/*.log"))
lineage_counts = {}
for path in done:
    lineage_counts[path.parent.name] = lineage_counts.get(path.parent.name, 0) + 1
expected_counts = {}
inventory = mroot / "checkpoints.csv"
if inventory.is_file():
    with inventory.open(newline="") as stream:
        for item in csv.DictReader(stream):
            key = item.get("lineage_key", item.get("lineage", ""))
            if key:
                expected_counts[key] = expected_counts.get(key, 0) + 2
mprime = {
    "checkpoint_replicates_total": 2260,
    "done_markers": len(done), "val_summaries": len(summaries), "logs_created": len(logs),
    "lineages_started": len({path.parent.name for path in logs}),
    "lineages_with_results": len(lineage_counts),
    "lineages_complete": sum(
        lineage_counts.get(key, 0) >= expected
        for key, expected in expected_counts.items()
    ),
    "lineages_expected": len(expected_counts),
    "min_done_per_started_lineage": min(lineage_counts.values()) if lineage_counts else 0,
    "max_done_per_started_lineage": max(lineage_counts.values()) if lineage_counts else 0,
    "root": str(mroot),
}
print(json.dumps({"jobs": rows, "mprime": mprime}, separators=(",", ":")))
''' % json.dumps(specs)
    payload = base64.b64encode(remote.encode()).decode()
    command = f"python3 -c \"import base64;exec(base64.b64decode('{payload}'))\""
    result = subprocess.run(SSH + [command], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data = collect(build_specs())
    jobs = data["jobs"]
    write_csv(TRACKING / "dynamic_experiment_jobs_latest.csv", jobs)
    write_csv(TRACKING / "mprime_validation_phase_b_progress_latest.csv", [data["mprime"]])
    (TRACKING / "dynamic_experiment_state_latest.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(json.dumps({"jobs": len(jobs), "mprime": data["mprime"]}, indent=2))


if __name__ == "__main__":
    main()

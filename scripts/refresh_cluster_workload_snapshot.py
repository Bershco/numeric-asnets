#!/usr/bin/env python3
"""Write a compact, provenance-bearing snapshot of this user's live Slurm work."""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_tracking" / "cluster_workload_latest.csv"
SUMMARY_OUT = ROOT / "experiment_tracking" / "cluster_workload_summary_latest.csv"
SSH = [
    r"C:\Windows\System32\OpenSSH\ssh.exe",
    "-F",
    r"C:\Users\roeeh\.ssh\config",
    "uni-cluster",
]


def classify(name: str) -> str:
    name_upper = name.upper()
    rules = (
        ("PW-COUNTERS-DIVERGENCE", "Counters PW divergence recovery"),
        ("PW70-CONFIRM", "PW70 confirmatory expansion"),
        ("PW70", "PW70 cross-domain correction"),
        ("HOR", "Binding Horizon Counters"),
        ("SR10TCM", "FO Counters terminal-led Stage-2 MCTS"),
        ("MPEXT6VA", "MPrime validation-led Stage-2"),
        ("MPEXT6TA", "MPrime terminal-led Stage-2"),
        ("P3TERM", "PRESERVE-3 terminal-led Stage-2"),
        ("MPRIME_CORR_VAL_OFF", "MPrime validation-led off policy controller"),
        ("MPRIME_CORR_VAL_ON", "MPrime validation-led on policy controller"),
        ("MPRIME_CORR_", "MPrime policy controller"),
        ("MPEXT6VP", "MPrime validation-led policy evaluation"),
        ("MPEXT6TP", "MPrime terminal-led policy evaluation"),
        ("P3_TPP_", "PRESERVE-3 policy controller"),
        ("P3T_", "PRESERVE-3 policy evaluation"),
    )
    for needle, label in rules:
        if needle in name_upper:
            return label
    return "Other live work"


def mem_gib(value: str) -> float:
    suffix = value[-1:].upper()
    number = float(value[:-1] if suffix in "KMGT" else value)
    factors = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
    return number * factors.get(suffix, 1 / 1024)


raw = subprocess.check_output(
    SSH
    + [
        "squeue -u hersco -h -o '%i|%j|%T|%C|%m|%M|%l|%R|%Q|%A'"
    ],
    text=True,
    timeout=120,
)
rows = []
for line in raw.splitlines():
    fields = line.strip().split("|")
    if len(fields) != 10:
        continue
    job_id, name, state, cpus, memory, elapsed, limit, reason, priority, array_job = fields
    rows.append(
        {
            "snapshot_time_idt": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "experiment": classify(name),
            "job_id": job_id,
            "job_name": name,
            "state": state,
            "cpus": int(cpus),
            "memory_gib": round(mem_gib(memory), 3),
            "elapsed": elapsed,
            "time_limit": limit,
            "reason_or_node": reason,
            "priority": priority,
            "array_job_id": array_job,
            "scheduler_source": "squeue://hersco/" + job_id,
        }
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["snapshot_time_idt"])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {OUT}")

summary = {}
for row in rows:
    key = (row["experiment"], row["state"])
    bucket = summary.setdefault(
        key,
        {
            "snapshot_time_idt": row["snapshot_time_idt"],
            "experiment": row["experiment"],
            "state": row["state"],
            "jobs": 0,
            "requested_cpus": 0,
            "requested_memory_gib": 0.0,
            "scheduler_source": "experiment_tracking/cluster_workload_latest.csv",
        },
    )
    bucket["jobs"] += 1
    bucket["requested_cpus"] += row["cpus"]
    bucket["requested_memory_gib"] += row["memory_gib"]

summary_rows = sorted(summary.values(), key=lambda item: (item["state"], item["experiment"]))
for row in summary_rows:
    row["requested_memory_gib"] = round(row["requested_memory_gib"], 3)
with SUMMARY_OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else ["snapshot_time_idt"])
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"wrote {len(summary_rows)} grouped rows to {SUMMARY_OUT}")

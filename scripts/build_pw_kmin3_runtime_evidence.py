#!/usr/bin/env python3
"""Build job- and successful-instance runtime evidence for the Kmin=3 arm."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
OUT = TRACKING / "mcts_progressive_widening_sensitivity"
SEEDS = {"1073581256", "1239739722", "1472491096", "534933607"}
INSTANCE_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) "
    r"status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)"
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def elapsed_seconds(text: str) -> float:
    day = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        day = int(day_text)
    fields = [int(part) for part in text.split(":")]
    if len(fields) == 2:
        hours, minutes, seconds = 0, fields[0], fields[1]
    else:
        hours, minutes, seconds = fields
    return day * 86400 + hours * 3600 + minutes * 60 + seconds


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    policy = [
        row for row in read(TRACKING / "policy_endpoint_results.csv")
        if row["experiment_id"] == "MAIN-VAL" and row["domain"] == "drone"
        and row["stage"] == "stage1" and row["seed"] in SEEDS
    ]
    fixed = [
        row for row in read(TRACKING / "stage1_mcts_results.csv")
        if row["domain"] == "drone" and row["seed"] in SEEDS
    ]
    pw = [
        row for row in read(OUT / "results.csv")
        if row["variant"] == "safe_pw_kmin3" and row["seed"] in SEEDS
    ]
    jobs = []
    for row in policy:
        jobs.append({
            "arm": "policy", "value_head": row["value_head"],
            "seed": row["seed"], "job_id": row["evaluation_job_id"],
            "score": row["score"], "total": row["total"],
            "elapsed": row["evaluation_elapsed"],
            "source_checkpoint": row["checkpoint"],
            "source_log": row["source_evaluation_log"],
        })
    for row in fixed:
        jobs.append({
            "arm": "fixed_top20", "value_head": row["value_head"],
            "seed": row["seed"], "job_id": row["mcts_job_id"],
            "score": row["successes"], "total": row["total_instances"],
            "elapsed": row["elapsed"], "source_checkpoint": "",
            "source_log": row["source_evaluation_log"],
        })
    for row in pw:
        jobs.append({
            "arm": "pw_kmin3", "value_head": row["value_head"],
            "seed": row["seed"], "job_id": row["job_id"],
            "score": row["score"], "total": row["total"],
            "elapsed": row["elapsed"], "source_checkpoint": "",
            "source_log": row["source_evaluation_log"],
        })
    jobs.sort(key=lambda row: (row["arm"], row["value_head"], int(row["seed"])))
    instances = []
    for job in jobs:
        path = Path(job["source_log"])
        body = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        matches = INSTANCE_RE.findall(body)
        for number, instance, status, elapsed, success, steps in matches:
            if float(success) != 1.0:
                continue
            instances.append({
                "arm": job["arm"], "value_head": job["value_head"],
                "seed": job["seed"], "job_id": job["job_id"],
                "instance_number": number, "instance": Path(instance).name,
                "runtime_seconds": elapsed, "steps": steps,
                "source_log": job["source_log"],
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "kmin3_runtime_jobs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(jobs)
    instance_fields = [
        "arm", "value_head", "seed", "job_id", "instance_number", "instance",
        "runtime_seconds", "steps", "source_log",
    ]
    with (OUT / "kmin3_success_runtime_instances.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=instance_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(instances)
    summary = []
    for arm in ("policy", "fixed_top20", "pw_kmin3"):
        arm_jobs = [row for row in jobs if row["arm"] == arm]
        values = [
            float(row["runtime_seconds"]) for row in instances if row["arm"] == arm
        ]
        summary.append({
            "arm": arm, "jobs": len(arm_jobs),
            "mean_job_runtime_seconds": f"{statistics.fmean(elapsed_seconds(row['elapsed']) for row in arm_jobs):.3f}",
            "successful_instances_with_runtime": len(values),
            "success_runtime_min_seconds": f"{min(values):.3f}" if values else "",
            "success_runtime_q1_seconds": f"{percentile(values, .25):.3f}" if values else "",
            "success_runtime_median_seconds": f"{statistics.median(values):.3f}" if values else "",
            "success_runtime_mean_seconds": f"{statistics.fmean(values):.3f}" if values else "",
            "success_runtime_q3_seconds": f"{percentile(values, .75):.3f}" if values else "",
            "success_runtime_p90_seconds": f"{percentile(values, .90):.3f}" if values else "",
            "success_runtime_max_seconds": f"{max(values):.3f}" if values else "",
            "note": "historical logs lack per-instance runtime" if arm == "policy" and not values else "",
        })
    with (OUT / "kmin3_runtime_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    print(f"jobs={len(jobs)} successful_instance_rows={len(instances)}")


if __name__ == "__main__":
    main()

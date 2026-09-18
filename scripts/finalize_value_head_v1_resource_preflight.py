#!/usr/bin/env python3
"""Fail-closed resource gate for the eight six-state V1 preflights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


DOMAINS = ("drone", "fo_counters", "rover", "mprime")
STAGES = ("stage1", "stage2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_rss(value: str) -> int:
    units = {"K": 1, "M": 1024, "G": 1024 * 1024, "T": 1024 * 1024 * 1024}
    if not value:
        return 0
    suffix = value[-1]
    return int(float(value[:-1]) * units[suffix]) if suffix in units else int(value)


def accounting(job_ids: list[str]) -> dict[str, dict[str, object]]:
    proc = subprocess.run(
        ["sacct", "-j", ",".join(job_ids),
         "--format=JobIDRaw,State,ElapsedRaw,MaxRSS", "-n", "-P"],
        check=True, text=True, capture_output=True,
    )
    result = {job_id: {"states": set(), "elapsed": 0, "max_rss_kib": 0}
              for job_id in job_ids}
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        row_id, state, elapsed, rss = parts[:4]
        parent = next((job for job in job_ids if row_id == job or row_id.startswith(job + ".")), None)
        if parent is None:
            continue
        result[parent]["states"].add(state.split()[0])
        result[parent]["elapsed"] = max(int(elapsed or 0), int(result[parent]["elapsed"]))
        result[parent]["max_rss_kib"] = max(parse_rss(rss), int(result[parent]["max_rss_kib"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    summaries = []
    errors = []
    for domain in DOMAINS:
        for stage in STAGES:
            path = args.preflight_root / f"{domain}_{stage}" / "summary.json"
            if not path.is_file():
                errors.append(f"missing summary {domain}/{stage}")
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(summary)
            if summary.get("mode") != "preflight" or summary.get("states") != 6:
                errors.append(f"invalid six-state summary {domain}/{stage}")
            if int(summary.get("successors", 0)) <= 0:
                errors.append(f"no successors {domain}/{stage}")
            counts = summary.get("label_counts", {})
            if set(counts) != {"replay_target", "deterministic_continuation", "enhsp_raw_h", "enhsp_search_v"}:
                errors.append(f"label factorial mismatch {domain}/{stage}")
            for source in ("deterministic_continuation", "enhsp_raw_h", "enhsp_search_v"):
                if counts.get(source, {}).get("error", 0):
                    errors.append(f"provider error {domain}/{stage}/{source}")
            if counts.get("enhsp_raw_h") != counts.get("enhsp_search_v"):
                errors.append(f"raw/transformed ENHSP status mismatch {domain}/{stage}")
    if len(summaries) != 8:
        errors.append(f"expected eight summaries, got {len(summaries)}")
    job_ids = [str(summary.get("slurm_job_id") or "") for summary in summaries]
    if any(not job_id for job_id in job_ids) or len(set(job_ids)) != 8:
        errors.append("missing or duplicate Slurm component job IDs")
        acct = {}
    else:
        acct = accounting(job_ids)
    for summary in summaries:
        task_id = summary["task_id"]
        row = next(row for row in candidates if row["task_id"] == task_id)
        stats = acct.get(str(summary["slurm_job_id"]), {})
        if not stats or "COMPLETED" not in stats.get("states", set()):
            errors.append(f"accounting not complete for {task_id}")
            continue
        max_rss = max(int(stats["max_rss_kib"]), int(summary["max_rss_kib"]))
        memory_limit = int(row["memory_gib"]) * 1024 * 1024
        projected_seconds = max(
            float(summary["elapsed_seconds"]), float(stats["elapsed"])
        ) * 10.0
        time_limit = int(row["time_limit_hours"]) * 3600
        if max_rss <= 0 or max_rss > memory_limit * 0.8:
            errors.append(f"memory headroom failed for {task_id}: {max_rss}/{memory_limit} KiB")
        if projected_seconds > time_limit * 0.8:
            errors.append(f"runtime headroom failed for {task_id}: {projected_seconds}/{time_limit}s")
        summary["accounting_max_rss_kib"] = max_rss
        summary["projected_full_seconds"] = projected_seconds
    gate = {
        "schema": "value-head-audit-resource-gate-v1",
        "status": "pass" if not errors else "blocked",
        "candidate_manifest": str(args.candidates),
        "candidate_manifest_sha256": sha256(args.candidates),
        "summaries": summaries,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")
    print(("RESOURCE_GATE_PASS" if not errors else "RESOURCE_GATE_BLOCKED") + f"|{args.output}")
    if errors:
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

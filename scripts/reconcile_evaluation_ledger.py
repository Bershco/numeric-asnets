#!/usr/bin/env python3
"""Reconcile submitted evaluation jobs with Slurm state and final scores."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


FINAL_RE = re.compile(r"\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9]+)")
FIELDS = [
    "experiment_id", "manifest_id", "domain", "stage", "value_head", "seed",
    "arm", "iterations", "policy_score", "matched_fixed_width", "slurm_job_id",
    "source_training_job_id", "snapshot_epoch", "analysis_roles",
    "slurm_state", "elapsed", "score", "total", "source_log",
]


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {}
    for row in read(args.manifest, ","):
        manifest_id = row.get("manifest_id") or (
            f"context-full-{row['domain']}-{row['value_head']}-{row['seed']}-{row['arm']}"
        )
        manifest[manifest_id] = row
    submissions = read(args.submissions, "\t")
    ids = [row["slurm_job_id"] for row in submissions]
    text = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
         "-o", "JobIDRaw,State,Elapsed,JobName%200,StdOut%1000"], text=True
    )
    info: dict[str, tuple[str, str, Path]] = {}
    for line in text.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5 and parts[0].isdigit():
            resolved = parts[4].replace("%j", parts[0]).replace("%x", parts[3])
            info[parts[0]] = (parts[1].split()[0], parts[2], Path(resolved))
    rows = []
    for submitted in submissions:
        spec = manifest[submitted["manifest_id"]]
        job_id = submitted["slurm_job_id"]
        state, elapsed, log = info.get(job_id, ("UNKNOWN", "", Path()))
        score = total = ""
        if log.is_file():
            matches = FINAL_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
            if matches:
                score, total = matches[-1]
        rows.append({
            "experiment_id": spec.get("experiment_id", ""),
            "manifest_id": submitted["manifest_id"], "domain": spec.get("domain", ""),
            "stage": spec.get("stage", ""), "value_head": spec.get("value_head", ""),
            "seed": spec.get("seed", ""), "arm": spec.get("arm", "pw_kmin3"),
            "iterations": spec.get("iterations", ""), "policy_score": spec.get("policy_score", ""),
            "matched_fixed_width": spec.get("matched_fixed_width", ""),
            "slurm_job_id": job_id,
            "source_training_job_id": spec.get("source_training_job_id", ""),
            "snapshot_epoch": spec.get("snapshot_epoch", ""),
            "analysis_roles": spec.get("analysis_roles", ""),
            "slurm_state": state, "elapsed": elapsed,
            "score": score, "total": total, "source_log": str(log),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    terminal = sum(row["slurm_state"] not in {"RUNNING", "PENDING"} for row in rows)
    scored = sum(bool(row["score"]) for row in rows)
    print(f"rows={len(rows)} terminal={terminal} scored={scored} output={args.output}")


if __name__ == "__main__":
    main()

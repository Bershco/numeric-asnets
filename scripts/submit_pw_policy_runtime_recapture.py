#!/usr/bin/env python3
"""Submit eight policy-only timing recaptures for the PW Kmin=3 comparison."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")
SUBMITTER = ROOT / "submit_training.sh"
CHECKOUT = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")
MANIFEST = CHECKOUT / "experiment_tracking/mcts_progressive_widening_sensitivity/kmin3_runtime_jobs.csv"
LEDGER = CHECKOUT / "experiment_tracking/mcts_progressive_widening_sensitivity/kmin3_policy_runtime_recapture_submissions.tsv"
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")
FIELDS = ["value_head", "seed", "source_checkpoint", "slurm_job_id", "submitted_at"]


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def submit(row: dict[str, str], dry: bool = False) -> str:
    command = [
        str(SUBMITTER), "--dom-drone", "--original-only",
        "--domain-architecture", "mcts", "--seed", row["seed"],
        "--workers", "10", "--jpddl-max-heap", "4g",
        "--time", "04:00:00", "--mem", "20G", "--cpus", "6",
        "--eval-from", row["source_checkpoint"],
        "--job-suffix", f"PWK3POLRT_{row['value_head']}_{row['seed']}",
        "--output-subdir", "mcts_pw_kmin3_policy_runtime",
    ]
    if row["value_head"] == "off":
        command.append("--vh-off")
    if dry:
        command.append("--dry-run")
    env = os.environ.copy(); env["ENHSP_CONFIG_OVERRIDE"] = "hadd-gbfs"
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(proc.stdout)
    if dry:
        if "Expected jobs: 1" not in proc.stdout or "[DRY]" not in proc.stdout:
            raise RuntimeError(proc.stdout)
        return "DRY"
    matches = JOB_RE.findall(proc.stdout)
    if len(matches) != 1:
        raise RuntimeError(proc.stdout)
    return matches[0]


def main() -> None:
    rows = [row for row in read(MANIFEST) if row["arm"] == "policy"]
    if len(rows) != 8 or any(not Path(row["source_checkpoint"]).is_dir() for row in rows):
        raise RuntimeError("expected eight valid policy checkpoints")
    existing = {
        (row["value_head"], row["seed"])
        for row in read(LEDGER, "\t")
    } if LEDGER.exists() else set()
    submit(rows[0], dry=True)
    for row in rows:
        key = (row["value_head"], row["seed"])
        if key in existing:
            continue
        job_id = submit(row)
        new = not LEDGER.exists()
        with LEDGER.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow({
                "value_head": row["value_head"], "seed": row["seed"],
                "source_checkpoint": row["source_checkpoint"],
                "slurm_job_id": job_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
        existing.add(key)
        print(f"[SUBMITTED] {job_id} {key}", flush=True)
    print(f"[COMPLETE] submitted={len(existing)}", flush=True)


if __name__ == "__main__":
    main()

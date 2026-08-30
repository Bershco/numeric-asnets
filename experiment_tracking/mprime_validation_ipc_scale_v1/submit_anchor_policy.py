#!/usr/bin/env python3
"""Idempotently submit corrected-MPrime anchor policy evaluations."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CAMPAIGN = Path("/home/hersco/training_new_domains/2026-08-28/mprime_corrected_downstream")
MANIFEST = CAMPAIGN / "policy_ready.csv"
LEDGER = CAMPAIGN / "anchor_policy_submissions.tsv"
SUBMITTER = Path("/home/hersco/training_new_domains/submit_training.sh")
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")
FIELDS = ["manifest_id", "domain", "value_head", "seed", "anchor",
          "source_training_job_id", "snapshot_epoch", "slurm_job_id",
          "submitted_at", "source_checkpoint"]


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    rows = read(MANIFEST)
    existing = {row["manifest_id"] for row in read(LEDGER, "\t")}
    for row in rows:
        checkpoint = Path(row["source_checkpoint_ref"])
        if not checkpoint.is_dir():
            raise RuntimeError(f"missing checkpoint {checkpoint}")
        if row["manifest_id"] in existing:
            continue
        token = row["anchor"].replace(".", "p")
        command = [
            str(SUBMITTER), "--dom-mprime", "--original-only", "--domain-architecture", "mcts",
            "--seed", row["seed"], "--workers", "10", "--jpddl-max-heap", "4g",
            "--time", "04:00:00", "--mem", "20G", "--cpus", "6",
            "--eval-from", row["source_checkpoint_ref"],
            "--job-suffix", f"MPATPOL_a{token}_src{row['source_training_job_id']}_e{int(row['snapshot_epoch']):04d}",
            "--output-subdir", "mprime_corrected_anchor_policy_eval",
        ]
        if row["value_head"] == "off":
            command.append("--vh-off")
        env = os.environ.copy(); env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
        result = subprocess.run(command, cwd=SUBMITTER.parent, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)
        matches = JOB_RE.findall(result.stdout)
        if len(matches) != 1:
            raise RuntimeError(result.stdout)
        new = not LEDGER.exists()
        with LEDGER.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow({
                "manifest_id": row["manifest_id"], "domain": "mprime",
                "value_head": row["value_head"], "seed": row["seed"], "anchor": row["anchor"],
                "source_training_job_id": row["source_training_job_id"],
                "snapshot_epoch": row["snapshot_epoch"], "slurm_job_id": matches[0],
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "source_checkpoint": row["source_checkpoint_ref"],
            })
            stream.flush(); os.fsync(stream.fileno())
        existing.add(row["manifest_id"])
        print(f"[SUBMITTED] {matches[0]} {row['manifest_id']}", flush=True)
    print(f"[COMPLETE] manifest_rows={len(rows)} submitted={len(existing)}")


if __name__ == "__main__":
    main()

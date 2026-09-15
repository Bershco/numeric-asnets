#!/usr/bin/env python3
"""Submit exactly the frozen 20-lineage MPrime validation-led Stage-2 campaign."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2")
MANIFEST = WORK / "manifest.csv"
LEDGER = WORK / "submissions.tsv"
SUBMITTER = Path("/home/hersco/training_new_domains/submit_training.sh")
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main() -> None:
    rows = read_rows(MANIFEST)
    if len(rows) != 20 or len({(r["value_head"], r["seed"]) for r in rows}) != 20:
        raise RuntimeError("manifest must contain exactly twenty unique VH/seed lineages")
    existing: dict[tuple[str, str], str] = {}
    if LEDGER.exists():
        existing = {(r["value_head"], r["seed"]): r["slurm_job_id"] for r in read_rows(LEDGER, "\t")}
    fields = ["value_head", "seed", "anchor", "source_training_job_id", "source_epoch",
              "source_checkpoint", "slurm_job_id", "submitted_at"]
    with LEDGER.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if handle.tell() == 0:
            writer.writeheader()
        for row in rows:
            key = (row["value_head"], row["seed"])
            if key in existing:
                print(f"EXISTS|{key[0]}|{key[1]}|{existing[key]}")
                continue
            checkpoint = Path(row["source_checkpoint"])
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            command = [
                str(SUBMITTER), "--dom-mprime", "--original-only",
                "--domain-architecture", "mcts", "--seed", row["seed"],
                "--workers", "3", "--jpddl-max-heap", "4g", "--time", "3-00:00:00",
                "--mem", "48G", "--cpus", "6", "--train-from", row["source_checkpoint"],
                "--use-estimator", "0.5", "--exploration-weight", "0.1",
                "--override-tree-sampling", "0", "--mcts-expansion-size", "20",
                "--mcts-iterations", "0", "--policy-anchor-kl-coeff", row["anchor"],
                "--max-opt-epochs", "100", "--supervised-lr", "0.0003",
                "--job-suffix", f"MPFINALV_A{row['anchor']}_src{row['source_training_job_id']}_e{int(row['source_epoch']):04d}",
                "--output-subdir", "mprime_final_validation_stage2",
            ]
            if row["value_head"] == "off":
                command.append("--vh-off")
            env = os.environ.copy()
            env["ENHSP_CONFIG_OVERRIDE"] = "hmrp-ha-gbfs"
            result = subprocess.run(command, cwd=WORK, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(result.stdout)
            ids = JOB_RE.findall(result.stdout)
            if len(ids) != 1:
                raise RuntimeError(result.stdout)
            record = {
                "value_head": row["value_head"], "seed": row["seed"], "anchor": row["anchor"],
                "source_training_job_id": row["source_training_job_id"],
                "source_epoch": row["source_epoch"], "source_checkpoint": row["source_checkpoint"],
                "slurm_job_id": ids[0], "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
            writer.writerow(record)
            handle.flush()
            os.fsync(handle.fileno())
            existing[key] = ids[0]
            print(f"SUBMITTED|{key[0]}|{key[1]}|{ids[0]}")


if __name__ == "__main__":
    main()

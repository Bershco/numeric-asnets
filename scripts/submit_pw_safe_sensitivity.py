#!/usr/bin/env python3
"""Submit the SAFE+PW sensitivity manifest idempotently."""

import base64
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = Path("experiment_tracking/mcts_progressive_widening_sensitivity/manifest.csv")
LEDGER = Path("experiment_tracking/mcts_progressive_widening_sensitivity/submissions.tsv")
SBATCH = Path("/home/hersco/training_new_domains/2026-08-28/mcts_pw_safe_sensitivity/mcts_pw_safe_sensitivity.sbatch")


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    existing = set()
    if LEDGER.exists():
        with LEDGER.open(newline="", encoding="utf-8") as stream:
            existing = {row["manifest_id"] for row in csv.DictReader(stream, delimiter="\t")}
    fields = ("manifest_id", "slurm_job_id", "submitted_at")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        for row in rows:
            if row["manifest_id"] in existing:
                continue
            encoded = base64.b64encode(row["source_checkpoint"].encode()).decode()
            exports = ",".join((
                "ALL", f"CHECKPOINT_B64={encoded}", f"SEED={row['seed']}",
                f"VALUE_HEAD={row['value_head']}", f"MANIFEST_ID={row['manifest_id']}",
                f"PW_MIN={row['pw_min_width']}", f"PW_C={row['pw_c']}",
                f"PW_ALPHA={row['pw_alpha']}", f"ITERATIONS={row['iterations']}",
            ))
            result = subprocess.run(
                ["sbatch", f"--job-name={row['manifest_id']}",
                 f"--output=/home/hersco/training_new_domains/2026-08-28/mcts_pw_safe_sensitivity/%x_%j.out",
                 f"--export={exports}", str(SBATCH)],
                check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split()[-1]
            writer.writerow({"manifest_id": row["manifest_id"],
                             "slurm_job_id": job_id,
                             "submitted_at": datetime.now(timezone.utc).isoformat()})
            stream.flush()
            print(f"SUBMITTED|{row['manifest_id']}|{job_id}")


if __name__ == "__main__":
    main()

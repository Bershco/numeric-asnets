#!/usr/bin/env python3
"""Idempotently submit the frozen Counters horizon pilot."""

from __future__ import annotations

import base64, csv, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")
MANIFEST = ROOT / "experiment_tracking/mcts_horizon_binding/counters_manifest.csv"
LEDGER = ROOT / "experiment_tracking/mcts_horizon_binding/counters_submissions.tsv"
SBATCH = ROOT / "scripts/mcts_horizon_counters.sbatch"
OUT = Path("/home/hersco/training_new_domains/2026-08-31/mcts_horizon_counters")

def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    if len(rows) != 8: raise RuntimeError(f"expected 8 rows, got {len(rows)}")
    existing = set()
    if LEDGER.exists(): existing = {r["manifest_id"] for r in csv.DictReader(LEDGER.open(newline="", encoding="utf-8"), delimiter="\t")}
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["manifest_id", "slurm_job_id", "submitted_at"]
    with LEDGER.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0: writer.writeheader()
        for row in rows:
            if row["manifest_id"] in existing: continue
            encoded = base64.b64encode(row["source_checkpoint"].encode()).decode()
            exports = ",".join(("ALL", f"CHECKPOINT_B64={encoded}", f"SEED={row['seed']}",
                                f"VALUE_HEAD={row['value_head']}", f"ARM={row['arm']}",
                                f"MANIFEST_ID={row['manifest_id']}"))
            result = subprocess.run(["sbatch", f"--job-name={row['manifest_id']}",
                f"--output={OUT}/%x_%j.out", f"--export={exports}", str(SBATCH)],
                check=True, text=True, capture_output=True)
            job = result.stdout.strip().split()[-1]
            writer.writerow({"manifest_id": row["manifest_id"], "slurm_job_id": job,
                             "submitted_at": datetime.now(timezone.utc).isoformat()})
            stream.flush(); print(f"SUBMITTED|{row['manifest_id']}|{job}")

if __name__ == "__main__": main()

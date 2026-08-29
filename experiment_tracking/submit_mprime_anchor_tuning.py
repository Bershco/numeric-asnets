#!/usr/bin/env python3
"""Materialize and idempotently submit corrected-MPrime Stage-2 anchor tuning."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ANCHORS = ("0", "0.03", "0.3", "1", "3", "10", "30")
SEEDS = {"1963100312", "2011206605"}
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")
ROOT = Path("/home/hersco/training_new_domains")
SUBMITTER = ROOT / "submit_training.sh"


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_manifest", type=Path)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    sources = [row for row in read(args.policy_manifest)
               if row["seed"] in SEEDS and "selected" in row["roles"]]
    if len(sources) != 4:
        raise RuntimeError(f"expected four selected sources, got {len(sources)}")
    rows = []
    for source in sources:
        for anchor in ANCHORS:
            token = anchor.replace(".", "p")
            rows.append({
                "manifest_id": f"mprime-corrected-anchor-{source['value_head']}-{source['seed']}-a{token}",
                "domain": "mprime", "value_head": source["value_head"], "seed": source["seed"],
                "anchor": anchor, "source_training_job_id": source["training_job"],
                "snapshot_epoch": source["epoch"], "source_checkpoint": source["checkpoint"],
                "status": "ready", "teacher": "hmrp-ha-gbfs",
                "notes": "MPrime reclassified to imperfect mainstream; corrected validation-selected Stage-1 source",
            })
    args.campaign.mkdir(parents=True, exist_ok=True)
    manifest = args.campaign / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    ledger = args.campaign / "submissions.tsv"
    existing = {row["manifest_id"] for row in read(ledger, "\t")}
    fields = list(rows[0]) + ["slurm_job_id", "submitted_at"]
    for row in rows:
        if not Path(row["source_checkpoint"]).is_dir():
            raise RuntimeError(f"missing source: {row['source_checkpoint']}")
        if row["manifest_id"] in existing:
            continue
        token = row["anchor"].replace(".", "p")
        command = [
            str(SUBMITTER), "--dom-mprime", "--original-only", "--domain-architecture", "mcts",
            "--seed", row["seed"], "--workers", "3", "--jpddl-max-heap", "4g",
            "--time", "3-00:00:00", "--mem", "48G", "--cpus", "6",
            "--train-from", row["source_checkpoint"], "--use-estimator", "0.5",
            "--exploration-weight", "0.1", "--override-tree-sampling", "0",
            "--mcts-expansion-size", "20", "--mcts-iterations", "0",
            "--policy-anchor-kl-coeff", row["anchor"], "--max-opt-epochs", "100",
            "--supervised-lr", "0.0003",
            "--job-suffix", f"MPCAT_src{row['source_training_job_id']}_e{int(row['snapshot_epoch']):04d}_a{token}",
            "--output-subdir", "mprime_corrected_anchor_tuning",
        ]
        if row["value_head"] == "off":
            command.append("--vh-off")
        env = os.environ.copy(); env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
        output = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if output.returncode:
            raise RuntimeError(output.stdout)
        job_ids = JOB_RE.findall(output.stdout)
        if len(job_ids) != 1:
            raise RuntimeError(output.stdout)
        new = not ledger.exists()
        with ledger.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow({**row, "slurm_job_id": job_ids[0],
                             "submitted_at": datetime.now(timezone.utc).isoformat()})
            stream.flush(); os.fsync(stream.fileno())
        existing.add(row["manifest_id"])
        print(f"[SUBMITTED] {job_ids[0]} {row['manifest_id']}", flush=True)
        time.sleep(0.1)
    print(f"[COMPLETE] manifest_rows={len(rows)} submitted={len(existing)}")


if __name__ == "__main__":
    main()

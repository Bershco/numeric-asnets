#!/usr/bin/env python3
"""Validate and idempotently submit corrected-validation MPrime Stage 1."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")
SUBMITTER = ROOT / "submit_training.sh"
REPO = Path("/home/hersco/bershco-nu-asnets/numeric-asnets")
EXPECTED_CODE_COMMIT = "b6985638"
EXPECTED_MODULE = "experiments_numeric.domain.mprime_validation_ipc_scale_v1"
EXPECTED_SEEDS = {
    "1963100312", "2011206605", "1073581256", "1239739722",
    "1472491096", "534933607", "2082152039", "1510771779",
    "923500475", "1972442430",
}
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows, found {len(rows)}")
    signatures = set()
    for row in rows:
        expected = {
            "domain": "mprime",
            "architecture": "experiments_numeric.architecture_2.mprime",
            "problem_module": EXPECTED_MODULE,
            "teacher": "hmrp-ha-gbfs",
            "original_training_set": "true",
            "supervised_lr": "0.003",
            "max_epochs": "1000",
            "workers": "3",
            "jpddl_heap": "4g",
            "cpus": "6",
            "memory": "48G",
            "time_limit": "1-00:00:00",
            "status": "ready",
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(f"{row['manifest_id']}: {field}={row[field]!r}")
        if row["seed"] not in EXPECTED_SEEDS or row["value_head"] not in {"off", "on"}:
            raise ValueError(f"{row['manifest_id']}: invalid seed/VH")
        signatures.add((row["value_head"], row["seed"]))
    if signatures != {(vh, seed) for vh in ("off", "on") for seed in EXPECTED_SEEDS}:
        raise ValueError("Manifest does not contain the complete 2x10 grid")


def command(row: dict[str, str], dry_run: bool) -> tuple[list[str], dict[str, str]]:
    cmd = [
        str(SUBMITTER), "--dom-mprime", "--original-only",
        "--domain-architecture", "policy", "--seed", row["seed"],
        "--workers", row["workers"], "--jpddl-max-heap", row["jpddl_heap"],
        "--time", row["time_limit"], "--mem", row["memory"],
        "--cpus", row["cpus"], "--max-opt-epochs", row["max_epochs"],
        "--supervised-lr", row["supervised_lr"],
        "--job-suffix", "_MPRIME_VAL_V1_S1",
        "--output-subdir", "mprime_validation_ipc_scale_v1_stage1",
    ]
    if row["value_head"] == "off":
        cmd.append("--vh-off")
    if dry_run:
        cmd.append("--dry-run")
    env = os.environ.copy()
    env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
    env["PROBLEM_MODULE_OVERRIDE"] = row["problem_module"]
    return cmd, env


def submit(row: dict[str, str], dry_run: bool) -> str:
    cmd, env = command(row, dry_run)
    result = subprocess.run(
        cmd, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(f"{row['manifest_id']} failed:\n{result.stdout}")
    if dry_run:
        if "Expected jobs: 1" not in result.stdout or "[DRY]" not in result.stdout:
            raise RuntimeError(f"Unexpected dry run:\n{result.stdout}")
        return "DRY"
    matches = JOB_RE.findall(result.stdout)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one job ID, found {matches}:\n{result.stdout}")
    return matches[0]


def existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["manifest_id"] for row in csv.DictReader(stream, delimiter="\t")}


def record(path: Path, row: dict[str, str], job_id: str) -> None:
    fields = ("manifest_id", "value_head", "seed", "problem_module", "slurm_job_id", "submitted_at")
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({
            "manifest_id": row["manifest_id"], "value_head": row["value_head"],
            "seed": row["seed"], "problem_module": row["problem_module"],
            "slurm_job_id": job_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != EXPECTED_CODE_COMMIT:
        raise RuntimeError(f"Cluster repository is {head}; expected {EXPECTED_CODE_COMMIT}")

    rows = load(args.manifest)
    validate(rows)
    for vh in ("off", "on"):
        submit(next(row for row in rows if row["value_head"] == vh), dry_run=True)
    print(f"[VALID] rows=20 dry_runs=2 code={head}", flush=True)

    done = existing(args.ledger)
    for row in rows:
        if row["manifest_id"] in done:
            continue
        job_id = submit(row, dry_run=False)
        record(args.ledger, row, job_id)
        print(f"[SUBMITTED] {job_id} {row['manifest_id']}", flush=True)
    print(f"[COMPLETE] ledger_rows={len(existing(args.ledger))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

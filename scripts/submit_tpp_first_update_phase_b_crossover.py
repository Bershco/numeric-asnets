#!/usr/bin/env python3
"""Idempotently submit the frozen-replay cross-over and endpoint dependency."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path("/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover")
TRAIN = ROOT / "tpp_first_update_phase_b_crossover_train.sbatch"
ENDPOINT = ROOT / "tpp_first_update_phase_b_crossover_endpoint.sbatch"
SMOKE = ROOT / "tpp_first_update_phase_b_crossover_smoke.sbatch"
FREEZER = ROOT / "freeze_tpp_phase_a_batch_checksums.py"
CHECKSUMS = ROOT / "frozen_schedule_checksums.csv"
LEDGER = ROOT / "submissions.tsv"


def submit(script: Path, *extra: str) -> str:
    output = subprocess.check_output(
        ["sbatch", "--parsable", *extra, str(script)], text=True).strip()
    return output.split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit; omission is a safe validation-only dry run.")
    args = parser.parse_args()
    for path in (SMOKE, TRAIN, ENDPOINT, FREEZER, CHECKSUMS):
        if not path.is_file():
            raise SystemExit(f"Missing deployed script: {path}")
    if LEDGER.exists():
        raise SystemExit(
            f"Refusing duplicate submission; ledger already exists: {LEDGER}")
    if not args.submit:
        print(f"DRY RUN: python3 {FREEZER} --verify --manifest {CHECKSUMS}")
        print(f"DRY RUN: sbatch --parsable {SMOKE}")
        print(f"DRY RUN: sbatch --parsable --dependency=afterok:<smoke> {TRAIN}")
        print(f"DRY RUN: sbatch --parsable --dependency=afterok:<train> {ENDPOINT}")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python3", str(FREEZER), "--verify", "--manifest", str(CHECKSUMS)])
    smoke_job = submit(SMOKE)
    training_job = submit(
        TRAIN, f"--dependency=afterok:{smoke_job}")
    endpoint_job = submit(
        ENDPOINT, f"--dependency=afterok:{training_job}")
    LEDGER.write_text(
        "role\tjob_id\tdependency\tscript\n"
        f"smoke\t{smoke_job}\t\t{SMOKE}\n"
        f"training\t{training_job}\tafterok:{smoke_job}\t{TRAIN}\n"
        f"endpoint\t{endpoint_job}\tafterok:{training_job}\t{ENDPOINT}\n",
        encoding="utf-8",
    )
    print(f"training={training_job} endpoint={endpoint_job}")


if __name__ == "__main__":
    main()

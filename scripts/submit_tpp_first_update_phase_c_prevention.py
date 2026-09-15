#!/usr/bin/env python3
"""Validate or submit the Phase-C smoke, treatment, and endpoint chain."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path("/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt")
SMOKE = ROOT / "tpp_first_update_phase_c_prevention_smoke.sbatch"
TRAIN = ROOT / "tpp_first_update_phase_c_prevention_train.sbatch"
ENDPOINT = ROOT / "tpp_first_update_phase_c_prevention_endpoint.sbatch"
FREEZER = ROOT / "freeze_tpp_phase_a_batch_checksums.py"
THRESHOLD_VERIFIER = ROOT / "verify_tpp_phase_c_stable_thresholds.py"
CALIBRATOR = ROOT / "calibrate_tpp_phase_c_thresholds.py"
CHECKSUMS = Path("/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv")
LEDGER = ROOT / "submissions.tsv"


def submit(script: Path, checkout: Path, *extra: str) -> str:
    output = subprocess.check_output([
        "sbatch", "--parsable", *extra,
        f"--export=ALL,CHECKOUT={checkout}", str(script),
    ], text=True).strip()
    return output.split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit; omission is a validation-only dry run.")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    for path in (
            SMOKE, TRAIN, ENDPOINT, FREEZER, THRESHOLD_VERIFIER, CALIBRATOR,
            CHECKSUMS):
        if not path.is_file():
            raise SystemExit(f"Missing deployed input: {path}")
    for path in (
            checkout / "asnets/asnets/policy_anchor_trust_region.py",
            checkout / "asnets/asnets/supervised.py"):
        if not path.is_file():
            raise SystemExit(f"Missing Phase-C implementation: {path}")
    if LEDGER.exists():
        raise SystemExit(f"Refusing duplicate submission: {LEDGER}")
    commands = (
        f"sbatch --parsable --export=ALL,CHECKOUT={checkout} {SMOKE}",
        f"sbatch --parsable --dependency=afterok:<smoke> --export=ALL,CHECKOUT={checkout} {TRAIN}",
        f"sbatch --parsable --dependency=afterok:<training> --export=ALL,CHECKOUT={checkout} {ENDPOINT}",
    )
    if not args.submit:
        print("DRY RUN: python3", FREEZER, "--verify --manifest", CHECKSUMS)
        for command in commands:
            print("DRY RUN:", command)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python3", str(FREEZER), "--verify", "--manifest", str(CHECKSUMS)])
    smoke_job = submit(SMOKE, checkout)
    training_job = submit(
        TRAIN, checkout, f"--dependency=afterok:{smoke_job}")
    endpoint_job = submit(
        ENDPOINT, checkout, f"--dependency=afterok:{training_job}")
    LEDGER.write_text(
        "role\tjob_id\tdependency\tscript\tcheckout\n"
        f"smoke\t{smoke_job}\t\t{SMOKE}\t{checkout}\n"
        f"training\t{training_job}\tafterok:{smoke_job}\t{TRAIN}\t{checkout}\n"
        f"endpoint\t{endpoint_job}\tafterok:{training_job}\t{ENDPOINT}\t{checkout}\n",
        encoding="utf-8",
    )
    print(f"smoke={smoke_job} training={training_job} endpoint={endpoint_job}")


if __name__ == "__main__":
    main()

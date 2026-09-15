#!/usr/bin/env python3
"""Validate or submit the exact-RNG TPP selected-pair closure chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path("/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_c_exact_rng")
SMOKE = ROOT / "tpp_first_update_phase_c_exact_rng_smoke_20260916.sbatch"
TRAIN = ROOT / "tpp_first_update_phase_c_exact_rng_train_20260916.sbatch"
ENDPOINT = ROOT / "tpp_first_update_phase_c_exact_rng_endpoint_20260916.sbatch"
FREEZER = ROOT / "freeze_tpp_phase_a_batch_checksums.py"
RNG_SMOKE = ROOT / "smoke_tpp_exact_rng_retry.py"
CHECKSUMS = Path("/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv")
THRESHOLDS = Path("/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt/calibrated_thresholds.json")
LEDGER = ROOT / "submissions.tsv"


def submit(script: Path, checkout: Path, commit: str, *extra: str) -> str:
    output = subprocess.check_output([
        "sbatch", "--parsable", *extra,
        f"--export=ALL,CHECKOUT={checkout},CODE_COMMIT={commit}", str(script),
    ], text=True).strip()
    return output.split(";", 1)[0]


def append_ledger(row: str) -> None:
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(row)
        stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    required = (
        SMOKE, TRAIN, ENDPOINT, FREEZER, RNG_SMOKE, CHECKSUMS, THRESHOLDS)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing deployed input: {path}")
    for relative in (
        "asnets/asnets/replay_rng.py",
        "asnets/asnets/supervised.py",
        "asnets/asnets/scripts/run_asnets.py",
        "asnets/asnets/scripts/run_experiment.py",
        "asnets/tests/test_replay_rng.py",
    ):
        path = checkout / relative
        if not path.is_file():
            raise SystemExit(f"Missing exact-RNG implementation: {path}")
    if LEDGER.exists():
        raise SystemExit(f"Refusing duplicate submission: {LEDGER}")
    subprocess.check_call([
        "python3", str(FREEZER), "--verify", "--manifest", str(CHECKSUMS)])
    if not args.submit:
        print(f"DRY RUN: smoke={SMOKE}")
        print(f"DRY RUN: training={TRAIN} dependency=afterok:<smoke>")
        print(f"DRY RUN: endpoint={ENDPOINT} dependency=afterok:<training>")
        print("DRY RUN: scientific tasks=4 (2 one-epoch + 2 endpoints)")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    submitted_at = datetime.now(timezone.utc).isoformat()
    LEDGER.write_text(
        "submitted_at_utc\trole\tjob_id\ttasks\tdependency\tscript\tcheckout\tcommit\n",
        encoding="utf-8")
    smoke_job = submit(SMOKE, checkout, commit)
    append_ledger(
        f"{submitted_at}\tsmoke\t{smoke_job}\t1\t\t{SMOKE}\t{checkout}\t{commit}\n")
    training_job = submit(
        TRAIN, checkout, commit, f"--dependency=afterok:{smoke_job}")
    append_ledger(
        f"{submitted_at}\ttraining\t{training_job}\t2\tafterok:{smoke_job}\t{TRAIN}\t{checkout}\t{commit}\n")
    endpoint_job = submit(
        ENDPOINT, checkout, commit, f"--dependency=afterok:{training_job}")
    append_ledger(
        f"{submitted_at}\tendpoint\t{endpoint_job}\t2\tafterok:{training_job}\t{ENDPOINT}\t{checkout}\t{commit}\n")
    print(
        f"smoke={smoke_job} training={training_job} endpoint={endpoint_job} "
        f"scientific_tasks=4 commit={commit}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Submit only incomplete MPrime anchor-rescore lineages, then recheck.

This controller is intentionally idempotent: a lineage task may be retried, but
the evaluator itself skips every checkpoint whose identity-matched done marker
already exists.  Once all 28 x 21 points exist, the analysis-only finalizer is
submitted.  It never submits Stage-2 training.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")
WORK = Path("/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a")
MANIFEST = ROOT / "experiment_tracking/mprime_anchor_phase_b_a_20260912/manifest.csv"
RESCORE_BATCH = ROOT / "scripts/mprime_anchor_phase_b_a_rescore_20260912.sbatch"
FINALIZER = ROOT / "scripts/mprime_anchor_phase_b_a_finalize_20260912.sbatch"
SELF = ROOT / "scripts/continue_mprime_anchor_phase_b_a_20260913.py"
LEDGER = WORK / "continuation_20260913.jsonl"


def submit(*args: str) -> str:
    output = subprocess.check_output(["sbatch", "--parsable", *args], text=True).strip()
    return output.split(";", 1)[0]


def record(payload: dict[str, object]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def missing_indices() -> tuple[list[int], int]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 28:
        raise RuntimeError(f"expected 28 manifest rows, got {len(rows)}")
    missing: list[int] = []
    complete_points = 0
    for row in rows:
        output = WORK / "rescore" / row["manifest_id"]
        done = list(output.glob("epoch_*_phase_b_a.done.json"))
        summaries = list(output.glob("epoch_*_phase_b_a.val.csv"))
        paired = {
            path.name.removesuffix(".done.json")
            for path in done
        } & {
            path.name.removesuffix(".val.csv")
            for path in summaries
        }
        complete_points += len(paired)
        if len(paired) != 21:
            missing.append(int(row["array_index"]))
    return missing, complete_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    missing, complete_points = missing_indices()
    payload: dict[str, object] = {
        "attempt": args.attempt,
        "complete_points": complete_points,
        "missing_indices": missing,
    }
    if not missing:
        payload["finalizer_job_id"] = submit(str(FINALIZER))
        record(payload)
        print(json.dumps(payload, sort_keys=True))
        return
    if args.attempt > args.max_attempts:
        payload["status"] = "manual_attention_required"
        record(payload)
        raise RuntimeError(
            f"still incomplete after {args.max_attempts} retries: {missing}"
        )
    array_spec = ",".join(map(str, missing))
    rescore_job = submit(
        f"--array={array_spec}",
        "--nice=10000",
        "--exclude=ise-cpu128-03,ise-cpu-intl-01,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-27",
        "--export=ALL,PREFLIGHT=0",
        str(RESCORE_BATCH),
    )
    controller_job = submit(
        f"--dependency=afterany:{rescore_job}",
        "--job-name=MPRIME_PBA_RECHECK",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:15:00",
        f"--output={WORK}/recheck_%j.log",
        "--wrap",
        (
            "source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate "
            f"&& python -u {SELF} --attempt {args.attempt + 1} "
            f"--max-attempts {args.max_attempts}"
        ),
    )
    payload.update({
        "rescore_job_id": rescore_job,
        "recheck_job_id": controller_job,
        "status": "rescore_submitted",
    })
    record(payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

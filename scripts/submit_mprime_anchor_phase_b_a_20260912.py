#!/usr/bin/env python3
"""Idempotently submit the smoke, gated full rescore, and analysis-only finalizer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")
WORK = Path("/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a")
LEDGER = WORK / "submission.json"


def submit(*args: str) -> str:
    result = subprocess.check_output(["sbatch", "--parsable", *args], text=True).strip()
    return result.split(";", 1)[0]


def main() -> None:
    manifest = ROOT / "experiment_tracking/mprime_anchor_phase_b_a_20260912/manifest.csv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    if len(rows) != 29:
        raise RuntimeError(f"expected header plus 28 manifest rows, got {len(rows)}")
    WORK.mkdir(parents=True, exist_ok=True)
    state = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    batch = str(ROOT / "scripts/mprime_anchor_phase_b_a_rescore_20260912.sbatch")
    if "preflight" not in state:
        state["preflight"] = submit(
            "--array=0", "--time=02:00:00", "--export=ALL,PREFLIGHT=1", batch
        )
        LEDGER.write_text(json.dumps(state, indent=2, sort_keys=True))
    if "rescore" not in state:
        state["rescore"] = submit(
            f"--dependency=afterok:{state['preflight']}",
            "--array=0-27", "--export=ALL,PREFLIGHT=0", batch,
        )
        LEDGER.write_text(json.dumps(state, indent=2, sort_keys=True))
    if "finalizer" not in state:
        finalizer = str(ROOT / "scripts/mprime_anchor_phase_b_a_finalize_20260912.sbatch")
        state["finalizer"] = submit(
            f"--dependency=afterok:{state['rescore']}", finalizer
        )
        LEDGER.write_text(json.dumps(state, indent=2, sort_keys=True))
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

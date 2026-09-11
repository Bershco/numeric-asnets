"""Freeze Phase C, create its domain module, and submit gated rescoring."""

import csv
import json
import subprocess
from pathlib import Path

from mprime_phase_c_20260911 import freeze


ROOT = Path("/home/hersco/training_new_domains/2026-09-11/mprime_phase_c")
REPO = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")
freeze(ROOT)
rows = list(csv.DictReader((ROOT / "frozen_validation_manifest.csv").open()))
paths = [str(ROOT / "candidates" / row["file"]) for row in rows]
module = REPO / "asnets/experiments_numeric/domain/mprime_phase_c_20260911.py"
content = (
    "from experiments_numeric.domain.mprime import *\n"
    + "TEST_RUNS = " + repr([([path], None) for path in paths]) + "\n"
    + "VALIDATION_PDDLS = " + repr({"hard": paths}) + "\n"
)
if module.exists():
    assert module.read_text() == content
else:
    module.write_text(content)
ledger = ROOT / "rescore_submission.json"
if ledger.exists():
    raise SystemExit("already submitted: " + ledger.read_text())
batch = str(ROOT / "mprime_phase_c_rescore_20260911.sbatch")
preflight = subprocess.check_output(
    ["sbatch", "--parsable", "--export=ALL,PREFLIGHT=1", "--time=02:00:00", batch], text=True,
).strip().split(";")[0]
ledger.write_text(json.dumps({"preflight": preflight}))
full = subprocess.check_output(
    [
        "sbatch", "--parsable", f"--dependency=afterok:{preflight}",
        "--array=0-39", "--export=ALL,PREFLIGHT=0", batch,
    ], text=True,
).strip().split(";")[0]
ledger.write_text(json.dumps({"preflight": preflight, "rescore": full}))
print(ledger.read_text())

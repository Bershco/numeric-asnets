#!/usr/bin/env bash
set -euo pipefail

work=/home/hersco/training_new_domains/2026-08-31/mprime_anchor_corrected_validation
manifest="$work/anchor_selection_invalid.csv"
batch=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context/scripts/mprime_anchor_corrected_validation_rescore.sbatch

python - "$manifest" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
expected = {
    0: ("off", "1963100312", "0"),
    4: ("off", "1963100312", "3"),
    15: ("on", "1963100312", "0.03"),
}
for index, wanted in expected.items():
    row = rows[index]
    got = (row["value_head"], row["seed"], row["anchor"])
    if got != wanted:
        raise SystemExit(f"array index {index}: expected {wanted}, got {got}")
print("[MPRIME CONTINUATION] manifest indices verified: 0,4,15")
PY

job_id=$(sbatch --parsable \
  --array=0,4,15 \
  --job-name=MPRIME_ANCHOR_VAL_CONT \
  --exclude=ise-cpu-intl-[11-12,18] \
  --output="$work/rescore_cont_%A_%a.log" \
  "$batch")
echo "continuation_job_id=$job_id array_indices=0,4,15"

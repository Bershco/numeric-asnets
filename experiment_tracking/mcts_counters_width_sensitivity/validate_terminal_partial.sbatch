#!/bin/bash
#SBATCH --job-name=COUNTERS_NARROW_POSTHOC_VAL
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-30/counters_narrow_posthoc_val/controller_%j.out

set -euo pipefail
: "${SOURCE_JOB_ID:?}"
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
VALIDATOR=/home/hersco/tools/VAL/build-yydepth12000/bin/Validate
OUT=/home/hersco/training_new_domains/2026-08-30/counters_narrow_posthoc_val
mkdir -p "$OUT"
SOURCE_LOG=$(sacct -j "$SOURCE_JOB_ID" -X -n -P -o StdOut | head -n1)
SOURCE_LOG=${SOURCE_LOG//%j/$SOURCE_JOB_ID}
[[ -s "$SOURCE_LOG" ]] || { echo "missing source log: $SOURCE_LOG" >&2; exit 2; }
python "$CHECKOUT/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$SOURCE_LOG" --domain counters --validator "$VALIDATOR" \
  --allow-incomplete --summary-csv "$OUT/${SOURCE_JOB_ID}_summary.csv"

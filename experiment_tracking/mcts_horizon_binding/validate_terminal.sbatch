#!/bin/bash
#SBATCH --job-name=HORIZON_POSTHOC_VAL
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
: "${SOURCE_JOB_ID:?}" "${MANIFEST_ID:?}"
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
OUT=/home/hersco/training_new_domains/2026-08-28/mcts_horizon_binding
LOG="$OUT/${SOURCE_JOB_ID}_${MANIFEST_ID}.txt"
[[ -s "$LOG" ]] || { echo "missing inference log: $LOG" >&2; exit 2; }
python "$CHECKOUT/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$LOG" --domain drone --validator "$VALIDATOR"

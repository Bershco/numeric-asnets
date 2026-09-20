#!/usr/bin/env bash
set -euo pipefail

FEATURE_REPO=${FEATURE_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe}
OUT=${OUT:-/home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_vhoff_all_imperfect}
MANIFEST="$FEATURE_REPO/experiment_tracking/mcts_leaf_evaluator_vhoff_all_imperfect_20260920/manifest.csv"
DONE="$OUT/smoke/done"
COMPATIBILITY="$OUT/compatibility.json"
SUBMISSION="$OUT/science_submission.txt"
RUNNER="$FEATURE_REPO/scripts/mcts_leaf_evaluator_vhoff_all_imperfect.sbatch"
EXCLUDE='ise-cpu-intl-[01,05,08-15,18,24-28]'

if [[ -s "$SUBMISSION" ]]; then
  echo "science already submitted: $(cat "$SUBMISSION")"
  exit 0
fi

python "$FEATURE_REPO/scripts/verify_mcts_leaf_evaluator_vhoff_all_imperfect.py" \
  "$MANIFEST" --smoke-done "$DONE" --write-compatibility "$COMPATIBILITY"

job_id=$(sbatch --parsable --job-name=LEAF6_VHOFF_SCI --array=0-23 \
  --exclude="$EXCLUDE" --output="$OUT/slurm/%A_%a.out" \
  --export=ALL,FEATURE_REPO="$FEATURE_REPO" "$RUNNER")
printf '%s\n' "$job_id" > "$SUBMISSION"
echo "science submitted: $job_id"

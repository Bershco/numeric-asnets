#!/bin/bash
#SBATCH --job-name=MPRIME_S2_POLICY_CTL
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2_policy/controller_%j.out

set -euo pipefail

CAMPAIGN=/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2_policy
TRAINING=/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-mprime-policy-live-20260915
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
READY="$CAMPAIGN/ready_manifest.csv"
LEDGER="$CAMPAIGN/submissions.tsv"
export STAGE2_POLICY_SUBMITTER="$CAMPAIGN/submit_training_mprime_policy_wrapper_20260916.sh"

mkdir -p "$CAMPAIGN"
while true; do
  scan=$(
    "$PYTHON" "$CHECKOUT/experiment_tracking/materialize_live_stage2_policy.py" \
      --training-manifest "$TRAINING/manifest.csv" \
      --training-ledger "$TRAINING/submissions.tsv" \
      --output "$READY" \
      --role-prefix mprime_final_validation_stage2
  )
  printf '[SCAN] %s\n' "$scan"
  "$PYTHON" "$CHECKOUT/experiment_tracking/submit_stage2_policy.py" \
    --manifest "$READY" \
    --ledger "$LEDGER" \
    --suffix-prefix MPFINALV_S2POL \
    --output-prefix mprime_final_validation_stage2_policy \
    --queue-cap 1999 \
    --max-per-cycle 24 \
    --max-active 24 \
    --one-cycle
  if [[ "$scan" == *"complete=20"* ]]; then
    rows=$(($(wc -l < "$READY") - 1))
    submitted=$(if [[ -f "$LEDGER" ]]; then echo $(($(wc -l < "$LEDGER") - 1)); else echo 0; fi)
    if (( rows == submitted )); then
      printf '[COMPLETE] terminal_lineages=20 submitted_rows=%d\n' "$submitted"
      exit 0
    fi
  fi
  sleep 300
done

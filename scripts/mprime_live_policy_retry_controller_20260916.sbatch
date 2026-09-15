#!/bin/bash
#SBATCH --job-name=MPRIME_S2_POL_RETRY
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2_policy/retry_controller_%j.out

set -euo pipefail

CAMPAIGN=/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2_policy
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-mprime-policy-live-20260915
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
export STAGE2_POLICY_SUBMITTER="$CAMPAIGN/submit_training_mprime_policy_wrapper_20260916.sh"

while true; do
  "$PYTHON" "$CHECKOUT/experiment_tracking/retry_failed_stage2_policy.py" \
    --manifest "$CAMPAIGN/ready_manifest.csv" \
    --primary-ledger "$CAMPAIGN/submissions.tsv" \
    --retry-ledger "$CAMPAIGN/retry_submissions.tsv" \
    --suffix-prefix MPFINALV_S2POL_R \
    --output-prefix mprime_final_validation_stage2_policy \
    --max-attempts 3 \
    --max-active 36 \
    --max-per-cycle 12
  sleep 300
done

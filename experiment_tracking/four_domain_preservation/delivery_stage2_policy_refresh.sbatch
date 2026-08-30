#!/bin/bash
#SBATCH --job-name=P4_DELIVERY_S2_POLICY_REFRESH
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-27/anchor_domain_finalizers/delivery_policy_controller_%j.log

set -euo pipefail

CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
CAMPAIGN=/home/hersco/training_new_domains/2026-08-27/anchor_domain_finalizers
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python

"$PYTHON" \
  "$CHECKOUT/experiment_tracking/materialize_confirmation_policy.py" \
  "$CAMPAIGN/delivery_stage2_confirmation_submissions.tsv" \
  "$CAMPAIGN/delivery_stage2_policy_ready.csv"

"$PYTHON" \
  "$CHECKOUT/experiment_tracking/submit_confirmation_stage2_policy.py" \
  --manifest "$CAMPAIGN/delivery_stage2_policy_ready.csv" \
  --ledger "$CAMPAIGN/delivery_stage2_policy_submissions.tsv" \
  --queue-cap 1999 \
  --max-per-cycle 100

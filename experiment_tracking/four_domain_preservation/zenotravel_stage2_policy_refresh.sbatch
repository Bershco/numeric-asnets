#!/bin/bash
#SBATCH --job-name=P4_ZENO_S2_POLICY_REFRESH
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-28/preserve4_zenotravel_stage2_policy_eval/controller_%j.log

set -euo pipefail
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
CAMPAIGN="$CHECKOUT/experiment_tracking/four_domain_preservation"

"$PYTHON" "$CHECKOUT/experiment_tracking/materialize_confirmation_policy.py" \
  "$CAMPAIGN/zenotravel_stage2_lineages.tsv" \
  "$CAMPAIGN/zenotravel_stage2_missing_policy_ready.csv"
"$PYTHON" "$CHECKOUT/experiment_tracking/submit_confirmation_stage2_policy.py" \
  --manifest "$CAMPAIGN/zenotravel_stage2_missing_policy_ready.csv" \
  --ledger "$CAMPAIGN/zenotravel_stage2_missing_policy_submissions.tsv" \
  --queue-cap 1999 --max-per-cycle 100

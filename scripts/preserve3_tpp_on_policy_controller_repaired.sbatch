#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --job-name=P3_TPP_ON_POLICY_REPAIRED
#SBATCH --output=/home/hersco/training_new_domains/2026-09-04/stage2_policy_controllers/tpp_on_repaired_%j.log

set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
TRACKING="$REPO/experiment_tracking"
OUT=/home/hersco/training_new_domains/2026-09-04/stage2_policy_controllers
mkdir -p "$OUT"

python "$TRACKING/materialize_stage2_policy_from_ledger.py" \
  "$TRACKING/four_domain_preservation/terminal_stage2_submissions.tsv" \
  "$OUT/tpp_on_policy_ready.csv" \
  --domain tpp \
  --value-head on \
  --continuation-overrides "$TRACKING/four_domain_preservation/terminal_stage2_tpp_on_terminal_override_20260904.csv" \
  --role-prefix preserve3_terminal_tpp_on

python "$REPO/experiment_tracking/submit_confirmation_stage2_policy.py" \
  --manifest "$OUT/tpp_on_policy_ready.csv" \
  --ledger "$OUT/tpp_on_policy_submissions.tsv" \
  --queue-cap 1999 \
  --max-per-cycle 250

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --job-name=P3_TPP_OFF_POLICY_REPAIRED
#SBATCH --output=/home/hersco/training_new_domains/2026-09-03/stage2_policy_controllers/tpp_off_repaired_%j.log

set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
TRACKING="$REPO/experiment_tracking"
OUT=/home/hersco/training_new_domains/2026-09-03/stage2_policy_controllers
mkdir -p "$OUT"

python "$TRACKING/materialize_stage2_policy_from_ledger.py" \
  "$TRACKING/four_domain_preservation/terminal_stage2_submissions.tsv" \
  "$OUT/tpp_off_policy_ready.csv" \
  --domain tpp \
  --branch terminal_led \
  --value-head off \
  --continuation-overrides "$TRACKING/four_domain_preservation/terminal_stage2_tpp_off_continuation_overrides.csv" \
  --role-prefix preserve3_terminal_tpp_off

python "$REPO/scripts/statistical_replication_manifest_controller.py" \
  --policy-manifest "$OUT/tpp_off_policy_ready.csv" \
  --ledger "$OUT/tpp_off_policy_submissions.tsv" \
  --queue-cap 1999 \
  --max-per-cycle 250 \
  --poll-seconds 60

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=20:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ENDO_KL_WATCH
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/controller_%j.txt

set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"; : "${CAMPAIGN_ROOT:?}"; : "${SCRIPT_ROOT:?}"
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
while true; do
  python3 "$SCRIPT_ROOT/endogenous_kl_incremental_controller_20260918.py" \
    --campaign "$CAMPAIGN_ROOT" \
    --policy-sbatch "$SCRIPT_ROOT/endogenous_kl_100epoch_policy_curve_20260918.sbatch" \
    --mcts-sbatch "$SCRIPT_ROOT/endogenous_kl_selected_mcts_20260918.sbatch" \
    --controller-sbatch "$SCRIPT_ROOT/endogenous_kl_incremental_controller_20260918.sbatch" \
    --checkout "$CHECKOUT" --code-commit "$CODE_COMMIT" --submit
  if test "${ONE_CYCLE:-0}" = 1; then
    exit 0
  fi
  if test -s "$CAMPAIGN_ROOT/selected_mcts_submissions.tsv"; then
    echo "[ENDO KL WATCH] selected MCTS submitted; controller complete"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-15
#SBATCH --job-name=ENDO_KL_MCTS
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/mcts_%A_%a.txt

set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"; : "${CAMPAIGN_ROOT:?}"; : "${MCTS_MANIFEST:?}"
python3 "$CHECKOUT/scripts/endogenous_kl_selected_mcts_20260918.py" \
  --manifest "$MCTS_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --checkout "$CHECKOUT" --code-commit "$CODE_COMMIT" --campaign "$CAMPAIGN_ROOT"

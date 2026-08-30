#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --job-name=ctrl_main_val_s2_drone_mcts
#SBATCH --output=/home/hersco/training_new_domains/2026-08-30/main_val_stage2_drone_mcts/controller_%j.log

set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
TRACKING="$REPO/experiment_tracking"
OUT=/home/hersco/training_new_domains/2026-08-30/main_val_stage2_drone_mcts
mkdir -p "$OUT"

python "$REPO/scripts/statistical_replication_manifest_controller.py" \
  --mcts-manifest "$TRACKING/main_val_stage2_drone_mcts_manifest.csv" \
  --ledger "$OUT/submissions.tsv" \
  --queue-cap 1999 \
  --max-per-cycle 16 \
  --poll-seconds 60

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --job-name=ctrl_stage2_mcts_branch_completion
#SBATCH --output=/home/hersco/training_new_domains/2026-09-03/stage2_mcts_branch_completion/controller_%j.log

set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
OUT=/home/hersco/training_new_domains/2026-09-03/stage2_mcts_branch_completion
mkdir -p "$OUT"

python "$REPO/scripts/statistical_replication_manifest_controller.py" \
  --mcts-manifest "$OUT/stage2_mcts_branch_completion_approved_20260903.csv" \
  --ledger "$OUT/submissions.tsv" \
  --queue-cap 1999 \
  --max-per-cycle 18 \
  --poll-seconds 60

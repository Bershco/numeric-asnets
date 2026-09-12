#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_ANCHOR_PBA_FINAL
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a/finalize_%j.log

set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
work=/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a
source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate
python -u "$repo/scripts/finalize_mprime_anchor_phase_b_a_20260912.py" \
  --manifest "$repo/experiment_tracking/mprime_anchor_phase_b_a_20260912/manifest.csv" \
  --rescore-root "$work/rescore" \
  --evidence-output "$work/anchor_evidence.csv" \
  --frozen-output "$work/proposed_anchor_freeze.csv"

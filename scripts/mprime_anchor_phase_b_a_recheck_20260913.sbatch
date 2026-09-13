#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_PBA_RECHECK
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a/recheck_%j.log

set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate
python -u "$repo/scripts/continue_mprime_anchor_phase_b_a_20260913.py" \
  --attempt "${MPRIME_RECHECK_ATTEMPT:-1}" --max-attempts 4

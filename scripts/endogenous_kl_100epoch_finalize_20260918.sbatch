#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/finalize_%j.txt
set -euo pipefail
: "${SCRIPT_ROOT:?}"; : "${CAMPAIGN_ROOT:?}"
python3 "$SCRIPT_ROOT/finalize_endogenous_kl_100epoch_pilot_20260918.py" "$CAMPAIGN_ROOT"

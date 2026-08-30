#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_FINALIZE_STAGE2
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-31/mprime_finalize_stage2_%j.log

set -euo pipefail
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
"$PYTHON" "$CHECKOUT/experiment_tracking/mprime_validation_ipc_scale_v1/finalize_and_submit_stage2.py"

#!/bin/bash
#SBATCH --job-name=MPRIME_ANCHOR_POLICY_REFRESH
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-28/mprime_corrected_downstream/policy_controller_%j.log

set -euo pipefail
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
"$PYTHON" "$CHECKOUT/experiment_tracking/mprime_validation_ipc_scale_v1/materialize_anchor_policy.py"
"$PYTHON" "$CHECKOUT/experiment_tracking/mprime_validation_ipc_scale_v1/submit_anchor_policy.py"

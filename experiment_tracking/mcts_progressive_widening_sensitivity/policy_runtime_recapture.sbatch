#!/bin/bash
#SBATCH --job-name=PW_KMIN3_POLICY_RUNTIME_CTRL
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-30/mcts_pw_kmin3_policy_runtime/controller_%j.log

set -euo pipefail
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
mkdir -p /home/hersco/training_new_domains/2026-08-30/mcts_pw_kmin3_policy_runtime
"$PYTHON" "$CHECKOUT/scripts/submit_pw_policy_runtime_recapture.py"

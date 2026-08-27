#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/hersco/training_new_domains/2026-08-27/mcts_safe_drone_diagnostic/preflight_%j.out

set -euo pipefail

SAFE_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
mkdir -p /home/hersco/training_new_domains/2026-08-27/mcts_safe_drone_diagnostic

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    cd '${SAFE_REPO}/asnets'
    export PYTHONPATH='${SAFE_REPO}/asnets:${SAFE_REPO}'
    '${PRODUCTION_REPO}/venv-asnets/bin/python' \
      -m unittest discover -s tests -p test_progressive_widening.py
  "

echo "[MCTS-SAFE PREFLIGHT] commit=efb003cc tests=passed"

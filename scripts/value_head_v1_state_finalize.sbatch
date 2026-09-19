#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/finalize_%j.txt

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
apptainer exec --bind /home/hersco:/home/hersco /home/hersco/Docker/image.sif /bin/bash -lc "
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  python '$REPO/scripts/finalize_value_head_v1_state_manifests.py' \
    --capture-root '$ROOT/captures' --output-dir '$ROOT/ready'
"

#!/usr/bin/env bash
#SBATCH --job-name=YARIN_EXT_POLICY
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --output=/home/hersco/training_new_domains/2026-09-10/yarin_external/policy_%A_%a.log

set -euo pipefail
FEATURE_REPO=${FEATURE_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-yarin-external}
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-10/yarin_external
CONTAINER=/home/hersco/Docker/image.sif
apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$FEATURE_REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$FEATURE_REPO':\${PYTHONPATH:-}
    export YARIN_EXTERNAL_DATA_ROOT='$ROOT/frozen'
    python '$FEATURE_REPO/scripts/run_yarin_external_policy.py' \
      --manifest '$FEATURE_REPO/experiment_tracking/advisor_followup_20260910/yarin_external_screen_manifest.csv' \
      --index '$SLURM_ARRAY_TASK_ID' --completion-root '$ROOT/completion'
  "

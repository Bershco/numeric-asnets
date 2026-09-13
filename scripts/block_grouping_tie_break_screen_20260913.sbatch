#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=1-06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --array=0-1
#SBATCH --nice=10000
#SBATCH --job-name=BG_TIE_SCREEN
#SBATCH --output=/home/hersco/training_new_domains/2026-09-13/block_grouping_tie_break_screen/%A_%a.sbatch.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
repo=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break}
production=/home/hersco/bershco-nu-asnets/numeric-asnets
manifest="$repo/experiment_tracking/block_grouping_tie_break_screen_20260913/manifest.csv"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$CODE_COMMIT" ]]
mkdir -p /home/hersco/training_new_domains/2026-09-13/block_grouping_tie_break_screen
apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    python -u '$repo/scripts/run_block_grouping_tie_break_screen_20260913.py' \
      --repo '$repo' --manifest '$manifest' --task '$SLURM_ARRAY_TASK_ID'
  "

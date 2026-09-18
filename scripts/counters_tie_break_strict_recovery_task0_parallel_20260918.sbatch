#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --array=0
#SBATCH --nice=10000
#SBATCH --no-requeue
#SBATCH --job-name=COUNTERS_TIE_LAST2
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-25,ise-cpu-intl-26,ise-cpu-intl-27,ise-cpu-intl-28
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/counters_tie_break_strict_task0_last2/%A_%a.sbatch.out

set -euo pipefail
repo=${COUNTERS_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break}
root=/home/hersco/training_new_domains/2026-09-18/counters_tie_break_strict_task0_last2
production=/home/hersco/bershco-nu-asnets/numeric-asnets
manifest="$repo/experiment_tracking/counters_tie_break_strict_stage1_20260913/manifest.csv"
runner="$repo/scripts/run_counters_tie_break_strict_recovery_20260915.py"
mkdir -p "$root"

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    export COUNTERS_RECOVERY_WORKERS=2
    python -u '$runner' --repo '$repo' --manifest '$manifest' --task 0
  "

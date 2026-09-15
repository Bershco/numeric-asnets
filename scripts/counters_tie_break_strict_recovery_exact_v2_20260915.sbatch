#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --array=7,19
#SBATCH --nice=10000
#SBATCH --job-name=COUNTERS_TIE_EXACT2
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-26,ise-cpu-intl-27,ise-cpu-intl-28
#SBATCH --output=/home/hersco/training_new_domains/2026-09-15/counters_tie_break_strict_recovery_v2/%A_%a.sbatch.out

set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
root=/home/hersco/training_new_domains/2026-09-15/counters_tie_break_strict_recovery_v2
production=/home/hersco/bershco-nu-asnets/numeric-asnets
manifest="$repo/experiment_tracking/counters_tie_break_strict_stage1_20260913/manifest.csv"
runner="$root/run_counters_tie_break_strict_recovery_20260915.py"
mkdir -p "$root"

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    python -u '$runner' --repo '$repo' --manifest '$manifest' --task '$SLURM_ARRAY_TASK_ID'
  "

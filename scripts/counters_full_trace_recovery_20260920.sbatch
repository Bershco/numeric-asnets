#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --no-requeue
#SBATCH --job-name=COUNTERS_TRACE_REC
#SBATCH --exclude=ise-cpu-intl-[01,05,15,18,24,26]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-20/counters_tie_break_full_trace_recovery/slurm/%A_%a.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${TRACE_MANIFEST:?TRACE_MANIFEST is required}"
repo=${COUNTERS_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break}
production=/home/hersco/bershco-nu-asnets/numeric-asnets
actual_commit=$(git -C "$repo" rev-parse HEAD)
[[ "$actual_commit" == "$CODE_COMMIT" ]]

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    python -u '$repo/scripts/run_counters_tie_break_full_trace_20260915.py' \
      --repo '$repo' --manifest '$TRACE_MANIFEST' --task '$SLURM_ARRAY_TASK_ID'
  "

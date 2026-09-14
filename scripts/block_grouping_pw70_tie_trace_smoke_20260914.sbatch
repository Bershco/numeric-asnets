#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --nice=10000
#SBATCH --exclude=ise-cpu-intl-[01,09-11,13,27]
#SBATCH --job-name=BG_PW70_TRACE_SMOKE
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/block_grouping_pw70_tie_trace/%j.smoke.out
set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
repo=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break}
production=/home/hersco/bershco-nu-asnets/numeric-asnets
[[ "$(git -C "$repo" rev-parse HEAD)" == "$CODE_COMMIT" ]]
mkdir -p /home/hersco/training_new_domains/2026-09-14/block_grouping_pw70_tie_trace
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "source '$production/venv-asnets/bin/activate'; export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}; export ENHSP_CONFIG_OVERRIDE=hmrmax-astar; python -u '$repo/scripts/run_block_grouping_pw70_tie_trace_20260914.py' --repo '$repo' --manifest '$repo/experiment_tracking/block_grouping_pw70_tie_trace_20260914/manifest.csv' --task 0 --preflight"

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --nice=10000
#SBATCH --no-requeue
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-26,ise-cpu-intl-27,ise-cpu-intl-28
#SBATCH --job-name=MCTS_DIV_ROOT_SMOKE
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/mcts_policy_divergence_stage1/compute_smoke_%j.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${CHECKOUT:?CHECKOUT must name the isolated deployed checkout}"
production=${PRODUCTION_CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets}
output_root=${STAGE1_OUTPUT_ROOT:-/home/hersco/training_new_domains/2026-09-18/mcts_policy_divergence_stage1}
registry="$CHECKOUT/experiment_tracking/experiment_results.csv"

[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$CODE_COMMIT" ]]
mkdir -p "$output_root/compute_smoke"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    python -u '$CHECKOUT/scripts/run_mcts_policy_divergence_stage1_20260918.py' \
      compute-smoke \
      --repo '$CHECKOUT' \
      --registry '$registry' \
      --output-root '$output_root' \
      --code-commit '$CODE_COMMIT' \
      --execute
  "

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=1-02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --nice=10000
#SBATCH --no-requeue
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-26,ise-cpu-intl-27,ise-cpu-intl-28
#SBATCH --job-name=MCTS_DIV_RECOVER
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/mcts_policy_divergence_stage1_recovery/slurm_%A_%a.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${CHECKOUT:?CHECKOUT must name the isolated deployed checkout}"
: "${SMOKE_GATE_JSON:?SMOKE_GATE_JSON is required}"
: "${RECOVERY_MANIFEST:?RECOVERY_MANIFEST is required}"
: "${RECOVERY_FREEZE:?RECOVERY_FREEZE is required}"
: "${RECOVERY_OUTPUT_ROOT:?RECOVERY_OUTPUT_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?submit this payload as a recovery array}"
production=${PRODUCTION_CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets}
tracking="$CHECKOUT/experiment_tracking/mcts_policy_divergence_cause_audit"
manifest="$tracking/stage1_grouped_tasks.csv"
freeze="$tracking/stage1_grouped_tasks.freeze.json"
registry="$CHECKOUT/experiment_tracking/experiment_results.csv"

[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$CODE_COMMIT" ]]
[[ -f "$SMOKE_GATE_JSON" ]]
[[ -f "$RECOVERY_MANIFEST" ]]
[[ -f "$RECOVERY_FREEZE" ]]
mkdir -p "$RECOVERY_OUTPUT_ROOT"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    python -u '$CHECKOUT/scripts/run_mcts_policy_divergence_stage1_20260918.py' \
      run-recovery \
      --repo '$CHECKOUT' \
      --manifest '$manifest' \
      --freeze '$freeze' \
      --registry '$registry' \
      --recovery-manifest '$RECOVERY_MANIFEST' \
      --recovery-freeze '$RECOVERY_FREEZE' \
      --recovery-index '$SLURM_ARRAY_TASK_ID' \
      --output-root '$RECOVERY_OUTPUT_ROOT' \
      --smoke-gate '$SMOKE_GATE_JSON' \
      --code-commit '$CODE_COMMIT' \
      --execute
  "

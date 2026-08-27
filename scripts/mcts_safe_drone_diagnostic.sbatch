#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]

set -euo pipefail
set -o pipefail

: "${CHECKPOINT_B64:?CHECKPOINT_B64 is required}"
: "${SEED:?SEED is required}"
: "${VALUE_HEAD:?VALUE_HEAD must be on or off}"
: "${SKIP_INSTANCE_NUMBERS:?SKIP_INSTANCE_NUMBERS is required}"
: "${RUN_LABEL:?RUN_LABEL is required}"

CHECKPOINT=$(printf '%s' "${CHECKPOINT_B64}" | base64 -d)

SAFE_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
OUTPUT_DIR=/home/hersco/training_new_domains/2026-08-27/mcts_safe_drone_diagnostic
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate

mkdir -p "${OUTPUT_DIR}/completion"
LOG="${OUTPUT_DIR}/${SLURM_JOB_ID}_${RUN_LABEL}.txt"
COMPLETION_FILE="${OUTPUT_DIR}/completion/${SLURM_JOB_ID}.jsonl"

VALUE_HEAD_ARG=""
if [[ "${VALUE_HEAD}" == "off" ]]; then
  VALUE_HEAD_ARG="--disable-value-head"
elif [[ "${VALUE_HEAD}" != "on" ]]; then
  echo "VALUE_HEAD must be on or off" >&2
  exit 2
fi

echo "[MCTS-SAFE] commit=c6bed5437134c818e088cea9f7b81eb157b87563"
echo "[MCTS-SAFE] checkpoint=${CHECKPOINT}"
echo "[MCTS-SAFE] seed=${SEED} value_head=${VALUE_HEAD}"
echo "[MCTS-SAFE] skip=${SKIP_INSTANCE_NUMBERS}"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "${CONTAINER}" /bin/bash -lc "
    set -o pipefail
    cd '${SAFE_REPO}/asnets'
    source '${PRODUCTION_REPO}/venv-asnets/bin/activate'
    export PYTHONPATH='${SAFE_REPO}':\${PYTHONPATH:-}
    ./run_experiment \
      experiments_numeric.architecture_2.drone_mcts \
      experiments_numeric.domain.drone \
      --resume-from '${CHECKPOINT}' \
      --eval-with-mcts \
      --eval-mcts-terminal-safe-action-selection \
      --eval-scheduling rolling \
      --eval-completion-file '${COMPLETION_FILE}' \
      --eval-instance-timeout 21600 \
      --skip-instance-numbers '${SKIP_INSTANCE_NUMBERS}' \
      --num-workers 1 \
      --jpddl-max-heap 20g \
      --worker-logs \
      --mcts-expansion-size 20 \
      --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 \
      --use-estimator 0.5 \
      ${VALUE_HEAD_ARG} \
      --random-seed '${SEED}'
  " 2>&1 | tee "${LOG}"

python "${SAFE_REPO}/asnets/tools/validate_eval_log.py" \
  --log "${LOG}" \
  --domain drone \
  --validator "${VALIDATOR}"

echo "[MCTS-SAFE] completed_and_val_checked=1"

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

set -euo pipefail
set -o pipefail

: "${CHECKPOINT_B64:?CHECKPOINT_B64 is required}"
: "${NODE_FAMILY:?NODE_FAMILY is required}"
: "${REPEAT:?REPEAT is required}"

CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
OVERLAY=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_audit/overlay
OUT=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_followup
CONTAINER=/home/hersco/Docker/image.sif

mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${NODE_FAMILY}_r${REPEAT}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"

export TF_DETERMINISTIC_OPS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=1239739722

echo "[MCTS-DETERMINISM-FOLLOWUP] node_family=$NODE_FAMILY repeat=$REPEAT host=$(hostname)"
echo "[MCTS-DETERMINISM-FOLLOWUP] checkpoint=$CHECKPOINT"
echo "[MCTS-DETERMINISM-FOLLOWUP] instance=problem_3_3_4 index=8 seed=1239739722 value_head=on"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --bind "$OVERLAY/spawn_train_worker.py:$CHECKOUT/asnets/asnets/spawn_train_worker.py" \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT/asnets'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    ./run_experiment \
      experiments_numeric.architecture_2.drone_mcts \
      experiments_numeric.domain.drone \
      --resume-from '$CHECKPOINT' \
      --eval-with-mcts \
      --eval-mcts-terminal-safe-action-selection \
      --eval-scheduling rolling \
      --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 7200 \
      --skip-instance-numbers '1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20' \
      --num-workers 1 \
      --jpddl-max-heap 10g \
      --worker-logs \
      --action-debug \
      --mcts-expansion-size 20 \
      --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 \
      --use-estimator 0.5 \
      --eval-max-actions 750 \
      --random-seed 1239739722
  " 2>&1 | tee "$LOG"

grep -q '^\[MCTS DETERMINISM\] ' "$LOG"
echo "[MCTS-DETERMINISM-FOLLOWUP] record_count=$(grep -c '^\[MCTS DETERMINISM\] ' "$LOG")"

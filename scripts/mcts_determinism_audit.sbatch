#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]

set -euo pipefail
set -o pipefail

: "${CHECKPOINT_B64:?CHECKPOINT_B64 is required}"
: "${RUN_ARM:?RUN_ARM must be ordinary or deterministic_cpu}"
: "${REPEAT:?REPEAT is required}"

CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
OVERLAY=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_audit/overlay
OUT=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_audit
CONTAINER=/home/hersco/Docker/image.sif

mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${RUN_ARM}_r${REPEAT}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"

if [[ "$RUN_ARM" == deterministic_cpu ]]; then
  export TF_DETERMINISTIC_OPS=1
  export TF_NUM_INTRAOP_THREADS=1
  export TF_NUM_INTEROP_THREADS=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export PYTHONHASHSEED=534933607
elif [[ "$RUN_ARM" != ordinary ]]; then
  echo "Unknown RUN_ARM=$RUN_ARM" >&2
  exit 2
fi

echo "[MCTS-DETERMINISM] arm=$RUN_ARM repeat=$REPEAT"
echo "[MCTS-DETERMINISM] checkpoint=$CHECKPOINT"
echo "[MCTS-DETERMINISM] instance=problem_5_2_2 index=11 seed=534933607"

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
      --skip-instance-numbers '1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20' \
      --num-workers 1 \
      --jpddl-max-heap 10g \
      --worker-logs \
      --action-debug \
      --mcts-expansion-size 20 \
      --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 \
      --use-estimator 0.5 \
      --disable-value-head \
      --random-seed 534933607
  " 2>&1 | tee "$LOG"

grep -q '^\[MCTS DETERMINISM\] ' "$LOG"
echo "[MCTS-DETERMINISM] record_count=$(grep -c '^\[MCTS DETERMINISM\] ' "$LOG")"


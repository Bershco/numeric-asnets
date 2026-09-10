#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --job-name=FO_VAL_S2_RECOVERY
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-26,ise-cpu-intl-28
#SBATCH --output=/home/hersco/training_new_domains/2026-09-10/fo_stage2_validation_recovery/%j.sbatch.out

set -euo pipefail
set -o pipefail

: "${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}"
: "${VALUE_HEAD:?VALUE_HEAD is required}"
: "${SEED:?SEED is required}"
: "${CHECKPOINT_B64:?CHECKPOINT_B64 is required}"
: "${SOURCE_COMPLETION:?SOURCE_COMPLETION is required}"
: "${SKIP_NUMBERS:?SKIP_NUMBERS is required}"

[[ "$VALUE_HEAD" == off || "$VALUE_HEAD" == on ]]
CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
OUT=/home/hersco/training_new_domains/2026-09-10/fo_stage2_validation_recovery

mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_source${SOURCE_JOB_ID}_${VALUE_HEAD}_${SEED}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"
cp "$SOURCE_COMPLETION" "$COMPLETION"
VH_ARG=''
[[ "$VALUE_HEAD" == off ]] && VH_ARG='--disable-value-head'

echo "[FO S2 RECOVERY] source_job=$SOURCE_JOB_ID value_head=$VALUE_HEAD seed=$SEED"
echo "[FO S2 RECOVERY] skip_completed=$SKIP_NUMBERS width=20 simulations=70 workers=3"
echo "[FO S2 RECOVERY] checkpoint=$CHECKPOINT"

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT/asnets'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    ./run_experiment experiments_numeric.architecture_2.fo_counters_mcts \
      experiments_numeric.domain.fo_counters --resume-from '$CHECKPOINT' \
      --eval-with-mcts \
      --mcts-expansion-size 20 --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --eval-max-actions 10000 \
      --skip-instance-numbers '$SKIP_NUMBERS' --num-workers 3 \
      --jpddl-max-heap 4g --worker-logs \
      $VH_ARG --random-seed '$SEED'
  " 2>&1 | tee "$LOG"

python "$CHECKOUT/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$LOG" --domain fo_counters --validator "$VALIDATOR" \
  --summary-csv "$OUT/${SLURM_JOB_ID}_source${SOURCE_JOB_ID}.val.csv"
echo "[FO S2 RECOVERY] completed=1 source_job=$SOURCE_JOB_ID"

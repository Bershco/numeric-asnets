#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --array=0-5
#SBATCH --nice=10000
#SBATCH --job-name=COUNTERS_TIE_3WAY
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-26,ise-cpu-intl-27,ise-cpu-intl-28
#SBATCH --output=/home/hersco/training_new_domains/2026-09-11/counters_tie_break_3way/%A_%a.sbatch.out

set -euo pipefail
set -o pipefail

: "${CHECKPOINT_OFF_B64:?CHECKPOINT_OFF_B64 is required}"
: "${CHECKPOINT_ON_B64:?CHECKPOINT_ON_B64 is required}"
: "${CODE_COMMIT:?CODE_COMMIT is required}"

case "$SLURM_ARRAY_TASK_ID" in
  0) VALUE_HEAD=off; TIE_BREAK=action_id ;;
  1) VALUE_HEAD=on;  TIE_BREAK=action_id ;;
  2) VALUE_HEAD=off; TIE_BREAK=q ;;
  3) VALUE_HEAD=on;  TIE_BREAK=q ;;
  4) VALUE_HEAD=off; TIE_BREAK=policy ;;
  5) VALUE_HEAD=on;  TIE_BREAK=policy ;;
  *) echo "invalid array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

if [[ "$VALUE_HEAD" == off ]]; then
  CHECKPOINT=$(printf '%s' "$CHECKPOINT_OFF_B64" | base64 -d)
  VH_ARG=--disable-value-head
else
  CHECKPOINT=$(printf '%s' "$CHECKPOINT_ON_B64" | base64 -d)
  VH_ARG=''
fi

CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
OUT=/home/hersco/training_new_domains/2026-09-11/counters_tie_break_3way
SKIP='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,55,56,57,59'

mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${VALUE_HEAD}_${TIE_BREAK}.txt"
COMPLETION="$OUT/completion/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

ACTUAL_COMMIT=$(git -C "$CHECKOUT" rev-parse HEAD)
[[ "$ACTUAL_COMMIT" == "$CODE_COMMIT" ]]

echo "[COUNTERS TIE 3WAY] value_head=$VALUE_HEAD tie_break=$TIE_BREAK seed=1963100312"
echo "[COUNTERS TIE 3WAY] instances=fz_instance_51,fz_instance_55,fz_instance_59 slots=50,54,58"
echo "[COUNTERS TIE 3WAY] width=5 simulations=20 puct=0.1 estimator=0.5 worker=1"
echo "[COUNTERS TIE 3WAY] checkpoint=$CHECKPOINT commit=$ACTUAL_COMMIT"

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT/asnets'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE=hmrmax-astar
    ./run_experiment experiments_numeric.architecture_2.counters_mcts \
      experiments_numeric.domain.counters --resume-from '$CHECKPOINT' \
      --eval-with-mcts --eval-mcts-root-visit-tie-break '$TIE_BREAK' \
      --mcts-expansion-size 5 --mcts-iterations 20 \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --eval-max-actions 10000 \
      --skip-instance-numbers '$SKIP' --num-workers 1 \
      --jpddl-max-heap 4g --worker-logs --action-debug \
      $VH_ARG --random-seed 1963100312
  " 2>&1 | tee "$LOG"

grep -q "root_visit_tie_break = $TIE_BREAK" "$LOG"
grep -q '^\[MCTS DETERMINISM\] ' "$LOG"
python "$CHECKOUT/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$LOG" --domain counters --validator "$VALIDATOR" \
  --summary-csv "$OUT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${VALUE_HEAD}_${TIE_BREAK}.val.csv"
python "$CHECKOUT/scripts/summarize_mcts_visit_distribution.py" \
  "$LOG" \
  --output-prefix "$OUT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${VALUE_HEAD}_${TIE_BREAK}"

echo "[COUNTERS TIE 3WAY] completed=1 trace_records=$(grep -c '^\[MCTS DETERMINISM\] ' "$LOG")"

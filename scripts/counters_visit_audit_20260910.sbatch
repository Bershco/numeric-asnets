#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --job-name=COUNTERS_VISIT_AUDIT
#SBATCH --output=/home/hersco/training_new_domains/2026-09-10/counters_visit_audit/%j.sbatch.out

set -euo pipefail
set -o pipefail

: "${ARM:?ARM is required}"
: "${VALUE_HEAD:?VALUE_HEAD is required}"
: "${CHECKPOINT_B64:?CHECKPOINT_B64 is required}"

[[ "$ARM" == failure || "$ARM" == parallel_control ]]
[[ "$VALUE_HEAD" == off || "$VALUE_HEAD" == on ]]

CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-yarin-external
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
OUT=/home/hersco/training_new_domains/2026-09-10/counters_visit_audit
SKIP='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,55,56,57,59'

mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${ARM}_${VALUE_HEAD}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"
VH_ARG=''
[[ "$VALUE_HEAD" == off ]] && VH_ARG='--disable-value-head'

echo "[COUNTERS VISIT AUDIT] arm=$ARM value_head=$VALUE_HEAD seed=1963100312"
echo "[COUNTERS VISIT AUDIT] instances=fz_instance_51,fz_instance_55,fz_instance_59 slots=50,54,58"
echo "[COUNTERS VISIT AUDIT] width=5 simulations=20 puct=0.1 estimator=0.5 worker=1"
echo "[COUNTERS VISIT AUDIT] checkpoint=$CHECKPOINT"

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
      --eval-with-mcts \
      --mcts-expansion-size 5 --mcts-iterations 20 \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --eval-max-actions 10000 \
      --skip-instance-numbers '$SKIP' --num-workers 1 \
      --jpddl-max-heap 4g --worker-logs --action-debug \
      $VH_ARG --random-seed 1963100312
  " 2>&1 | tee "$LOG"

grep -q '^\[MCTS DETERMINISM\] ' "$LOG"
python "$CHECKOUT/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$LOG" --domain counters --validator "$VALIDATOR" \
  --summary-csv "$OUT/${SLURM_JOB_ID}_${ARM}_${VALUE_HEAD}.val.csv"
python "$CHECKOUT/scripts/summarize_mcts_visit_distribution.py" \
  "$LOG" --output-prefix "$OUT/${SLURM_JOB_ID}_${ARM}_${VALUE_HEAD}"

echo "[COUNTERS VISIT AUDIT] completed=1 trace_records=$(grep -c '^\[MCTS DETERMINISM\] ' "$LOG")"

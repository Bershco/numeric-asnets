#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --job-name=FO_VAL_S2_EXACT9
#SBATCH --exclude=ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-08,ise-cpu-intl-09,ise-cpu-intl-10,ise-cpu-intl-13,ise-cpu-intl-14,ise-cpu-intl-26,ise-cpu-intl-27,ise-cpu-intl-28
#SBATCH --output=/home/hersco/training_new_domains/2026-09-12/fo_stage2_validation_exact9/%j.sbatch.out

set -euo pipefail
set -o pipefail

SOURCE_JOB_ID=20943885
VALUE_HEAD=off
SEED=2082152039
CHECKPOINT='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.fo_counters-experiments_numeric.architecture_2.fo_counters_mcts-2026-08-21T11:21:49.188976/P[domain,instance_2,instance_3,instance_4]-S[0.0003,50,enhsp-hmrmax-astar]-MO[]-T[518400]-04906475-faf7d6/snapshots/snapshot_17_1.0000'
SKIP_NUMBERS='1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20'
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-yarin-external
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
OUT=/home/hersco/training_new_domains/2026-09-12/fo_stage2_validation_exact9

mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_source${SOURCE_JOB_ID}_${VALUE_HEAD}_${SEED}_instance9.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"

echo "[FO S2 EXACT RECOVERY] source_job=$SOURCE_JOB_ID value_head=$VALUE_HEAD seed=$SEED"
echo "[FO S2 EXACT RECOVERY] evaluator_instance_number=9 width=20 simulations=70 workers=1"
echo "[FO S2 EXACT RECOVERY] checkpoint=$CHECKPOINT"

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
      --eval-with-mcts --mcts-expansion-size 20 --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --eval-max-actions 10000 \
      --skip-instance-numbers '$SKIP_NUMBERS' --num-workers 1 \
      --jpddl-max-heap 4g --worker-logs --disable-value-head \
      --random-seed '$SEED'
  " 2>&1 | tee "$LOG"

python "$CHECKOUT/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$LOG" --domain fo_counters --validator "$VALIDATOR" \
  --summary-csv "$OUT/${SLURM_JOB_ID}_source${SOURCE_JOB_ID}_instance9.val.csv"
echo "[FO S2 EXACT RECOVERY] completed=1 source_job=$SOURCE_JOB_ID evaluator_instance_number=9"

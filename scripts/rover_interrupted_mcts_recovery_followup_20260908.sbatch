#!/usr/bin/env bash
#SBATCH --job-name=ROVER_INTERRUPTION_RECOVERY_FOLLOWUP
#SBATCH --partition=main
#SBATCH --time=13:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-08/rover_interrupted_mcts_recovery_followup/%A_%a.sbatch.out

set -euo pipefail
set -o pipefail

MANIFEST=/home/hersco/training_new_domains/2026-09-08/rover_interrupted_mcts_recovery_followup/manifest.tsv
ROW=$(awk -F '\t' -v target="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == target {print; exit}' "$MANIFEST")
[[ -n "$ROW" ]] || { echo "missing manifest row for array index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }

IFS=$'\t' read -r ARRAY_INDEX RECOVERY_ID SOURCE_JOB_ID EXPERIMENT_SCOPE VALUE_HEAD SEED \
  ARCHITECTURE MISSING_NUMBERS SKIP_NUMBERS MISSING_COUNT CHECKPOINT CHECKPOINT_B64 \
  WIDTH ITERATIONS PUCT ESTIMATOR WORKERS REQUESTED_CPUS REQUESTED_MEMORY TIME_LIMIT \
  INSTANCE_TIMEOUT SOURCE_LOG SOURCE_COMPLETION SOURCE_INVOCATION_COUNT <<< "$ROW"

[[ "$WIDTH" == 20 && ( "$ITERATIONS" == 70 || "$ITERATIONS" == 200 ) && "$PUCT" == 0.1 && "$ESTIMATOR" == 0.5 ]]
[[ "$WORKERS" == 1 && "$REQUESTED_CPUS" == 2 && "$REQUESTED_MEMORY" == 160G ]]
[[ "$MISSING_COUNT" -ge 1 && "$MISSING_COUNT" -le 4 ]]
CHECKPOINT_DECODED=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
[[ "$CHECKPOINT_DECODED" == "$CHECKPOINT" ]]

PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
OUT=/home/hersco/training_new_domains/2026-09-08/rover_interrupted_mcts_recovery_followup
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${RECOVERY_ID}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"

VH_ARG=''
[[ "$VALUE_HEAD" == off ]] && VH_ARG='--disable-value-head'
[[ "$VALUE_HEAD" == on || "$VALUE_HEAD" == off ]]

echo "[ROVER RECOVERY] source_job=$SOURCE_JOB_ID scope=$EXPERIMENT_SCOPE"
echo "[ROVER RECOVERY] missing=$MISSING_NUMBERS skip=$SKIP_NUMBERS"
echo "[ROVER RECOVERY] checkpoint=$CHECKPOINT"
echo "[ROVER RECOVERY] fixed_width=$WIDTH simulations=$ITERATIONS puct=$PUCT estimator=$ESTIMATOR workers=1 memory=160G"

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -o pipefail
    cd '$PRODUCTION_REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    ./run_experiment experiments_numeric.architecture_2.${ARCHITECTURE} \
      experiments_numeric.domain.rover --resume-from '$CHECKPOINT' \
      --eval-with-mcts \
      --mcts-expansion-size '$WIDTH' --mcts-iterations '$ITERATIONS' \
      --mcts-exploration-weight '$PUCT' --use-estimator '$ESTIMATOR' \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --skip-instance-numbers '$SKIP_NUMBERS' \
      --num-workers 1 --jpddl-max-heap 4g --worker-logs \
      $VH_ARG --random-seed '$SEED'
  " 2>&1 | tee "$LOG"

python "$PRODUCTION_REPO/asnets/tools/validate_eval_log.py" \
  --log "$LOG" --domain rover --validator "$VALIDATOR"

echo "[ROVER RECOVERY] completed_and_val_checked=1 missing=$MISSING_NUMBERS"

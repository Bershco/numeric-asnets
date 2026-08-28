#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]

set -euo pipefail
set -o pipefail
: "${CHECKPOINT_B64:?}" "${SEED:?}" "${VALUE_HEAD:?}" "${MANIFEST_ID:?}"
: "${PW_MIN:?}" "${PW_C:?}" "${PW_ALPHA:?}" "${ITERATIONS:?}"

CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
FEATURE_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe
CODE_OVERLAY=/home/hersco/bershco-nu-asnets/numeric-asnets-pw-sensitivity-overlay
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
OUT=/home/hersco/training_new_domains/2026-08-28/mcts_pw_safe_sensitivity
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${MANIFEST_ID}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"
vh_arg=''
[[ "$VALUE_HEAD" == off ]] && vh_arg='--disable-value-head'

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -o pipefail
    cd '$FEATURE_REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$CODE_OVERLAY':'$FEATURE_REPO':\${PYTHONPATH:-}
    ./run_experiment experiments_numeric.architecture_2.drone_mcts \
      experiments_numeric.domain.drone --resume-from '$CHECKPOINT' \
      --eval-with-mcts --eval-mcts-terminal-safe-action-selection \
      --mcts-progressive-widening --mcts-pw-min-width '$PW_MIN' \
      --mcts-pw-c '$PW_C' --mcts-pw-alpha '$PW_ALPHA' \
      --mcts-expansion-size 20 --mcts-iterations '$ITERATIONS' \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --num-workers 3 --jpddl-max-heap 4g \
      --worker-logs $vh_arg --random-seed '$SEED'
  " 2>&1 | tee "$LOG"

python "$PRODUCTION_REPO/asnets/tools/validate_eval_log.py" \
  --log "$LOG" --domain drone --validator "$VALIDATOR"

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]

set -euo pipefail
: "${CHECKPOINT_B64:?}" "${SEED:?}" "${VALUE_HEAD:?}" "${ARM:?}" "${MANIFEST_ID:?}"
CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
OUT=/home/hersco/training_new_domains/2026-08-28/mcts_safe_context_full
mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${MANIFEST_ID}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"
vh_arg=''; [[ "$VALUE_HEAD" == off ]] && vh_arg='--disable-value-head'
context_arg=''; [[ "$ARM" == contextual ]] && context_arg='--eval-mcts-contextual-nodes'

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT/asnets'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    ./run_experiment experiments_numeric.architecture_2.drone_mcts \
      experiments_numeric.domain.drone --resume-from '$CHECKPOINT' \
      --eval-with-mcts --eval-mcts-terminal-safe-action-selection \
      --eval-mcts-context-diagnostics $context_arg \
      --mcts-expansion-size 20 --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --num-workers 3 --jpddl-max-heap 4g \
      --worker-logs $vh_arg --random-seed '$SEED'
  " 2>&1 | tee "$LOG"
python "$PRODUCTION/asnets/tools/validate_eval_log.py" \
  --log "$LOG" --domain drone --validator "$VALIDATOR"

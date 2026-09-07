#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]

set -euo pipefail
set -o pipefail
: "${CHECKPOINT_B64:?}" "${ARM:?}" "${WIDTH:?}" "${ITERATIONS:?}"
: "${PROGRESSIVE_WIDENING:?}" "${MANIFEST_ID:?}"

CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
FEATURE_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
OUT=/home/hersco/training_new_domains/2026-09-07/tpp_catastrophic_seed_mcts
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_${MANIFEST_ID}.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"

pw_args=''
if [[ "$PROGRESSIVE_WIDENING" == true ]]; then
  pw_args='--mcts-progressive-widening --mcts-pw-min-width 3 --mcts-pw-c 0.6 --mcts-pw-alpha 0.5'
fi

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -o pipefail
    cd '$FEATURE_REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$FEATURE_REPO':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE='hadd-astar'
    ./run_experiment experiments_numeric.architecture_2.tpp_mcts \
      experiments_numeric.domain.tpp --resume-from '$CHECKPOINT' \
      --eval-with-mcts --eval-mcts-terminal-safe-action-selection \
      $pw_args --mcts-expansion-size '$WIDTH' --mcts-iterations '$ITERATIONS' \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --num-workers 3 --jpddl-max-heap 4g \
      --worker-logs --disable-value-head --random-seed 1972442430
  " 2>&1 | tee "$LOG"

python "$PRODUCTION_REPO/asnets/tools/validate_eval_log.py" \
  --log "$LOG" --domain tpp --validator "$VALIDATOR"

#!/usr/bin/env bash
#SBATCH --job-name=FO_PW70_ONE_INSTANCE
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-09/fo_pw70_single_instance/%j.sbatch.out

set -euo pipefail
set -o pipefail

SOURCE_JOB=21039205
SEED=2082152039
CHECKPOINT_B64='L2hvbWUvaGVyc2NvL2JlcnNoY28tbnUtYXNuZXRzL251bWVyaWMtYXNuZXRzL2FzbmV0cy9leHBlcmltZW50LXJlc3VsdHMvZXhwZXJpbWVudHNfbnVtZXJpYy5kb21haW4uZm9fY291bnRlcnMtZXhwZXJpbWVudHNfbnVtZXJpYy5hcmNoaXRlY3R1cmVfMi5mb19jb3VudGVycy0yMDI2LTA4LTIwVDE4OjEzOjU0Ljk0ODAyNC9QW2RvbWFpbixpbnN0YW5jZV8yLGluc3RhbmNlXzMsaW5zdGFuY2VfNF0tU1swLjAwMyw1MCxlbmhzcC1obXJtYXgtYXN0YXJdLU1PW10tVFs1MTg0MDBdLTA0OTA1NDQ4LWZhZjdkNi9zbmFwc2hvdHMvc25hcHNob3RfNjlfMS4wMDA='
CHECKPOINT=$(printf '%s' "$CHECKPOINT_B64" | base64 -d)
SKIP='1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20'
FEATURE_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
OUT=/home/hersco/training_new_domains/2026-09-09/fo_pw70_single_instance
CONTAINER=/home/hersco/Docker/image.sif
VALIDATOR=/home/hersco/tools/VAL/build/bin/Validate
mkdir -p "$OUT/completion"
LOG="$OUT/${SLURM_JOB_ID}_fo-off-2082152039-instance15-pw70.txt"
COMPLETION="$OUT/completion/${SLURM_JOB_ID}.jsonl"

echo "[FO PW70 RECOVERY] source_job=$SOURCE_JOB target_position=15 target=instance_15.pddl"
echo "[FO PW70 RECOVERY] seed=$SEED width=20 simulations=70 kmin=3 c=0.6 alpha=0.5 workers=1"
echo "[FO PW70 RECOVERY] checkpoint=$CHECKPOINT"

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -o pipefail
    cd '$FEATURE_REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$FEATURE_REPO':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE='hmrmax-astar'
    ./run_experiment experiments_numeric.architecture_2.fo_counters_mcts \
      experiments_numeric.domain.fo_counters --resume-from '$CHECKPOINT' \
      --eval-with-mcts --eval-mcts-terminal-safe-action-selection \
      --mcts-progressive-widening --mcts-pw-min-width 3 \
      --mcts-pw-c 0.6 --mcts-pw-alpha 0.5 \
      --mcts-expansion-size 20 --mcts-iterations 70 \
      --mcts-exploration-weight 0.1 --use-estimator 0.5 \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout 21600 --skip-instance-numbers '$SKIP' \
      --num-workers 1 --jpddl-max-heap 4g --worker-logs \
      --disable-value-head --random-seed '$SEED'
  " 2>&1 | tee "$LOG"

python "$PRODUCTION_REPO/asnets/tools/validate_eval_log.py" \
  --log "$LOG" --domain fo_counters --validator "$VALIDATOR"

echo "[FO PW70 RECOVERY] complete=1 target_position=15"

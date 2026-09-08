#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]

set -euo pipefail

: "${ANCHOR_SOURCE_CHECKPOINT:?missing ANCHOR_SOURCE_CHECKPOINT}"
: "${ANCHOR_SEED:?missing ANCHOR_SEED}"
: "${ANCHOR_TEACHER:?missing ANCHOR_TEACHER}"
: "${ANCHOR_TARGET:?missing ANCHOR_TARGET}"

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-anchor-kl
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets

echo "[ANCHOR-KL-CONTROL] commit=$(git -C "$REPO" rev-parse HEAD)"
echo "[ANCHOR-KL-CONTROL] seed=$ANCHOR_SEED source=$ANCHOR_SOURCE_CHECKPOINT"
echo "[ANCHOR-KL-CONTROL] initial_coeff=3 target=$ANCHOR_TARGET teacher=$ANCHOR_TEACHER"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO':\${PYTHONPATH:-}
    ./run_experiment \
      experiments_numeric.architecture_2.tpp_mcts \
      experiments_numeric.domain.tpp \
      --resume-from '$ANCHOR_SOURCE_CHECKPOINT' \
      --resume-train \
      --no-eval \
      --original-training-set \
      --disable-value-head \
      --random-seed '$ANCHOR_SEED' \
      --worker-logs \
      --num-workers 3 \
      --jpddl-max-heap 4g \
      --override-mse-coeff 0.5 \
      --mcts-exploration-weight 0.1 \
      --override-enhsp-config '$ANCHOR_TEACHER' \
      --mcts-expansion-size 20 \
      --mcts-iterations 0 \
      --sample-k-additional-states 0 \
      --use-estimator 0.5 \
      --max-opt-epochs 100 \
      --supervised-lr 0.0003 \
      --policy-anchor-kl-coeff 3 \
      --policy-anchor-kl-mode adaptive_target \
      --policy-anchor-kl-target '$ANCHOR_TARGET'
  "

#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-09/anchor_kl_epoch0_policy/policy_%A_%a.txt

set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-anchor-kl
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ROLE=catastrophic_outlier
    SEED=1972442430
    CHECKPOINT='/home/hersco/bershco-nu-asnets/numeric-asnets-anchor-kl/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp_mcts-2026-09-09T00:32:19.393735/P[domain,pfile1,pfile2,pfile4]-S[0.0003,50,enhsp-hadd-astar]-MO[]-T[518400]-04933186-72aea46/snapshots/snapshot_0_1.0000'
    ;;
  1)
    ROLE=stable_control
    SEED=1963100312
    CHECKPOINT='/home/hersco/bershco-nu-asnets/numeric-asnets-anchor-kl/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp_mcts-2026-09-09T00:32:22.724107/P[domain,pfile1,pfile2,pfile4]-S[0.0003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04933186-2fc32f3/snapshots/snapshot_0_1.0000'
    ;;
  *) exit 2 ;;
esac

echo "[ANCHOR-KL-EPOCH0-POLICY] role=${ROLE} seed=${SEED} checkpoint=${CHECKPOINT}"
test -f "${CHECKPOINT}/weights.joblib"

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
      --resume-from '$CHECKPOINT' \
      --disable-value-head \
      --random-seed '$SEED' \
      --num-workers 5 \
      --jpddl-max-heap 4g
  "

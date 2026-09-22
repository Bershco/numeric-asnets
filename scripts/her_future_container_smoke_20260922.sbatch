#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=ise-cpu128-03,ise-cpu256-[11-12],ise-cpu-intl-[01,05,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-22/her_future_smoke/smoke_%j.out

set -euo pipefail

checkout=/home/hersco/bershco-nu-asnets/her-future-smoke-20260922
production=/home/hersco/bershco-nu-asnets/numeric-asnets
container=/home/hersco/Docker/image.sif
source_checkpoint='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.rover-experiments_numeric.architecture_2.rover-2026-08-20T18:13:54.950033/P[domain,pfile1,pfile2,pfile3,pfile4]-S[0.003,50,enhsp-hmrp-ha-gbfs]-MO[]-T[518400]-04905448-2fc32f3/snapshots/snapshot_70_0.625'
mkdir -p /home/hersco/training_new_domains/2026-09-22/her_future_smoke

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --bind "$production/asnets/asnets/ops/_asnet_ops_impl.so:$checkout/asnets/asnets/ops/_asnet_ops_impl.so" \
  --pwd "$checkout/asnets" \
  "$container" /bin/bash -lc "
    set -euo pipefail
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$checkout':'$checkout/asnets':\${PYTHONPATH:-}
    export ASNET_HER_NETWORK_PREFLIGHT=1
    python -m unittest -v tests.test_her_future
    ./run_experiment \
      experiments_numeric.architecture_2.rover_mcts \
      experiments_numeric.domain.rover \
      --resume-from '$source_checkpoint' \
      --resume-train \
      --no-eval \
      --no-valid \
      --override-enhsp-config hmrp-ha-gbfs \
      --override-mse-coeff 0.5 \
      --max-opt-epochs 1 \
      --supervised-lr 0.0003 \
      --worker-logs \
      --num-workers 1 \
      --mcts-exploration-weight 0.1 \
      --mcts-expansion-size 20 \
      --use-estimator 0.5 \
      --original-training-set \
      --random-seed 1963100312 \
      --mcts-her-strategy future \
      --her-k 2
  "

log=${SLURM_JOB_STDOUT:-/home/hersco/training_new_domains/2026-09-22/her_future_smoke/smoke_${SLURM_JOB_ID}.out}
grep -Eq '\[HER STATS\].*emitted=[1-9][0-9]*' "$log"
grep -Eq '\[HER GOAL PREFLIGHT\].*goal_props=[1-9][0-9]* goal_flnts=0' "$log"
grep -q '\[HER NETWORK PREFLIGHT\] base_equals_static_tail=true changed_tail_consumed=true' "$log"
if grep -q '\[TREE SAMPLE STATS\]' "$log"; then
  echo 'HER-only smoke emitted TREE_SAMPLE records' >&2
  exit 1
fi

#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/imperfect_kl_semantics_screen/historical_epoch0_%j.txt
set -euo pipefail
: "${CHECKOUT:?}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.counters-experiments_numeric.architecture_2.counters_mcts-2026-08-21T18:14:27.409234/P[domain,fz_instance_4,inv_instance_4,rnd_instanc...]-S[0.0003,50,enhsp-hmrmax-astar]-MO[]-T[518400]-04906888-3c74d66/snapshots/snapshot_0_1.0000'
test -f "${SOURCE}/weights.joblib"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail; cd '$REPO/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  ./run_experiment experiments_numeric.architecture_2.counters experiments_numeric.domain.counters \
    --resume-from '$SOURCE' --disable-value-head --random-seed 534933607 \
    --num-workers 5 --jpddl-max-heap 4g
"

#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-1
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a/stage1_probe_%A_%a.txt

set -euo pipefail
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a

case "${SLURM_ARRAY_TASK_ID}" in
  0) ROLE=catastrophic_outlier; SEED=1972442430; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000' ;;
  1) ROLE=stable_control; SEED=2082152039; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667' ;;
  *) exit 2 ;;
esac

OUT="${ROOT}/${ROLE}_${SEED}"
mkdir -p "$OUT/stage1_probe"
export ASN_POLICY_PROBE_CAPTURE_DIR="$OUT/stage1_probe"

apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  cd '$REPO/asnets'
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO':\${PYTHONPATH:-}
  export ASN_POLICY_PROBE_CAPTURE_DIR='$ASN_POLICY_PROBE_CAPTURE_DIR'
  ./run_experiment experiments_numeric.architecture_2.tpp_mcts experiments_numeric.domain.tpp \
    --resume-from '$SOURCE' --disable-value-head --random-seed '$SEED' \
    --num-workers 5 --jpddl-max-heap 4g
"


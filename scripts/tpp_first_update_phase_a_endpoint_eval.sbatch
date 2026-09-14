#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-1
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a/endpoint_%A_%a.txt

set -euo pipefail
REPO=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a

case "${SLURM_ARRAY_TASK_ID}" in
  0) ROLE=catastrophic_outlier; SEED=1972442430 ;;
  1) ROLE=stable_control; SEED=2082152039 ;;
  *) exit 2 ;;
esac
OUT="${ROOT}/${ROLE}_${SEED}"
CHECKPOINT=$(cat "$OUT/endpoint_path.txt")
test -f "${CHECKPOINT}/weights.joblib"
mkdir -p "$OUT/endpoint_probe"
export ASN_POLICY_PROBE_INPUT_DIR="$OUT/stage1_probe"
export ASN_POLICY_PROBE_OUTPUT_DIR="$OUT/endpoint_probe"

apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  cd '$REPO/asnets'
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  export ASN_POLICY_PROBE_INPUT_DIR='$ASN_POLICY_PROBE_INPUT_DIR'
  export ASN_POLICY_PROBE_OUTPUT_DIR='$ASN_POLICY_PROBE_OUTPUT_DIR'
  ./run_experiment experiments_numeric.architecture_2.tpp_mcts experiments_numeric.domain.tpp \
    --resume-from '$CHECKPOINT' --disable-value-head --random-seed '$SEED' \
    --num-workers 5 --jpddl-max-heap 4g
"

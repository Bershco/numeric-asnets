#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-11
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/tpp_kl_multirng_susceptibility/endpoint_%A_%a.txt

set -euo pipefail
: "${CHECKOUT:?Set CHECKOUT to the isolated diagnostic checkout}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the reviewed diagnostic commit}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/tpp_kl_multirng_susceptibility
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"
RNGS=(271828 1618033 20260916)
RNG=${RNGS[$((SLURM_ARRAY_TASK_ID / 4))]}
COMBO=$((SLURM_ARRAY_TASK_ID % 4))
case "$COMBO" in
  0) ROLE=catastrophic_outlier; TREATMENT=legacy_dropout_current; SEED=1972442430; PROBE_INPUT="${PHASE_A}/catastrophic_outlier_1972442430/stage1_probe" ;;
  1) ROLE=catastrophic_outlier; TREATMENT=deterministic_current; SEED=1972442430; PROBE_INPUT="${PHASE_A}/catastrophic_outlier_1972442430/stage1_probe" ;;
  2) ROLE=stable_control; TREATMENT=legacy_dropout_current; SEED=2082152039; PROBE_INPUT="${PHASE_A}/stable_control_2082152039/stage1_probe" ;;
  3) ROLE=stable_control; TREATMENT=deterministic_current; SEED=2082152039; PROBE_INPUT="${PHASE_A}/stable_control_2082152039/stage1_probe" ;;
esac
ARM="${ROLE}_${TREATMENT}_rng${RNG}"
OUT="${ROOT}/${ARM}"
CHECKPOINT=$(cat "${OUT}/endpoint_path.txt")
test -f "${CHECKPOINT}/weights.joblib" && test -d "$PROBE_INPUT"
mkdir -p "${OUT}/endpoint_probe"
export ASN_POLICY_PROBE_INPUT_DIR="$PROBE_INPUT"
export ASN_POLICY_PROBE_OUTPUT_DIR="${OUT}/endpoint_probe"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
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

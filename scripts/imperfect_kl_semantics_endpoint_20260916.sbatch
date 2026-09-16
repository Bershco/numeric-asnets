#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-19
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/imperfect_kl_semantics_screen/endpoint_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"
REPO=$CHECKOUT; PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/imperfect_kl_semantics_screen/v2_exact_legacy_architecture
PAIR=$((SLURM_ARRAY_TASK_ID / 2)); MODE=$((SLURM_ARRAY_TASK_ID % 2))
case "$PAIR" in
  0) DOMAIN=block_grouping; EVAL_MOD=experiments_numeric.architecture.actprop_2l_comparison; SEED=42 ;;
  1) DOMAIN=block_grouping; EVAL_MOD=experiments_numeric.architecture.actprop_2l_comparison; SEED=2026 ;;
  2) DOMAIN=drone; EVAL_MOD=experiments_numeric.architecture.actprop_2l_all; SEED=2026 ;;
  3) DOMAIN=drone; EVAL_MOD=experiments_numeric.architecture.actprop_2l_all; SEED=42 ;;
  4) DOMAIN=fo_counters; EVAL_MOD=experiments_numeric.architecture.actprop_2l_fluent; SEED=2026 ;;
  5) DOMAIN=fo_counters; EVAL_MOD=experiments_numeric.architecture.actprop_2l_fluent; SEED=42 ;;
  6) DOMAIN=rover; EVAL_MOD=experiments_numeric.architecture.actprop_2l_all; SEED=42 ;;
  7) DOMAIN=rover; EVAL_MOD=experiments_numeric.architecture.actprop_2l_all; SEED=2026 ;;
  8) DOMAIN=counters; EVAL_MOD=experiments_numeric.architecture_2.counters; SEED=534933607 ;;
  9) DOMAIN=counters; EVAL_MOD=experiments_numeric.architecture_2.counters; SEED=2082152039 ;;
esac
if (( MODE == 0 )); then TREATMENT=now_legacy; else TREATMENT=deterministic_current; fi
OUT="${ROOT}/${DOMAIN}_${SEED}_${TREATMENT}"; CHECKPOINT=$(cat "${OUT}/endpoint_path.txt"); test -f "${CHECKPOINT}/weights.joblib"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
 set -euo pipefail; cd '$REPO/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate'; export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
 ./run_experiment '$EVAL_MOD' experiments_numeric.domain.'$DOMAIN' --resume-from '$CHECKPOINT' --disable-value-head --random-seed '$SEED' --num-workers 5 --jpddl-max-heap 4g
"

#!/bin/bash
# Rebuild only the eight scientific state manifests from durable source captures.
# Submit as --array=0-7 after all exact source recoveries succeed.
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/manifest_%A_%a.txt

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1

case "$SLURM_ARRAY_TASK_ID" in
  0) DOMAIN=drone; SEED=534933607 ;;
  1) DOMAIN=drone; SEED=923500475 ;;
  2) DOMAIN=fo_counters; SEED=534933607 ;;
  3) DOMAIN=fo_counters; SEED=923500475 ;;
  4) DOMAIN=rover; SEED=534933607 ;;
  5) DOMAIN=rover; SEED=923500475 ;;
  6) DOMAIN=mprime; SEED=534933607 ;;
  7) DOMAIN=mprime; SEED=923500475 ;;
  *) exit 2 ;;
esac

OUT="$ROOT/captures/${DOMAIN}_${SEED}"
MANIFEST="$OUT/v1_states/vhv1-${DOMAIN}-${SEED}-paired.jsonl"
apptainer exec --bind /home/hersco:/home/hersco /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  python '$REPO/scripts/materialize_value_head_v1_state_manifest.py' \
    --domain '$DOMAIN' --seed '$SEED' --capture-root '$OUT' --output '$MANIFEST'
"

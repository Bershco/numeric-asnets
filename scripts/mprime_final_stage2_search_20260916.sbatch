#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --array=0-39
#SBATCH --job-name=MPRIME_FINAL_S2_SEARCH
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_search/slurm/%A_%a.out

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"
CHECKOUT=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION=${PRODUCTION:-/home/hersco/bershco-nu-asnets/numeric-asnets}
CONTAINER=${CONTAINER:-/home/hersco/Docker/image.sif}
VALIDATOR=${VALIDATOR:-/home/hersco/tools/VAL/build/bin/Validate}
CAMPAIGN=${CAMPAIGN:-/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_search}
MANIFEST=${MANIFEST:-$CAMPAIGN/manifest.csv}

mkdir -p "$CAMPAIGN/slurm"
python "$CAMPAIGN/run_mprime_final_stage2_search_20260916.py" \
  --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --checkout "$CHECKOUT" --production "$PRODUCTION" \
  --container "$CONTAINER" --validator "$VALIDATOR" \
  --output-root "$CAMPAIGN" --code-commit "$CODE_COMMIT"

#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --job-name=MPRIME_S2_EXACT_REC
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_search/reconciliation_20260917/slurm/%A_%a.out

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${RECOVERY_MANIFEST:?RECOVERY_MANIFEST is required}"

CHECKOUT=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION=${PRODUCTION:-/home/hersco/bershco-nu-asnets/numeric-asnets}
CONTAINER=${CONTAINER:-/home/hersco/Docker/image.sif}
VALIDATOR=${VALIDATOR:-/home/hersco/tools/VAL/build/bin/Validate}
CAMPAIGN=${CAMPAIGN:-/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_search}
MANIFEST=${MANIFEST:-$CAMPAIGN/manifest.csv}

mkdir -p "$CAMPAIGN/reconciliation_20260917/slurm"
python "$CHECKOUT/scripts/run_mprime_final_stage2_search_exact_recovery_20260917.py" \
  --manifest "$MANIFEST" \
  --recovery-manifest "$RECOVERY_MANIFEST" \
  --recovery-index "$SLURM_ARRAY_TASK_ID" \
  --checkout "$CHECKOUT" --production "$PRODUCTION" \
  --container "$CONTAINER" --validator "$VALIDATOR" \
  --campaign-root "$CAMPAIGN" --code-commit "$CODE_COMMIT"

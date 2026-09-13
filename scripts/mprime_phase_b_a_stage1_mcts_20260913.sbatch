#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --array=0-9
#SBATCH --job-name=MPRIME_S1_PBA_MCTS
#SBATCH --output=/home/hersco/training_new_domains/2026-09-13/mprime_phase_b_a_stage1_mcts/slurm/%A_%a.out

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${VALUE_HEAD:?VALUE_HEAD must be off or on}"
[[ "$VALUE_HEAD" == off || "$VALUE_HEAD" == on ]]

CHECKOUT=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION=${PRODUCTION:-/home/hersco/bershco-nu-asnets/numeric-asnets}
CONTAINER=${CONTAINER:-/home/hersco/Docker/image.sif}
VALIDATOR=${VALIDATOR:-/home/hersco/tools/VAL/build/bin/Validate}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/hersco/training_new_domains/2026-09-13/mprime_phase_b_a_stage1_mcts}
MANIFEST=${MANIFEST:-$CHECKOUT/experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/manifest_${VALUE_HEAD}.csv}

mkdir -p "$OUTPUT_ROOT/slurm"

python "$CHECKOUT/scripts/run_mprime_phase_b_a_stage1_mcts_20260913.py" \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --checkout "$CHECKOUT" \
  --production "$PRODUCTION" \
  --container "$CONTAINER" \
  --validator "$VALIDATOR" \
  --output-root "$OUTPUT_ROOT" \
  --code-commit "$CODE_COMMIT"

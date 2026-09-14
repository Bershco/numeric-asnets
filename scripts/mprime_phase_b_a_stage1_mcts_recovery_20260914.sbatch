#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --array=0-14
#SBATCH --job-name=MPRIME_S1_EXACT_REC
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/mprime_phase_b_a_stage1_mcts_recovery/slurm/%A_%a.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"

CHECKOUT=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION=${PRODUCTION:-/home/hersco/bershco-nu-asnets/numeric-asnets}
CONTAINER=${CONTAINER:-/home/hersco/Docker/image.sif}
VALIDATOR=${VALIDATOR:-/home/hersco/tools/VAL/build/bin/Validate}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/hersco/training_new_domains/2026-09-14/mprime_phase_b_a_stage1_mcts_recovery}
DEPLOY_DIR=${DEPLOY_DIR:-$CHECKOUT/scripts}
RECOVERY_MANIFEST=${RECOVERY_MANIFEST:-$CHECKOUT/experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/recovery_manifest_20260914.csv}

mkdir -p "$OUTPUT_ROOT/slurm"
python "$DEPLOY_DIR/run_mprime_phase_b_a_stage1_mcts_recovery_20260914.py" \
  --recovery-manifest "$RECOVERY_MANIFEST" \
  --recovery-index "$SLURM_ARRAY_TASK_ID" \
  --checkout "$CHECKOUT" --production "$PRODUCTION" \
  --container "$CONTAINER" --validator "$VALIDATOR" \
  --output-root "$OUTPUT_ROOT" --code-commit "$CODE_COMMIT"

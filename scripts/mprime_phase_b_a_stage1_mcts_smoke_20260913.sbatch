#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --job-name=MPRIME_S1_PBA_SMOKE
#SBATCH --output=/home/hersco/training_new_domains/2026-09-13/mprime_phase_b_a_stage1_mcts/smoke_%j.out

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"
VALUE_HEAD=${VALUE_HEAD:-off}
SMOKE_INDEX=${SMOKE_INDEX:-0}
SMOKE_INSTANCE=${SMOKE_INSTANCE:-1}
[[ "$VALUE_HEAD" == off || "$VALUE_HEAD" == on ]]

CHECKOUT=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION=${PRODUCTION:-/home/hersco/bershco-nu-asnets/numeric-asnets}
CONTAINER=${CONTAINER:-/home/hersco/Docker/image.sif}
VALIDATOR=${VALIDATOR:-/home/hersco/tools/VAL/build/bin/Validate}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/hersco/training_new_domains/2026-09-13/mprime_phase_b_a_stage1_mcts}
MANIFEST=${MANIFEST:-$CHECKOUT/experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/manifest_${VALUE_HEAD}.csv}

mkdir -p "$OUTPUT_ROOT"

python "$CHECKOUT/scripts/run_mprime_phase_b_a_stage1_mcts_20260913.py" \
  --manifest "$MANIFEST" \
  --index "$SMOKE_INDEX" \
  --checkout "$CHECKOUT" \
  --production "$PRODUCTION" \
  --container "$CONTAINER" \
  --validator "$VALIDATOR" \
  --output-root "$OUTPUT_ROOT" \
  --code-commit "$CODE_COMMIT" \
  --smoke-instance "$SMOKE_INSTANCE" \
  --smoke-timeout 600

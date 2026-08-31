#!/bin/bash
#SBATCH --job-name=STAGE2_POLICY_REFRESH
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-31/stage2_policy_controllers/controller_%j.log

set -euo pipefail

: "${TRAINING_LEDGER:?TRAINING_LEDGER is required}"
: "${READY_MANIFEST:?READY_MANIFEST is required}"
: "${POLICY_LEDGER:?POLICY_LEDGER is required}"
: "${ROLE_PREFIX:?ROLE_PREFIX is required}"
: "${SUFFIX_PREFIX:?SUFFIX_PREFIX is required}"
: "${OUTPUT_PREFIX:?OUTPUT_PREFIX is required}"

CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PYTHON=/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python
mkdir -p "$(dirname "$READY_MANIFEST")" "$(dirname "$POLICY_LEDGER")" \
  /home/hersco/training_new_domains/2026-08-31/stage2_policy_controllers

ARGS=("$TRAINING_LEDGER" "$READY_MANIFEST" --role-prefix "$ROLE_PREFIX")
if [[ -n "${DOMAIN_FILTER:-}" ]]; then ARGS+=(--domain "$DOMAIN_FILTER"); fi
if [[ -n "${BRANCH_FILTER:-}" ]]; then ARGS+=(--branch "$BRANCH_FILTER"); fi
if [[ "${EXCLUDE_REUSED:-0}" == 1 ]]; then ARGS+=(--exclude-reused); fi

"$PYTHON" "$CHECKOUT/experiment_tracking/materialize_stage2_policy_from_ledger.py" "${ARGS[@]}"
"$PYTHON" "$CHECKOUT/experiment_tracking/submit_stage2_policy.py" \
  --manifest "$READY_MANIFEST" --ledger "$POLICY_LEDGER" \
  --suffix-prefix "$SUFFIX_PREFIX" --output-prefix "$OUTPUT_PREFIX" \
  --queue-cap 1999 --max-per-cycle 100

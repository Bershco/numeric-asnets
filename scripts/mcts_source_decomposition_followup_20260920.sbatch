#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --exclude=ise-cpu-intl-[01,05,15,18,24,26]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-20/mcts_source_decomposition_followup/slurm_%A_%a.out

set -euo pipefail
set -o pipefail

FEATURE_REPO=${FEATURE_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe-context}
PRODUCTION_REPO=${PRODUCTION_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets}
OUT=${OUT:-/home/hersco/training_new_domains/2026-09-20/mcts_source_decomposition_followup}
MANIFEST=${MANIFEST:-$FEATURE_REPO/experiment_tracking/mcts_policy_divergence_cause_audit/source_decomposition_followup_manifest.json}
FREEZE=${FREEZE:-$FEATURE_REPO/experiment_tracking/mcts_policy_divergence_cause_audit/source_decomposition_followup_manifest.freeze.json}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-$FEATURE_REPO/experiment_tracking/mcts_policy_divergence_cause_audit/stage1_grouped_tasks.csv}
RECONCILIATION=${RECONCILIATION:-$FEATURE_REPO/experiment_tracking/mcts_policy_divergence_cause_audit/stage1_final_40_reconciled.csv}
REGISTRY=${REGISTRY:-$FEATURE_REPO/experiment_tracking/experiment_results.csv}
CODE_COMMIT=${CODE_COMMIT:?export the reviewed deployed commit}
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit as array 0-5}

[[ $(git -C "$FEATURE_REPO" rev-parse HEAD) == "$CODE_COMMIT" ]] || {
  echo "deployed checkout does not match CODE_COMMIT=$CODE_COMMIT" >&2
  exit 2
}

mkdir -p "$OUT/slurm"
source "$PRODUCTION_REPO/venv-asnets/bin/activate"
export PYTHONPATH="$FEATURE_REPO:${PYTHONPATH:-}"

python "$FEATURE_REPO/scripts/run_mcts_source_decomposition_followup_20260920.py" \
  --repo "$FEATURE_REPO" \
  --manifest "$MANIFEST" \
  --freeze "$FREEZE" \
  --source-manifest "$SOURCE_MANIFEST" \
  --reconciliation "$RECONCILIATION" \
  --registry "$REGISTRY" \
  --task-index "$TASK_ID" \
  --output-root "$OUT" \
  --code-commit "$CODE_COMMIT" \
  --execute

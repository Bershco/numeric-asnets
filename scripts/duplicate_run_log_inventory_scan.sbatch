#!/usr/bin/env bash
#SBATCH --job-name=dup-log-scan
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array}"

python3 "$REPO_ROOT/scripts/inventory_duplicate_run_logs.py" scan-shard \
  --campaign "$CAMPAIGN_DIR" \
  --shard "$SLURM_ARRAY_TASK_ID"

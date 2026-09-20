#!/usr/bin/env bash
#SBATCH --job-name=dup-log-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"

python3 "$REPO_ROOT/scripts/inventory_duplicate_run_logs.py" summarize \
  --campaign "$CAMPAIGN_DIR"

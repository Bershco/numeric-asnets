#!/usr/bin/env bash
#SBATCH --job-name=dup-log-prep
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${SCAN_ROOT:?set SCAN_ROOT}"
: "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"
: "${SHARD_COUNT:=32}"

python3 "$REPO_ROOT/scripts/inventory_duplicate_run_logs.py" prepare \
  --root "$SCAN_ROOT" \
  --campaign "$CAMPAIGN_DIR" \
  --shards "$SHARD_COUNT"

#!/usr/bin/env bash
set -euo pipefail

repo=/home/hersco/bershco-nu-asnets/numeric-asnets
campaign=/home/hersco/quota_recovery/duplicate_run_log_inventory_20260920_attempt3
logdir=/home/hersco/quota_recovery/duplicate_run_log_inventory_20260920_attempt3_slurm
exclude='ise-cpu-intl-[01,05,15,18,24,26]'

mkdir -p "$logdir"
python3 -m py_compile "$repo/scripts/inventory_duplicate_run_logs.py"
bash -n \
  "$repo/scripts/duplicate_run_log_inventory_prepare.sbatch" \
  "$repo/scripts/duplicate_run_log_inventory_scan.sbatch" \
  "$repo/scripts/duplicate_run_log_inventory_summarize.sbatch"

prep=$(sbatch --parsable \
  --no-requeue \
  --exclude="$exclude" \
  --output="$logdir/prepare-%j.out" \
  --export="ALL,REPO_ROOT=$repo,SCAN_ROOT=$repo/asnets/experiment-results,CAMPAIGN_DIR=$campaign,SHARD_COUNT=32" \
  "$repo/scripts/duplicate_run_log_inventory_prepare.sbatch")
scan=$(sbatch --parsable \
  --no-requeue \
  --exclude="$exclude" \
  --dependency="afterok:$prep" \
  --array=0-31%4 \
  --output="$logdir/scan-%A_%a.out" \
  --export="ALL,REPO_ROOT=$repo,CAMPAIGN_DIR=$campaign" \
  "$repo/scripts/duplicate_run_log_inventory_scan.sbatch")
summary=$(sbatch --parsable \
  --no-requeue \
  --exclude="$exclude" \
  --dependency="afterany:$scan" \
  --output="$logdir/summary-%j.out" \
  --export="ALL,REPO_ROOT=$repo,CAMPAIGN_DIR=$campaign" \
  "$repo/scripts/duplicate_run_log_inventory_summarize.sbatch")

printf 'prep=%s scan=%s summary=%s\n' "$prep" "$scan" "$summary"

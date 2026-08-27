#!/usr/bin/env bash
#SBATCH --job-name=CD4_POLICY_CONT_REFRESH
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-26/quota_recovery/controller_anchor_policy_cont_%j.log

set -euo pipefail
work=/home/hersco/training_new_domains/2026-08-25/completed_domains_anchor_tuning
recovery=/home/hersco/training_new_domains/2026-08-26/quota_recovery
cd "$work"
python3 cd4_policy_controller.py \
  --policy-manifest "$recovery/anchor_policy_completed_continuations_20260827.csv" \
  --ledger "$recovery/anchor_policy_completed_continuations_20260827_submissions.tsv" \
  --queue-cap 1999 --max-per-cycle 100 --poll-seconds 60

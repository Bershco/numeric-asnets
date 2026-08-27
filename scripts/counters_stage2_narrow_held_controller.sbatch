#!/usr/bin/env bash
#SBATCH --job-name=COUNTERS_S2_NARROW_HELD
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-27/counters_stage2_narrow_held/controller_%j.log

set -euo pipefail
campaign=/home/hersco/training_new_domains/2026-08-27/counters_stage2_narrow_held
mkdir -p "$campaign"

# Propagates to the sbatch calls made by submit_training.sh, not to this
# already-running controller. Every replacement evaluation enters Slurm held.
export SBATCH_HOLD=1

python3 "$campaign/statistical_replication_manifest_controller.py" \
  --policy-manifest "$campaign/empty_policy_manifest.csv" \
  --mcts-manifest "$campaign/main_term_stage2_narrow_held_manifest.csv" \
  --ledger "$campaign/submissions.tsv" \
  --queue-cap 1999 --max-per-cycle 20 --poll-seconds 10 \
  --allow-empty-manifests

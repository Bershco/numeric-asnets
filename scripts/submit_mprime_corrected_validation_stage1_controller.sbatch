#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_VAL_V1_S1_CTL
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-26/mprime_validation_ipc_scale_v1/stage1_controller_%j.log

set -euo pipefail

campaign=/home/hersco/training_new_domains/2026-08-26/mprime_validation_ipc_scale_v1
python3 "$campaign/submit_mprime_corrected_validation_stage1.py" \
  --manifest "$campaign/stage1_corrected_validation_manifest.csv" \
  --ledger "$campaign/stage1_corrected_validation_submission_ledger.tsv"

#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/recovery_controller_%j.txt

set -euo pipefail
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
bash "$REPO/scripts/submit_value_head_v1_exact_recovery_20260919.sh"

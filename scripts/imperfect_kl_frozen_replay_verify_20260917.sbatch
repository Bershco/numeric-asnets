#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_frozen_replay_crossover/verify_%j.txt
set -euo pipefail
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_frozen_replay_crossover
python3 "$ROOT/verify_imperfect_kl_frozen_replay_pairs_20260917.py" \
  "$ROOT" "$ROOT/manifest.csv" "$ROOT/pair_verification.csv"

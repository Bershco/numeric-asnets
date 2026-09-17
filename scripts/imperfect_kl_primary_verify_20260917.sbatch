#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover/verify_%j.txt
set -euo pipefail
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover
python3 "$ROOT/verify_pairs.py" "$ROOT" "$ROOT/new_pairs_verified.csv" "$ROOT/pair_verification.csv"

#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover/finalize_%j.txt
set -euo pipefail
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover
python3 "$ROOT/finalize.py" "$ROOT" "$ROOT/primary_lineages_verified.csv"

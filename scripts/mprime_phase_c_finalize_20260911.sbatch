#!/bin/bash
#SBATCH --job-name=MPRIME_PHASEC_FINALIZE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-11/mprime_phase_c/finalize_%j.log
set -euo pipefail
work=/home/hersco/training_new_domains/2026-09-11/mprime_phase_c
python -u "$work/mprime_phase_c_finalize_20260911.py"

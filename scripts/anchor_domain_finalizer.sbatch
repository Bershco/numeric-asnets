#!/usr/bin/env bash
#SBATCH --job-name=ANCHOR_DOMAIN_FINALIZER
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-27/anchor_domain_finalizers/%x_%j.log

set -euo pipefail
domain=${1:?domain required}
campaign=/home/hersco/training_new_domains/2026-08-27/anchor_domain_finalizers
mkdir -p "$campaign"
python3 "$campaign/finalize_anchor_domain.py" "$domain"

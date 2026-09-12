#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_ANCHOR_PBA
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a/rescore_%A_%a.log

set -euo pipefail

repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
work=/home/hersco/training_new_domains/2026-09-12/mprime_anchor_phase_b_a
manifest="$repo/experiment_tracking/mprime_anchor_phase_b_a_20260912/manifest.csv"
validator_manifest="$repo/experiment_tracking/mprime_validation_phase_b_20260906/frozen_validation_manifest.csv"
validator_candidates="$repo/experiment_tracking/mprime_validation_phase_b_20260906/candidates"
extra=()
if [[ ${PREFLIGHT:-0} == 1 ]]; then extra+=(--preflight); fi

mkdir -p "$work"
apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --pwd "$repo" \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    python -u '$repo/scripts/mprime_anchor_phase_b_a_rescore_20260912.py' \
      --repo '$repo' \
      --work '$work' \
      --manifest '$manifest' \
      --validator-manifest '$validator_manifest' \
      --validator-candidates '$validator_candidates' \
      --task '${SLURM_ARRAY_TASK_ID:-0}' \
      ${extra[*]}
  "

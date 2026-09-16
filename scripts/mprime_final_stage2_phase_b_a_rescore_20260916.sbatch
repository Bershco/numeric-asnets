#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_S2_PBA
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_phase_b_a/rescore_%A_%a.log

set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
work=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_phase_b_a
extra=()
if [[ ${PREFLIGHT:-0} == 1 ]]; then extra+=(--preflight); fi
mkdir -p "$work"
apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --pwd "$repo" /home/hersco/Docker/image.sif /bin/bash -lc "
    source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    python -u '$work/mprime_final_stage2_phase_b_a_rescore_20260916.py' \
      --repo '$repo' --work '$work' \
      --manifest '$work/manifest.csv' \
      --ready-manifest '$work/ready_manifest.csv' \
      --validator-manifest '$repo/experiment_tracking/mprime_validation_phase_b_20260906/frozen_validation_manifest.csv' \
      --validator-candidates '$repo/experiment_tracking/mprime_validation_phase_b_20260906/candidates' \
      --code-commit 'ef274cd23605de815800656c97b360ee7ff584b8' \
      --task '${SLURM_ARRAY_TASK_ID:-0}' ${extra[*]}
  "

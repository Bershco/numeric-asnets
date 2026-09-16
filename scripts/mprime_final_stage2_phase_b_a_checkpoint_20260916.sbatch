#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_S2_PBA_FINE
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --exclude=ise-cpu128-03
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_phase_b_a/fine_slurm/%A_%a.log

set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
work=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_phase_b_a
: "${KEY_MANIFEST:?KEY_MANIFEST required}"
: "${KEY_MANIFEST_SHA256:?KEY_MANIFEST_SHA256 required}"
mkdir -p "$work/fine_slurm"
apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --pwd "$repo" /home/hersco/Docker/image.sif /bin/bash -lc "
    source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    python -u '$work/mprime_final_stage2_phase_b_a_checkpoint_20260916.py' \
      --repo '$repo' --work '$work' \
      --key-manifest '$KEY_MANIFEST' --key-manifest-sha256 '$KEY_MANIFEST_SHA256' \
      --validator-manifest '$repo/experiment_tracking/mprime_validation_phase_b_20260906/frozen_validation_manifest.csv' \
      --validator-candidates '$repo/experiment_tracking/mprime_validation_phase_b_20260906/candidates' \
      --task '${SLURM_ARRAY_TASK_ID}'
  "

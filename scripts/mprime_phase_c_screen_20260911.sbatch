#!/bin/bash
#SBATCH --job-name=MPRIME_PHASEC_SCREEN
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --exclude=ise-cpu128-[03-04],ise-cpu-intl-[08-15,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-11/mprime_phase_c/planner_%A_%a.log
set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
work=/home/hersco/training_new_domains/2026-09-11/mprime_phase_c
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd --pwd "$repo" /home/hersco/Docker/image.sif /bin/bash -lc "source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate; python -u '$work/mprime_phase_c_20260911.py' screen --root '$work' --repo '$repo' --task '${SLURM_ARRAY_TASK_ID:-0}'"

#!/bin/bash
#SBATCH --job-name=MPRIME_PHASEB_PLANNER
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --exclude=ise-cpu128-[03-04],ise-cpu-intl-[08-15,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-06/mprime_phase_b/planner_%A_%a.log
set -euo pipefail
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
work=/home/hersco/training_new_domains/2026-09-06/mprime_phase_b
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd --pwd "$repo" /home/hersco/Docker/image.sif /bin/bash -lc "source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate; python '$work/mprime_phase_b_20260906.py' screen --root '$work' --repo '$repo' --task '${SLURM_ARRAY_TASK_ID:-0}'"

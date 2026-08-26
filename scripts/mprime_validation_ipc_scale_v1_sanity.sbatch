#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_VAL_V1_SANITY
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-26/mprime_validation_ipc_scale_v1/sanity_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

repo=/home/hersco/bershco-nu-asnets/numeric-asnets
work=/home/hersco/training_new_domains/2026-08-26/mprime_validation_ipc_scale_v1
audit_root="$work/audit_input"

mkdir -p "$audit_root/mprime"
for tier in easy medium hard; do
  ln -sfn "$repo/problems/numeric/mprime/validation_ipc_scale_v1/$tier" \
    "$audit_root/mprime/$tier"
done

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --pwd "$repo" \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    source venv-asnets/bin/activate
    python -u problem_generator/audit_generated_instances_enhsp.py \
      --project-root '$repo' \
      --problem-generator-root '$repo/problem_generator' \
      --domain-root '$repo/problems/numeric' \
      --generated-root '$audit_root' \
      --only-domain mprime \
      --expected-per-difficulty 10 \
      --planner \
      --enhsp-config hmrp-ha-gbfs \
      --enhsp-timeout 60 \
      --output-csv '$work/planner_sanity.csv'
  "

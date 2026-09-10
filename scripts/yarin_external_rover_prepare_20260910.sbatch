#!/usr/bin/env bash
#SBATCH --job-name=YARIN_ROVER_PREP
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/hersco/training_new_domains/2026-09-10/yarin_external/rover_prep_%j.log

set -euo pipefail
FEATURE_REPO=${FEATURE_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-yarin-external}
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-10/yarin_external
CONTAINER=/home/hersco/Docker/image.sif
chmod 755 "$ROOT/upstream/rovergen"
apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$FEATURE_REPO'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    python scripts/prepare_yarin_external_sets.py --repo-root . \
      --output-root '$ROOT/frozen' --rovergen '$ROOT/upstream/rovergen' \
      --domains rover
    python scripts/analyze_external_generator_distribution.py --repo-root . \
      --external-root '$ROOT/frozen' --output-dir '$ROOT/distribution'
    test \"\$(find '$ROOT/frozen/rover/external' -name '*.pddl' | wc -l)\" -eq 20
    ! grep -REiq '\(at[[:space:]]' '$ROOT/frozen/rover/external'
    grep -REiq '\(in[[:space:]]+rover' '$ROOT/frozen/rover/external'
  "

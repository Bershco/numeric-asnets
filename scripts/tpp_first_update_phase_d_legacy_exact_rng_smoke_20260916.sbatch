#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_d_legacy_exact_rng/smoke_%j.txt

set -euo pipefail
: "${CHECKOUT:?Set CHECKOUT to the isolated diagnostic checkout}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the reviewed diagnostic commit}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_d_legacy_exact_rng
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
    ./run_experiment --help | grep -q -- '--policy-anchor-fixed-step-rng-seed'
    python -m unittest \
      tests.test_replay_rng \
      tests.test_policy_anchor_kl_controller \
      tests.test_policy_anchor_trust_region
    python '$ROOT/smoke_tpp_fixed_step_rng.py'
    python -m py_compile \
      asnets/supervised.py \
      asnets/scripts/run_asnets.py \
      asnets/scripts/run_experiment.py
  "

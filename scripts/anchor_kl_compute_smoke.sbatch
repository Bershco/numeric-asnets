#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=anchor-kl-smoke
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-08/anchor_kl_control/smoke_%j.out

set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-anchor-kl
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets

echo "[SMOKE] node=$(hostname) commit=$(git -C "$REPO" rev-parse HEAD)"
test "$(git -C "$REPO" rev-parse HEAD)" = 9db17cc891c9e4fe04d676b3d1b41fba136dba90
test -z "$(git -C "$REPO" status --porcelain)"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO':\${PYTHONPATH:-}
    test -e '$REPO/asnets/asnets/ops/_asnet_ops_impl.so'
    python -c 'import asnets.ops.asnet_ops'
    python -m unittest tests.test_policy_anchor_kl_controller
    ./run_experiment --help | grep -q -- '--policy-anchor-kl-mode'
    ./run_experiment --help | grep -q -- '--policy-anchor-kl-target'
  "

echo '[SMOKE] PASS'

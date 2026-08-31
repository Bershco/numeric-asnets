#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
OVERLAY=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_audit/overlay
CONTAINER=/home/hersco/Docker/image.sif

test -s "$OVERLAY/spawn_train_worker.py"
test -s "$OVERLAY/mcts_determinism_audit.sbatch"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  --bind "$OVERLAY/spawn_train_worker.py:$CHECKOUT/asnets/asnets/spawn_train_worker.py" \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT/asnets'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    python -m py_compile asnets/spawn_train_worker.py
    python - <<'PY'
from asnets.spawn_train_worker import print_mcts_determinism_record
assert callable(print_mcts_determinism_record)
print('[MCTS-DETERMINISM-PREFLIGHT] import=ok')
PY
  "


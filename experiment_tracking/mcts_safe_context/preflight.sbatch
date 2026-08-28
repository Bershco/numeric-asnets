#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --job-name=mcts-safe-context-preflight
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-28/mcts_safe_context/preflight_%j.out

set -euo pipefail
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif
mkdir -p /home/hersco/training_new_domains/2026-08-28/mcts_safe_context

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    python -m py_compile \
      post_training/monte_carlo_tree_search.py \
      post_training/training_mcts.py \
      asnets/asnets/state_reprs.py \
      asnets/asnets/scripts/run_asnets.py \
      asnets/asnets/scripts/run_experiment.py \
      asnets/asnets/parllel_explore_spawn_grads.py \
      asnets/asnets/spawn_train_worker.py
    python -m unittest discover \
      -s asnets/tests -p test_progressive_widening.py
  "

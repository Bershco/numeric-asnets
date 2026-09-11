#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=COUNTERS_TIE_SMOKE
#SBATCH --output=/home/hersco/training_new_domains/2026-09-11/counters_tie_break_3way/%j.smoke.out

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"
CHECKOUT=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
PRODUCTION=/home/hersco/bershco-nu-asnets/numeric-asnets
CONTAINER=/home/hersco/Docker/image.sif

mkdir -p /home/hersco/training_new_domains/2026-09-11/counters_tie_break_3way
ACTUAL_COMMIT=$(git -C "$CHECKOUT" rev-parse HEAD)
[[ "$ACTUAL_COMMIT" == "$CODE_COMMIT" ]]

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -euo pipefail
    cd '$CHECKOUT'
    source '$PRODUCTION/venv-asnets/bin/activate'
    export PYTHONPATH='$CHECKOUT':'$CHECKOUT/asnets':\${PYTHONPATH:-}
    python -m unittest \
      asnets.tests.test_progressive_widening.RootVisitTieBreakTests \
      asnets.tests.test_progressive_widening.HorizonTests \
      asnets.tests.test_progressive_widening.TerminalSafeActionSelectionTests
    cd '$CHECKOUT/asnets'
    ./run_experiment --help | grep -q -- '--eval-mcts-root-visit-tie-break'
  "

echo "[COUNTERS TIE SMOKE] completed=1 commit=$ACTUAL_COMMIT"

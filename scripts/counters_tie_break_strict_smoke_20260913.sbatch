#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=COUNTERS_TIE_STRICT_SMOKE
#SBATCH --output=/home/hersco/training_new_domains/2026-09-13/counters_tie_break_strict_stage1/%j.smoke.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
production=/home/hersco/bershco-nu-asnets/numeric-asnets
manifest="$repo/experiment_tracking/counters_tie_break_strict_stage1_20260913/manifest.csv"
actual_commit=$(git -C "$repo" rev-parse HEAD)
[[ "$actual_commit" == "$CODE_COMMIT" ]]
mkdir -p /home/hersco/training_new_domains/2026-09-13/counters_tie_break_strict_stage1

python - <<'PY'
import csv
from pathlib import Path
p = Path('/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break/experiment_tracking/counters_tie_break_strict_stage1_20260913/manifest.csv')
rows = list(csv.DictReader(p.open()))
assert len(rows) == 20
assert {r['tie_break'] for r in rows} == {'action_id', 'policy'}
assert len({r['seed'] for r in rows}) == 10
assert all(Path(r['checkpoint']).is_dir() for r in rows)
print('manifest/checkpoints OK')
PY

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    cd '$repo'
    source '$production/venv-asnets/bin/activate'
    export PYTHONPATH='$repo':'$repo/asnets':\${PYTHONPATH:-}
    python -m unittest discover -s '$repo/asnets/tests' -p 'test_progressive_widening.py'
    cd '$repo/asnets'
    ./run_experiment --help | grep -q -- '--eval-mcts-root-visit-tie-break'
  "
echo '[COUNTERS TIE STRICT SMOKE] completed=1'

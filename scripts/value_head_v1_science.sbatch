#!/bin/bash
# Submit one 0-3 array per domain through the gated release controller only.
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/science_%x_%A_%a.txt

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${DOMAIN:?missing DOMAIN}"
: "${READY_CANDIDATES:?missing READY_CANDIDATES}"
: "${RESOURCE_GATE:?missing RESOURCE_GATE}"
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
read -r EXPECTED_SHA GATE_STATUS < <(python3 - "$RESOURCE_GATE" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
print(gate.get('candidate_manifest_sha256', ''), gate.get('status', ''))
PY
)
test "$GATE_STATUS" = pass
test "$(sha256sum "$READY_CANDIDATES" | awk '{print $1}')" = "$EXPECTED_SHA"
TASK_INDEX=$(python3 - "$READY_CANDIDATES" "$DOMAIN" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='')))
indices = [i for i, row in enumerate(rows) if row['domain'] == sys.argv[2]]
assert len(indices) == 4, indices
print(indices[int(sys.argv[3])])
PY
)
TASK_ID=$(python3 - "$READY_CANDIDATES" "$TASK_INDEX" <<'PY'
import csv, sys
print(list(csv.DictReader(open(sys.argv[1], newline='')))[int(sys.argv[2])]['task_id'])
PY
)
OUT="$ROOT/science/$TASK_ID"
mkdir -p "$OUT"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  cd '$REPO/asnets'
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  export SLURM_JOB_ID='${SLURM_JOB_ID}'
  python '$REPO/scripts/run_value_head_v1_task.py' \
    --candidates '$READY_CANDIDATES' --task-index '$TASK_INDEX' \
    --output-dir '$OUT' --mode full
"

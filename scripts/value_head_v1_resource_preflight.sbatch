#!/bin/bash
# Submit one 0-1 array per domain with domain-specific memory/time.
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/resource_%x_%A_%a.txt

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${DOMAIN:?submit with DOMAIN}"
: "${READY_CANDIDATES:?submit with READY_CANDIDATES}"
: "${EXPECTED_CANDIDATES_SHA256:?submit with EXPECTED_CANDIDATES_SHA256}"
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
test "$(sha256sum "$READY_CANDIDATES" | awk '{print $1}')" = "$EXPECTED_CANDIDATES_SHA256"
case "$SLURM_ARRAY_TASK_ID" in 0) STAGE=stage1 ;; 1) STAGE=stage2 ;; *) exit 2 ;; esac
TASK_INDEX=$(python3 - "$READY_CANDIDATES" "$DOMAIN" "$STAGE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='')))
matches = [i for i, row in enumerate(rows) if row['domain'] == sys.argv[2]
           and row['seed'] == '534933607' and row['stage'] == sys.argv[3]]
assert len(matches) == 1, matches
print(matches[0])
PY
)
OUT="$ROOT/resource_preflight/${DOMAIN}_${STAGE}"
mkdir -p "$OUT"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  cd '$REPO/asnets'
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  export SLURM_JOB_ID='${SLURM_JOB_ID}'
  python '$REPO/scripts/run_value_head_v1_task.py' \
    --candidates '$READY_CANDIDATES' --task-index '$TASK_INDEX' \
    --output-dir '$OUT' --mode preflight
"

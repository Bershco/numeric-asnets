#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/state_smoke_%j.txt

set -euo pipefail
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
OUT="$ROOT/state_smoke_${SLURM_JOB_ID}"
CHECKPOINT='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:54.901707/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905448-3c74d66/snapshots/snapshot_161_0.500'
mkdir -p "$OUT/captures"
test "$(sha256sum "$CHECKPOINT/weights.joblib" | awk '{print $1}')" = efa1b6a7b8bad82f1ee52e5ccc49043bda2eff2d3e0c4f028133996850caecb2

apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  cd '$REPO/asnets'
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  export ASN_VH_STATE_CAPTURE_DIR='$OUT/captures'
  export ASN_VH_STATE_CAPTURE_ACTOR=policy
  export ASN_VH_STATE_CAPTURE_SOURCE=stage1_on_policy
  export ASN_VH_STATE_CAPTURE_SEED=534933708
  export ASN_VH_STATE_CAPTURE_VALIDATE_ROUNDTRIP=1
  ./run_experiment experiments_numeric.architecture_2.drone \
    experiments_numeric.domain.value_head_v1_drone_smoke \
    --resume-from '$CHECKPOINT' --random-seed 534933607 \
    --num-workers 1 --jpddl-max-heap 4g
  python - <<'PY'
import json, pathlib
from asnets.value_head_audit import validate_canonical_state_record
paths = list(pathlib.Path('$OUT/captures').glob('*.jsonl'))
assert len(paths) == 1, paths
records = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
assert records, 'no captured states'
for record in records:
    validate_canonical_state_record(record)
print(f'ROUNDTRIP_SMOKE_OK|states={len(records)}|file={paths[0]}')
PY
"

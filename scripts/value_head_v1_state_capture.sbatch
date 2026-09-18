#!/bin/bash
# Submit with --array=0-1 and DOMAIN plus domain-appropriate --mem/--time.
#SBATCH --partition=main
#SBATCH --cpus-per-task=5
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/capture_%x_%A_%a.txt

set -euo pipefail
: "${DOMAIN:?submit with --export=ALL,DOMAIN=<domain>}"
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
case "$SLURM_ARRAY_TASK_ID" in
  0) SEED=534933607 ;;
  1) SEED=923500475 ;;
  *) exit 2 ;;
esac
case "$DOMAIN" in
  drone) ARCH=drone; MODULE=value_head_v1_drone; COMMON_SEED=31001; RANDOM_SEED=31002 ;;
  fo_counters) ARCH=fo_counters; MODULE=value_head_v1_fo_counters; COMMON_SEED=32001; RANDOM_SEED=32002 ;;
  rover) ARCH=rover; MODULE=value_head_v1_rover; COMMON_SEED=33001; RANDOM_SEED=33002 ;;
  mprime) ARCH=mprime; MODULE=value_head_v1_mprime; COMMON_SEED=34001; RANDOM_SEED=34002 ;;
  *) echo "unsupported domain $DOMAIN" >&2; exit 2 ;;
esac
OUT="$ROOT/captures/${DOMAIN}_${SEED}"
mkdir -p "$OUT"

mapfile -t CP < <(python3 - "$REPO/experiment_tracking/value_head_quality_audit/v1_checkpoint_candidates.csv" "$DOMAIN" "$SEED" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as stream:
    rows = [r for r in csv.DictReader(stream) if r['domain'] == sys.argv[2] and r['seed'] == sys.argv[3]]
for stage in ('stage1', 'stage2'):
    row = next(r for r in rows if r['stage'] == stage)
    print(row['checkpoint_path'])
    print(row['checkpoint_sha256'])
PY
)
STAGE1=${CP[0]}; STAGE1_SHA=${CP[1]}; STAGE2=${CP[2]}; STAGE2_SHA=${CP[3]}
test "$(sha256sum "$STAGE1/weights.joblib" | awk '{print $1}')" = "$STAGE1_SHA"
test "$(sha256sum "$STAGE2/weights.joblib" | awk '{print $1}')" = "$STAGE2_SHA"

run_capture() {
  local actor=$1 source=$2 capture_seed=$3 checkpoint=$4
  local capture_dir="$OUT/$source"
  mkdir -p "$capture_dir"
  apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
    export ASN_VH_STATE_CAPTURE_DIR='$capture_dir'
    export ASN_VH_STATE_CAPTURE_ACTOR='$actor'
    export ASN_VH_STATE_CAPTURE_SOURCE='$source'
    export ASN_VH_STATE_CAPTURE_SEED='$capture_seed'
    export ASN_VH_STATE_CAPTURE_VALIDATE_ROUNDTRIP=1
    export ASN_VH_ENHSP_TIMEOUT=300
    ./run_experiment 'experiments_numeric.architecture_2.$ARCH' \
      'experiments_numeric.domain.$MODULE' --resume-from '$checkpoint' \
      --random-seed '$SEED' --num-workers 5 --jpddl-max-heap 8g
  " >"$OUT/${source}.stdout" 2>&1
}

run_capture enhsp common_planner "$COMMON_SEED" "$STAGE1"
run_capture random common_random_legal "$RANDOM_SEED" "$STAGE1"
run_capture policy stage1_on_policy "$((SEED + 101))" "$STAGE1"
run_capture policy stage2_on_policy "$((SEED + 102))" "$STAGE2"

MANIFEST="$OUT/v1_states/vhv1-${DOMAIN}-${SEED}-paired.jsonl"
apptainer exec --bind /home/hersco:/home/hersco /home/hersco/Docker/image.sif /bin/bash -lc "
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  python '$REPO/scripts/materialize_value_head_v1_state_manifest.py' \
    --domain '$DOMAIN' --seed '$SEED' --capture-root '$OUT' --output '$MANIFEST'
"

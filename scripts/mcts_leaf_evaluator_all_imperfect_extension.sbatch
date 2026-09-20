#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --exclude=ise-cpu-intl-[01,05,08-15,18,24-28]

set -euo pipefail
set -o pipefail

FEATURE_REPO=${FEATURE_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe}
PRODUCTION_REPO=${PRODUCTION_REPO:-/home/hersco/bershco-nu-asnets/numeric-asnets}
MANIFEST=${MANIFEST:-$FEATURE_REPO/experiment_tracking/mcts_leaf_evaluator_all_imperfect_20260920/manifest.csv}
OUT=${OUT:-/home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_all_imperfect}
CONTAINER=${CONTAINER:-/home/hersco/Docker/image.sif}
VALIDATOR=${VALIDATOR:-/home/hersco/tools/VAL/build/bin/Validate}
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
SMOKE=${SMOKE:-0}
COMPATIBILITY_MARKER=${COMPATIBILITY_MARKER:-$OUT/compatibility.json}

mapfile -d '' -t ROW < <(python - "$MANIFEST" "$TASK_ID" <<'PY'
import csv
import sys

manifest, index = sys.argv[1], int(sys.argv[2])
with open(manifest, newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
if index < 0 or index >= len(rows):
    raise SystemExit(f"task index {index} outside 0..{len(rows)-1}")
row = rows[index]
fields = (
    'manifest_id', 'domain', 'seed', 'source_checkpoint', 'leaf_evaluator',
    'use_estimator', 'rollout_horizon', 'width', 'iterations', 'puct',
    'teacher', 'workers', 'instance_timeout_seconds', 'code_commit',
)
for field in fields:
    sys.stdout.write(row[field])
    sys.stdout.write('\0')
PY
)

MANIFEST_ID=${ROW[0]}
DOMAIN=${ROW[1]}
SEED=${ROW[2]}
CHECKPOINT=${ROW[3]}
LEAF_EVALUATOR=${ROW[4]}
USE_ESTIMATOR=${ROW[5]}
ROLLOUT_HORIZON=${ROW[6]}
WIDTH=${ROW[7]}
ITERATIONS=${ROW[8]}
PUCT=${ROW[9]}
TEACHER=${ROW[10]}
WORKERS=${ROW[11]}
INSTANCE_TIMEOUT=${ROW[12]}
EXPECTED_COMMIT=${ROW[13]}

[[ -d "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
ACTUAL_COMMIT=$(git -C "$FEATURE_REPO" rev-parse HEAD)
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
  echo "checkout commit $ACTUAL_COMMIT does not match frozen $EXPECTED_COMMIT" >&2
  exit 2
}
if [[ "$LEAF_EVALUATOR" == policy_rollout && "$USE_ESTIMATOR" != 0.0 ]]; then
  echo "rollout arm must not carry an estimator coefficient" >&2
  exit 2
fi

MODE=science
EXTRA_ARGS=()
EVAL_MAX_ACTIONS=10000
if [[ "$SMOKE" == 1 ]]; then
  MODE=smoke
  WORKERS=1
  INSTANCE_TIMEOUT=600
  EVAL_MAX_ACTIONS=100
  EXTRA_ARGS+=(--restrict-test-probs 0)
elif [[ ! -s "$COMPATIBILITY_MARKER" ]]; then
  echo "science release requires compatibility marker: $COMPATIBILITY_MARKER" >&2
  exit 2
fi

RUN_OUT="$OUT/$MODE"
mkdir -p "$RUN_OUT/completion" "$RUN_OUT/done"
LOG="$RUN_OUT/${SLURM_JOB_ID}_${MANIFEST_ID}.txt"
COMPLETION="$RUN_OUT/completion/${MANIFEST_ID}.jsonl"
CHECKPOINT_SHA256=$(find "$CHECKPOINT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')

LEAF_ARGS=(--eval-mcts-leaf-evaluator "$LEAF_EVALUATOR" --mcts-rollout-horizon "$ROLLOUT_HORIZON")
if [[ "$USE_ESTIMATOR" != 0.0 ]]; then
  LEAF_ARGS+=(--use-estimator "$USE_ESTIMATOR")
fi

apptainer exec --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  "$CONTAINER" /bin/bash -lc "
    set -o pipefail
    cd '$FEATURE_REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$FEATURE_REPO':\${PYTHONPATH:-}
    export ENHSP_CONFIG_OVERRIDE='$TEACHER'
    ./run_experiment experiments_numeric.architecture_2.${DOMAIN}_mcts \
      experiments_numeric.domain.${DOMAIN} --resume-from '$CHECKPOINT' \
      --eval-with-mcts --mcts-expansion-size '$WIDTH' \
      --mcts-iterations '$ITERATIONS' --mcts-exploration-weight '$PUCT' \
      --eval-scheduling rolling --eval-completion-file '$COMPLETION' \
      --eval-instance-timeout '$INSTANCE_TIMEOUT' --eval-max-actions '$EVAL_MAX_ACTIONS' \
      --num-workers '$WORKERS' \
      --jpddl-max-heap 4g --worker-logs --random-seed '$SEED' \
      ${LEAF_ARGS[*]} ${EXTRA_ARGS[*]}
  " 2>&1 | tee "$LOG"

python "$PRODUCTION_REPO/asnets/tools/validate_eval_log.py" \
  --log "$LOG" --domain "$DOMAIN" --validator "$VALIDATOR"

[[ -s "$COMPLETION" ]] || {
  echo "missing or empty completion ledger: $COMPLETION" >&2
  exit 2
}
grep -q "leaf_evaluator=$LEAF_EVALUATOR" "$LOG"
python - "$RUN_OUT/done/${MANIFEST_ID}.json" <<PY
import json
import os
import sys
payload = {
    "manifest_id": "$MANIFEST_ID",
    "mode": "$MODE",
    "checkpoint_sha256": "$CHECKPOINT_SHA256",
    "leaf_evaluator": "$LEAF_EVALUATOR",
    "use_estimator": float("$USE_ESTIMATOR"),
    "width": int("$WIDTH"),
    "iterations": int("$ITERATIONS"),
    "code_commit": "$ACTUAL_COMMIT",
    "job_id": os.environ["SLURM_JOB_ID"],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY

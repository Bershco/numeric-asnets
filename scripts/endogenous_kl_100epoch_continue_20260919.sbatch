#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/continue_%A_%a.txt

set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"; : "${CAMPAIGN_ROOT:?}"
MANIFEST=${RECOVERY_MANIFEST:-$CAMPAIGN_ROOT/recovery_manifest.csv}
mapfile -t F < <(python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
row=next(r for r in rows if int(r['arm_index']) == int(sys.argv[2]))
for key in ('domain','seed','semantics','training_module','teacher','source_checkpoint',
            'source_checkpoint_sha256','start_epoch','remaining_epochs','segment'):
    print(row[key])
PY
)
DOMAIN=${F[0]}; SEED=${F[1]}; SEMANTICS=${F[2]}; TRAIN_MOD=${F[3]}; HEUR=${F[4]}
SOURCE=${F[5]}; EXPECTED_SHA=${F[6]}; START_EPOCH=${F[7]}; REMAINING=${F[8]}; SEGMENT=${F[9]}
ARM=$SLURM_ARRAY_TASK_ID
OUT="$CAMPAIGN_ROOT/outputs/arm_${ARM}_${DOMAIN}_${SEED}_${SEMANTICS}"
SEG="$OUT/continuations/$SEGMENT"
mkdir -p "$SEG"
test ! -f "$OUT/training_complete.json"
test ! -f "$SEG/complete.json"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$CODE_COMMIT"
test "$(sha256sum "$SOURCE/weights.joblib" | awk '{print $1}')" = "$EXPECTED_SHA"

python3 - "$SEG/plan.json" "$ARM" "$START_EPOCH" "$REMAINING" "$SOURCE" "$EXPECTED_SHA" <<'PY'
import json,pathlib,sys
path,arm,start,remaining,source,sha=sys.argv[1:]
payload=dict(arm_index=int(arm),start_epoch=int(start),remaining_epochs=int(remaining),
             source_checkpoint=source,source_checkpoint_sha256=sha)
p=pathlib.Path(path)
if p.exists() and json.loads(p.read_text()) != payload:
    raise SystemExit('continuation plan changed')
p.write_text(json.dumps(payload,sort_keys=True)+'\n')
PY

KL_FLAG=()
[[ "$SEMANTICS" == deterministic_current ]] && KL_FLAG=(--policy-anchor-kl-deterministic-current)
export ASN_FIRST_UPDATE_AUDIT_PATH="$SEG/first_update_audit.jsonl"
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
set +e
(
  apptainer exec --bind /home/hersco:/home/hersco \
    --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
    /home/hersco/Docker/image.sif /bin/bash -lc "
      set -euo pipefail; cd '$CHECKOUT/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate';
      export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-};
      export ASN_FIRST_UPDATE_AUDIT_PATH='$ASN_FIRST_UPDATE_AUDIT_PATH';
      ./run_experiment '$TRAIN_MOD' experiments_numeric.domain.'$DOMAIN' \\
        --resume-from '$SOURCE' --resume-train --no-eval --original-training-set --disable-value-head \\
        --random-seed '$SEED' --worker-logs --num-workers 3 --jpddl-max-heap 4g \\
        --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 --override-enhsp-config '$HEUR' \\
        --mcts-expansion-size 20 --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 \\
        --max-opt-epochs '$REMAINING' --supervised-lr 0.0003 --policy-anchor-kl-coeff 3 ${KL_FLAG[*]}
  " 2>&1 | tee "$SEG/training.stdout"
) &
runner=$!
while kill -0 "$runner" 2>/dev/null; do
  root=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "$SEG/training.stdout" 2>/dev/null || true)
  if test -n "$root"; then
    printf '%s\n' "$root" > "$SEG/snapshot_root.txt.tmp"
    mv "$SEG/snapshot_root.txt.tmp" "$SEG/snapshot_root.txt"
    break
  fi
  sleep 5
done
wait "$runner"; status=$?
set -e
test "$status" = 0
root=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "$SEG/training.stdout")
test -d "$root"
printf '%s\n' "$root" > "$SEG/snapshot_root.txt"
test -f "$root/snapshot_$((REMAINING-1))_"*/weights.joblib

python3 - "$SEG/complete.json" "$OUT/training_complete.json" "$ARM" "$START_EPOCH" "$REMAINING" "$root" <<'PY'
import json,pathlib,sys
segment,overall,arm,start,remaining,root=sys.argv[1:]
payload=dict(arm_index=int(arm),start_epoch=int(start),remaining_epochs=int(remaining),
             final_epoch=int(start)+int(remaining)-1,snapshot_root=root)
pathlib.Path(segment).write_text(json.dumps(payload,sort_keys=True)+'\n')
assert payload['final_epoch'] == 99, payload
pathlib.Path(overall).write_text(json.dumps(payload,sort_keys=True)+'\n')
PY


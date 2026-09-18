#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=18:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-7
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/train_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"; : "${CAMPAIGN_ROOT:?}"
test -f "$CAMPAIGN_ROOT/smoke_passed"
test "$(git -C "$CHECKOUT" rev-parse --short=8 HEAD)" = "${CODE_COMMIT:0:8}"
MANIFEST="$CAMPAIGN_ROOT/manifest.csv"
mapfile -t F < <(python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[int(sys.argv[2])]
for k in ('domain','seed','semantics','source_checkpoint','source_checkpoint_sha256','training_module','teacher'): print(r[k])
PY
)
DOMAIN=${F[0]}; SEED=${F[1]}; SEMANTICS=${F[2]}; SOURCE=${F[3]}; EXPECTED_SHA=${F[4]}; TRAIN_MOD=${F[5]}; HEUR=${F[6]}
test "$(sha256sum "$SOURCE/weights.joblib" | awk '{print $1}')" = "$EXPECTED_SHA"
OUT="$CAMPAIGN_ROOT/outputs/arm_${SLURM_ARRAY_TASK_ID}_${DOMAIN}_${SEED}_${SEMANTICS}"
mkdir -p "$OUT"
test ! -f "$OUT/training_complete.json"
KL_FLAG=(); [[ "$SEMANTICS" == deterministic_current ]] && KL_FLAG=(--policy-anchor-kl-deterministic-current)
printf '%q ' "${KL_FLAG[@]}" > "$OUT/kl_flag.txt"
export ASN_FIRST_UPDATE_AUDIT_PATH="$OUT/first_update_audit.jsonl"
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
 set -euo pipefail; cd '$CHECKOUT/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate';
 export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-}; export ASN_FIRST_UPDATE_AUDIT_PATH='$ASN_FIRST_UPDATE_AUDIT_PATH';
 ./run_experiment '$TRAIN_MOD' experiments_numeric.domain.'$DOMAIN' \
   --resume-from '$SOURCE' --resume-train --no-eval --original-training-set --disable-value-head \
   --random-seed '$SEED' --worker-logs --num-workers 3 --jpddl-max-heap 4g \
   --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 --override-enhsp-config '$HEUR' \
   --mcts-expansion-size 20 --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 \
   --max-opt-epochs 100 --supervised-lr 0.0003 --policy-anchor-kl-coeff 3 ${KL_FLAG[*]}
" 2>&1 | tee "$OUT/training.stdout"
SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "$OUT/training.stdout")
test -d "$SNAPSHOT_DIR"; printf '%s\n' "$SNAPSHOT_DIR" > "$OUT/snapshot_root.txt"
python3 - "$OUT/training_complete.json" "$SLURM_ARRAY_TASK_ID" "$DOMAIN" "$SEED" "$SEMANTICS" "$SOURCE" "$EXPECTED_SHA" "$SNAPSHOT_DIR" <<'PY'
import json,pathlib,sys
out,idx,domain,seed,sem,source,sha,snap=sys.argv[1:]
epochs=[0,*range(5,100,5),99]
for e in epochs:
 assert list(pathlib.Path(snap).glob(f'snapshot_{e}_*/weights.joblib')), (snap,e)
pathlib.Path(out).write_text(json.dumps(dict(arm_index=int(idx),domain=domain,seed=int(seed),semantics=sem,source_checkpoint=source,source_checkpoint_sha256=sha,snapshot_root=snap,epochs=epochs),sort_keys=True)+'\n')
PY

#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-167
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/policy_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"; : "${CAMPAIGN_ROOT:?}"
POLICY_MANIFEST=${POLICY_MANIFEST:-$CAMPAIGN_ROOT/policy_eval_manifest.csv}
mapfile -t F < <(python3 - "$POLICY_MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
m=[r for r in rows if int(r['curve_index'])==int(sys.argv[2])]
assert len(m)==1,(sys.argv[2],len(m)); r=m[0]
for k in ('identity','arm_index','domain','seed','semantics','epoch','validation_score','total','evaluation_module','source_checkpoint','source_checkpoint_sha256','checkpoint','checkpoint_sha256','code_commit'): print(r[k])
PY
)
IDENTITY=${F[0]}; ARM=${F[1]}; DOMAIN=${F[2]}; SEED=${F[3]}; SEMANTICS=${F[4]}; EPOCH=${F[5]}; VALIDATION_SCORE=${F[6]}; TOTAL=${F[7]}; EVAL_MOD=${F[8]}; SOURCE=${F[9]}; SOURCE_SHA=${F[10]}; CHECKPOINT=${F[11]}; CHECKPOINT_SHA=${F[12]}; DECLARED_COMMIT=${F[13]}
test "$DECLARED_COMMIT" = "$CODE_COMMIT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$CODE_COMMIT"
test -f "$CHECKPOINT/weights.joblib"
test "$(sha256sum "$CHECKPOINT/weights.joblib" | awk '{print $1}')" = "$CHECKPOINT_SHA"
TRAIN_OUT="$CAMPAIGN_ROOT/outputs/arm_${ARM}_${DOMAIN}_${SEED}_${SEMANTICS}"
OUT="$TRAIN_OUT/policy_epoch_$(printf '%04d' "$EPOCH")"; mkdir -p "$OUT"
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
 set -euo pipefail; cd '$CHECKOUT/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate';
 export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-};
 ./run_experiment '$EVAL_MOD' experiments_numeric.domain.'$DOMAIN' --resume-from '$CHECKPOINT' \
   --disable-value-head --random-seed '$SEED' --num-workers 5 --jpddl-max-heap 4g
" 2>&1 | tee "$OUT/evaluation.stdout"
python3 - "$OUT/result.json" "$OUT/evaluation.stdout" "$IDENTITY" "$ARM" "$DOMAIN" "$SEED" "$SEMANTICS" "$EPOCH" "$VALIDATION_SCORE" "$TOTAL" "$SOURCE" "$SOURCE_SHA" "$CHECKPOINT" "$CHECKPOINT_SHA" "$CODE_COMMIT" <<'PY'
import json,pathlib,re,sys
out,log,identity,arm,domain,seed,sem,epoch,val_score,total,source,source_sha,checkpoint,checkpoint_sha,commit=sys.argv[1:]
m=re.findall(r'\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9.]+)',pathlib.Path(log).read_text(errors='replace'))
assert m, log
score,seen=map(lambda x:int(float(x)),m[-1]); assert seen==int(total)
pathlib.Path(out).write_text(json.dumps(dict(identity=identity,arm_index=int(arm),domain=domain,seed=int(seed),semantics=sem,epoch=int(epoch),validation_score=float(val_score),score=score,seen=seen,total=int(total),source_checkpoint=source,source_checkpoint_sha256=source_sha,checkpoint=checkpoint,checkpoint_sha256=checkpoint_sha,code_commit=commit,evaluation_log=log),sort_keys=True)+'\n')
PY

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
MANIFEST="$CAMPAIGN_ROOT/manifest.csv"; EPOCHS=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 99)
ARM=$((SLURM_ARRAY_TASK_ID / 21)); EPOCH=${EPOCHS[$((SLURM_ARRAY_TASK_ID % 21))]}
mapfile -t F < <(python3 - "$MANIFEST" "$ARM" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[int(sys.argv[2])]
for k in ('domain','seed','semantics','total','evaluation_module','source_checkpoint','source_checkpoint_sha256'): print(r[k])
PY
)
DOMAIN=${F[0]}; SEED=${F[1]}; SEMANTICS=${F[2]}; TOTAL=${F[3]}; EVAL_MOD=${F[4]}; SOURCE=${F[5]}; SOURCE_SHA=${F[6]}
TRAIN_OUT="$CAMPAIGN_ROOT/outputs/arm_${ARM}_${DOMAIN}_${SEED}_${SEMANTICS}"; test -f "$TRAIN_OUT/training_complete.json"
SNAPSHOT_ROOT=$(cat "$TRAIN_OUT/snapshot_root.txt"); mapfile -t MATCHES < <(find "$SNAPSHOT_ROOT" -maxdepth 1 -type d -name "snapshot_${EPOCH}_*" | sort)
test "${#MATCHES[@]}" = 1; CHECKPOINT=${MATCHES[0]}; test -f "$CHECKPOINT/weights.joblib"
OUT="$TRAIN_OUT/policy_epoch_$(printf '%04d' "$EPOCH")"; mkdir -p "$OUT"
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
 set -euo pipefail; cd '$CHECKOUT/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate';
 export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-};
 ./run_experiment '$EVAL_MOD' experiments_numeric.domain.'$DOMAIN' --resume-from '$CHECKPOINT' \
   --disable-value-head --random-seed '$SEED' --num-workers 5 --jpddl-max-heap 4g
" 2>&1 | tee "$OUT/evaluation.stdout"
python3 - "$OUT/result.json" "$OUT/evaluation.stdout" "$ARM" "$DOMAIN" "$SEED" "$SEMANTICS" "$EPOCH" "$TOTAL" "$SOURCE" "$SOURCE_SHA" "$CHECKPOINT" <<'PY'
import json,pathlib,re,sys
out,log,arm,domain,seed,sem,epoch,total,source,sha,checkpoint=sys.argv[1:]
m=re.findall(r'\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9.]+)',pathlib.Path(log).read_text(errors='replace'))
assert m, log
score,seen=map(lambda x:int(float(x)),m[-1]); assert seen==int(total)
pathlib.Path(out).write_text(json.dumps(dict(arm_index=int(arm),domain=domain,seed=int(seed),semantics=sem,epoch=int(epoch),score=score,total=int(total),source_checkpoint=source,source_checkpoint_sha256=sha,checkpoint=checkpoint,evaluation_log=log),sort_keys=True)+'\n')
PY

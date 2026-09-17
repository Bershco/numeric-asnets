#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-95
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover/endpoint_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover; MANIFEST=$ROOT/new_pairs_verified.csv
PAIR=$((SLURM_ARRAY_TASK_ID / 2)); MODE=$((SLURM_ARRAY_TASK_ID % 2))
mapfile -t FIELDS < <(python3 - "$MANIFEST" "$PAIR" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[int(sys.argv[2])]
for k in ('lineage_index','domain','seed','total'): print(r[k])
PY
)
LINEAGE=${FIELDS[0]}; DOMAIN=${FIELDS[1]}; SEED=${FIELDS[2]}; TOTAL=${FIELDS[3]}
case "$DOMAIN" in
 block_grouping) EVAL_MOD=experiments_numeric.architecture_2.block_grouping ;;
 drone) EVAL_MOD=experiments_numeric.architecture_2.drone ;;
 fo_counters) EVAL_MOD=experiments_numeric.architecture_2.fo_counters ;;
 rover) EVAL_MOD=experiments_numeric.architecture_2.rover ;;
 counters) EVAL_MOD=experiments_numeric.architecture_2.counters ;;
 *) exit 2 ;;
esac
if (( MODE==0 )); then TREATMENT=legacy_capture; else TREATMENT=deterministic_current; fi
OUT="$ROOT/outputs/lineage_${LINEAGE}_${DOMAIN}_${SEED}_${TREATMENT}"; CHECKPOINT=$(cat "$OUT/endpoint_path.txt"); test -f "$CHECKPOINT/weights.joblib"
LOG="$OUT/policy_endpoint.stdout"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "set -euo pipefail; cd '$CHECKOUT/asnets'; source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate; export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-}; ./run_experiment '$EVAL_MOD' experiments_numeric.domain.'$DOMAIN' --resume-from '$CHECKPOINT' --disable-value-head --random-seed '$SEED' --num-workers 5 --jpddl-max-heap 4g" 2>&1 | tee "$LOG"
python3 - "$LOG" "$OUT/endpoint_result.json" "$LINEAGE" "$DOMAIN" "$SEED" "$TREATMENT" "$TOTAL" <<'PY'
import json,pathlib,re,sys
log,out,lineage,domain,seed,treatment,total=sys.argv[1:]
m=re.findall(r'\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9.]+)',pathlib.Path(log).read_text(errors='replace'))
assert m, f'no final score in {log}'
score,seen=m[-1]; assert int(float(seen))==int(total)
pathlib.Path(out).write_text(json.dumps(dict(lineage_index=int(lineage),domain=domain,seed=int(seed),treatment=treatment,score=int(float(score)),total=int(total),log=log),sort_keys=True)+'\n')
PY

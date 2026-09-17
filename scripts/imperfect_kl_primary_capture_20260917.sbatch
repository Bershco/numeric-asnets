#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-47
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover/capture_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
MANIFEST=$ROOT/new_pairs_verified.csv
mapfile -t FIELDS < <(python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
r=rows[int(sys.argv[2])]
for k in ('lineage_index','domain','seed','source_checkpoint','source_checkpoint_sha256','optimizer_rng_base_seed'):
 print(r[k])
PY
)
LINEAGE=${FIELDS[0]}; DOMAIN=${FIELDS[1]}; SEED=${FIELDS[2]}; SOURCE=${FIELDS[3]}; EXPECTED_SHA=${FIELDS[4]}; RNG=${FIELDS[5]}
case "$DOMAIN" in
 block_grouping) TRAIN_MOD=experiments_numeric.architecture_2.block_grouping_mcts; HEUR=hadd-gbfs ;;
 drone) TRAIN_MOD=experiments_numeric.architecture_2.drone_mcts; HEUR=hadd-astar ;;
 fo_counters) TRAIN_MOD=experiments_numeric.architecture_2.fo_counters_mcts; HEUR=hmrmax-astar ;;
 rover) TRAIN_MOD=experiments_numeric.architecture_2.rover_mcts; HEUR=hmrp-ha-gbfs ;;
 counters) TRAIN_MOD=experiments_numeric.architecture_2.counters_mcts; HEUR=hmrmax-astar ;;
 *) exit 2 ;;
esac
WEIGHTS="$SOURCE/weights.joblib"; test -f "$WEIGHTS"; test "$(sha256sum "$WEIGHTS" | awk '{print $1}')" = "$EXPECTED_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"
OUT="$ROOT/outputs/lineage_${LINEAGE}_${DOMAIN}_${SEED}_legacy_capture"
mkdir -p "$OUT"
export ASN_FIRST_UPDATE_AUDIT_PATH="$OUT/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="$OUT/optimizer_step_weights"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
 set -euo pipefail; cd '$REPO/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate'
 export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
 export ASN_FIRST_UPDATE_AUDIT_PATH='$ASN_FIRST_UPDATE_AUDIT_PATH'; export ASN_FIRST_UPDATE_WEIGHT_DIR='$ASN_FIRST_UPDATE_WEIGHT_DIR'
 ./run_experiment '$TRAIN_MOD' experiments_numeric.domain.'$DOMAIN' --resume-from '$SOURCE' --resume-train --no-eval --original-training-set --disable-value-head --random-seed '$SEED' --num-workers 3 --jpddl-max-heap 4g --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 --override-enhsp-config '$HEUR' --mcts-expansion-size 20 --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 --max-opt-epochs 1 --supervised-lr 0.0003 --policy-anchor-kl-coeff 3 --policy-anchor-fixed-step-rng-seed '$RNG'
" 2>&1 | tee "$OUT/training.stdout"
SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "$OUT/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "$ENDPOINT/weights.joblib"; printf '%s\n' "$ENDPOINT" > "$OUT/endpoint_path.txt"
test "$(find "$ASN_FIRST_UPDATE_AUDIT_PATH.batches" -maxdepth 1 -type f -name 'optimizer_step_*.npz' | wc -l)" = 60

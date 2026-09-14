#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-1
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a/train_%A_%a.txt

set -euo pipefail

REPO=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ROLE=catastrophic_outlier
    SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    ;;
  1)
    ROLE=stable_control
    SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    ;;
  *) exit 2 ;;
esac

OUT="${ROOT}/${ROLE}_${SEED}"
mkdir -p "$OUT"
test -f "${SOURCE}/weights.joblib"

export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"

echo "[TPP-FIRST-UPDATE-A] commit=$(git -C "$REPO" rev-parse HEAD)"
echo "[TPP-FIRST-UPDATE-A] role=$ROLE seed=$SEED source=$SOURCE"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
    export ASN_FIRST_UPDATE_AUDIT_PATH='$ASN_FIRST_UPDATE_AUDIT_PATH'
    ./run_experiment \
      experiments_numeric.architecture_2.tpp_mcts \
      experiments_numeric.domain.tpp \
      --resume-from '$SOURCE' \
      --resume-train \
      --no-eval \
      --original-training-set \
      --disable-value-head \
      --random-seed '$SEED' \
      --worker-logs \
      --num-workers 3 \
      --jpddl-max-heap 4g \
      --override-mse-coeff 0.5 \
      --mcts-exploration-weight 0.1 \
      --override-enhsp-config hadd-astar \
      --mcts-expansion-size 20 \
      --mcts-iterations 0 \
      --sample-k-additional-states 0 \
      --use-estimator 0.5 \
      --max-opt-epochs 1 \
      --supervised-lr 0.0003 \
      --policy-anchor-kl-coeff 3
  " 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"

python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
steps = [r for r in records if r.get("record_type") == "optimizer_step"]
schedule = [r for r in records if r.get("record_type") == "worker_schedule"]
assert len(steps) == 60, len(steps)
assert len(schedule) == 1, len(schedule)
assert all(pathlib.Path(r["frozen_batch_file"]).is_file() for r in steps)
print(f"[TPP-FIRST-UPDATE-A] audit verified: {len(steps)} optimizer steps")
PY

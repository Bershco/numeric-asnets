#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-1
#SBATCH --exclude=ise-cpu-intl-13
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/train_%A_%a.txt

set -euo pipefail

REPO=${CHECKOUT:-/home/hersco/bershco-nu-asnets/tpp-first-update-b-9732f3a5}
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
CHECKSUMS="${ROOT}/frozen_schedule_checksums.csv"

python3 "${ROOT}/freeze_tpp_phase_a_batch_checksums.py" \
  --verify --manifest "$CHECKSUMS"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=bad_checkpoint_x_stable_replay
    SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
    ;;
  1)
    ARM=stable_checkpoint_x_bad_replay
    SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    FROZEN_BATCHES="${PHASE_A}/catastrophic_outlier_1972442430/first_update_audit.jsonl.batches"
    ;;
  *) exit 2 ;;
esac

OUT="${ROOT}/${ARM}"
mkdir -p "$OUT"
test -f "${SOURCE}/weights.joblib"
export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="${OUT}/optimizer_step_weights"

echo "[TPP-FIRST-UPDATE-B] commit=$(git -C "$REPO" rev-parse HEAD)"
echo "[TPP-FIRST-UPDATE-B] arm=$ARM seed=$SEED source=$SOURCE frozen_batches=$FROZEN_BATCHES"

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
    export ASN_FIRST_UPDATE_WEIGHT_DIR='$ASN_FIRST_UPDATE_WEIGHT_DIR'
    ./run_experiment \
      experiments_numeric.architecture_2.tpp_mcts \
      experiments_numeric.domain.tpp \
      --resume-from '$SOURCE' \
      --resume-train \
      --no-eval \
      --original-training-set \
      --disable-value-head \
      --random-seed '$SEED' \
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
      --policy-anchor-kl-coeff 3 \
      --frozen-replay-batch-dir '$FROZEN_BATCHES'
  " 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"

python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" "$FROZEN_BATCHES" <<'PY'
import json, pathlib, sys
audit = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).resolve()
records = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
starts = [row for row in records if row.get("record_type") == "frozen_replay_start"]
steps = [row for row in records if row.get("record_type") == "optimizer_step"]
weight_rows = [row for row in records if row.get("record_type") == "optimizer_step_weights"]
ends = [row for row in records if row.get("record_type") == "frozen_replay_complete"]
assert len(starts) == 1 and pathlib.Path(starts[0]["source_directory"]) == source
assert len(steps) == 60 and len(ends) == 1 and ends[0]["updates"] == 60
assert len(weight_rows) == 60
assert all(pathlib.Path(row["path"]).is_file() for row in weight_rows)
assert all(pathlib.Path(row["source_frozen_batch_file"]).parent.resolve() == source for row in steps)
assert all(row["source_frozen_batch_sha256"] for row in steps)
print("[TPP-FIRST-UPDATE-B] exact frozen 60-step replay verified")
PY

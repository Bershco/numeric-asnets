#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-1
#SBATCH --exclude=ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt/train_%A_%a.txt

set -euo pipefail

: "${CHECKOUT:?Set CHECKOUT to the deployed Phase-C checkout}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
CHECKSUMS=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv
THRESHOLDS="${ROOT}/calibrated_thresholds.json"

python3 "${ROOT}/freeze_tpp_phase_a_batch_checksums.py" \
  --verify --manifest "$CHECKSUMS"
read -r MEAN_KL_LIMIT P99_KL_LIMIT < <(python3 - "$THRESHOLDS" <<'PY'
import json, pathlib, sys
row = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert row["calibration_role"] == "stable_control" and row["calibration_steps"] == 60
print(row["mean_kl_limit"], row["p99_kl_limit"])
PY
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=catastrophic_outlier
    SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    FROZEN_BATCHES="${PHASE_A}/catastrophic_outlier_1972442430/first_update_audit.jsonl.batches"
    ;;
  1)
    ARM=stable_control
    SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
    ;;
  *) exit 2 ;;
esac

OUT="${ROOT}/${ARM}"
mkdir -p "$OUT"
test -f "${SOURCE}/weights.joblib"
export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="${OUT}/optimizer_step_weights"

echo "[TPP-FIRST-UPDATE-C] commit=$(git -C "$REPO" rev-parse HEAD)"
echo "[TPP-FIRST-UPDATE-C] arm=$ARM seed=$SEED source=$SOURCE frozen_batches=$FROZEN_BATCHES"

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
      --policy-anchor-kl-deterministic-current \
      --policy-anchor-trust-mean-kl-limit '$MEAN_KL_LIMIT' \
      --policy-anchor-trust-p99-kl-limit '$P99_KL_LIMIT' \
      --policy-anchor-trust-max-retries 2 \
      --policy-anchor-trust-lr-factor 0.5 \
      --frozen-replay-batch-dir '$FROZEN_BATCHES'
  " 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"

python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" "$FROZEN_BATCHES" "$THRESHOLDS" <<'PY'
import json, pathlib, sys
audit = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).resolve()
limits = json.loads(pathlib.Path(sys.argv[3]).read_text())
rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
starts = [row for row in rows if row.get("record_type") == "frozen_replay_start"]
steps = [row for row in rows if row.get("record_type") == "optimizer_step"]
weights = [row for row in rows if row.get("record_type") == "optimizer_step_weights"]
ends = [row for row in rows if row.get("record_type") == "frozen_replay_complete"]
assert len(starts) == 1 and pathlib.Path(starts[0]["source_directory"]) == source
assert len(steps) == 60 and len(weights) == 60 and len(ends) == 1
assert ends[0]["updates"] == 60
for step in steps:
    assert step["kl_current_forward"] == "deterministic"
    assert step["replay_loss_forward"] == "training"
    assert step["trust_region"] == {
        "mean_kl_limit": limits["mean_kl_limit"],
        "p99_kl_limit": limits["p99_kl_limit"],
        "max_retries": 2,
        "learning_rate_factor": 0.5,
    }
    attempts = step["trust_region_attempts"]
    assert 1 <= len(attempts) <= 3
    assert step["trust_region_retries"] == len(attempts) - 1
    assert all(item["optimizer_iteration_after"] == item["optimizer_iteration_before"] + 1 for item in attempts)
    assert len({item["optimizer_iteration_before"] for item in attempts}) == 1
    assert all(item["excessive"] for item in attempts[:-1])
    assert all(item["coefficient"] == 3.0 for item in attempts)
    assert all(abs(item["learning_rate"] - 0.0003 * 0.5 ** index) < 1e-9 for index, item in enumerate(attempts))
    assert step["update_applied"] == (not attempts[-1]["excessive"])
    assert abs(step["base_learning_rate"] - 0.0003) < 1e-9
    assert step["applied_learning_rate"] == (attempts[-1]["learning_rate"] if step["update_applied"] else None)
    assert step["learning_rate_restored"]
    assert all(pathlib.Path(item["path"]).is_file() for item in weights)
    assert pathlib.Path(step["source_frozen_batch_file"]).parent.resolve() == source
    assert step["source_frozen_batch_sha256"]
assert ends[0]["accepted_updates"] == sum(step["update_applied"] for step in steps)
assert ends[0]["trust_region_retries"] == sum(step["trust_region_retries"] for step in steps)
print("[TPP-FIRST-UPDATE-C] deterministic KL and LR-backtracking audit verified")
PY

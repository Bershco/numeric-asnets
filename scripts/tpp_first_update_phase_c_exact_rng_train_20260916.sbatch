#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_c_exact_rng/train_%A_%a.txt

set -euo pipefail
: "${CHECKOUT:?Set CHECKOUT to the deployed exact-RNG checkout}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the reviewed exact-RNG commit}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_c_exact_rng
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
CHECKSUMS=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv
THRESHOLDS=/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt/calibrated_thresholds.json
EXACT_RNG_BASE_SEED=314159

test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"
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
    ARM=catastrophic_outlier_active
    TREATMENT=active_guard
    SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    FROZEN_BATCHES="${PHASE_A}/catastrophic_outlier_1972442430/first_update_audit.jsonl.batches"
    ;;
  1)
    ARM=stable_control_active
    TREATMENT=active_guard
    SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
    ;;
  2)
    ARM=catastrophic_outlier_inactive
    TREATMENT=inactive_guard
    SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    FROZEN_BATCHES="${PHASE_A}/catastrophic_outlier_1972442430/first_update_audit.jsonl.batches"
    MEAN_KL_LIMIT=1000000
    P99_KL_LIMIT=1000000
    ;;
  3)
    ARM=stable_control_inactive
    TREATMENT=inactive_guard
    SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
    MEAN_KL_LIMIT=1000000
    P99_KL_LIMIT=1000000
    ;;
  *) exit 2 ;;
esac

OUT="${ROOT}/${ARM}"
mkdir -p "$OUT"
test -f "${SOURCE}/weights.joblib"
export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="${OUT}/optimizer_step_weights"

echo "[TPP-EXACT-RNG] commit=$(git -C "$REPO" rev-parse HEAD)"
echo "[TPP-EXACT-RNG] arm=$ARM treatment=$TREATMENT seed=$SEED exact_rng_base_seed=$EXACT_RNG_BASE_SEED"

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
      --resume-from '$SOURCE' --resume-train --no-eval \
      --original-training-set --disable-value-head --random-seed '$SEED' \
      --num-workers 3 --jpddl-max-heap 4g \
      --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 \
      --override-enhsp-config hadd-astar --mcts-expansion-size 20 \
      --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 \
      --max-opt-epochs 1 --supervised-lr 0.0003 \
      --policy-anchor-kl-coeff 3 \
      --policy-anchor-kl-deterministic-current \
      --policy-anchor-trust-mean-kl-limit '$MEAN_KL_LIMIT' \
      --policy-anchor-trust-p99-kl-limit '$P99_KL_LIMIT' \
      --policy-anchor-trust-max-retries 2 \
      --policy-anchor-trust-lr-factor 0.5 \
      --policy-anchor-trust-restore-rng-seed '$EXACT_RNG_BASE_SEED' \
      --frozen-replay-batch-dir '$FROZEN_BATCHES'
  " 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"

python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" "$FROZEN_BATCHES" "$TREATMENT" "$SEED" <<'PY'
import json, pathlib, sys
audit = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).resolve()
treatment = sys.argv[3]
seed = int(sys.argv[4])
rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
steps = [row for row in rows if row.get("record_type") == "optimizer_step"]
ends = [row for row in rows if row.get("record_type") == "frozen_replay_complete"]
assert len(steps) == 60 and len(ends) == 1 and ends[0]["updates"] == 60
retry_steps = 0
for step_index, step in enumerate(steps):
    assert step["trust_region_exact_rng_base_seed"] == 314159
    assert step["trust_region_exact_rng_step_seed"] == (314159 + 1000003 * step_index) % 2147483647
    attempts = step["trust_region_attempts"]
    hashes = {attempt["raw_gradient_sha256"] for attempt in attempts}
    assert len(hashes) == 1
    assert hashes == {step["trust_region_exact_gradient_sha256"]}
    assert all(attempt["exact_rng_retry"] for attempt in attempts)
    assert all(attempt["coefficient"] == 3.0 for attempt in attempts)
    assert pathlib.Path(step["source_frozen_batch_file"]).parent.resolve() == source
    retry_steps += int(len(attempts) > 1)
if treatment == "active_guard" and seed == 1972442430:
    assert retry_steps > 0, "Catastrophic arm never exercised exact-RNG retry"
if treatment == "inactive_guard":
    assert retry_steps == 0, "Inactive guard unexpectedly retried an update"
print(f"[TPP-EXACT-RNG] verified 60 steps; retry_steps={retry_steps}")
PY

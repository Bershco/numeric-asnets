#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-1
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_d_legacy_exact_rng/train_%A_%a.txt

set -euo pipefail
: "${CHECKOUT:?Set CHECKOUT to the isolated diagnostic checkout}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the reviewed diagnostic commit}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_d_legacy_exact_rng
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
PHASE_C=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_c_exact_rng
CHECKSUMS=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv
FIXED_RNG_BASE_SEED=314159
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"

python3 "${ROOT}/freeze_tpp_phase_a_batch_checksums.py" \
  --verify --manifest "$CHECKSUMS"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=catastrophic_outlier_legacy_exact_rng
    SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    SOURCE_SHA=62c01cac5cbacb776e0c2b0e3d421d0314bb1abe282c14925c51602e2bfaecec
    FROZEN_BATCHES="${PHASE_A}/catastrophic_outlier_1972442430/first_update_audit.jsonl.batches"
    MATCHED_EXACT_AUDIT="${PHASE_C}/catastrophic_outlier_inactive/first_update_audit.jsonl"
    ;;
  1)
    ARM=stable_control_legacy_exact_rng
    SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    SOURCE_SHA=e97c179d214e0a84b824f584accdecf56373de6d92c353166ef5b2ed18b7b353
    FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
    MATCHED_EXACT_AUDIT="${PHASE_C}/stable_control_inactive/first_update_audit.jsonl"
    ;;
  *) exit 2 ;;
esac

OUT="${ROOT}/${ARM}"
mkdir -p "$OUT"
test -f "${SOURCE}/weights.joblib"
test "$(sha256sum "${SOURCE}/weights.joblib" | awk '{print $1}')" = "$SOURCE_SHA"
test -f "$MATCHED_EXACT_AUDIT"
export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="${OUT}/optimizer_step_weights"

echo "[TPP-PHASE-D] arm=$ARM seed=$SEED fixed_rng_base_seed=$FIXED_RNG_BASE_SEED kl_current_forward=training"

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
      --policy-anchor-fixed-step-rng-seed '$FIXED_RNG_BASE_SEED' \
      --frozen-replay-batch-dir '$FROZEN_BATCHES'
  " 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"

python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" "$MATCHED_EXACT_AUDIT" "$FROZEN_BATCHES" <<'PY'
import json, math, pathlib, sys

def steps(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()
            if line.strip() and json.loads(line).get("record_type") == "optimizer_step"]

def disagreement_bits(step):
    return [row["target_pred_argmax_disagree_pre"]
            for batch in step["batches"] for row in batch["rows"]]

legacy = steps(sys.argv[1])
matched = steps(sys.argv[2])
source = pathlib.Path(sys.argv[3]).resolve()
assert len(legacy) == len(matched) == 60
for index, (left, right) in enumerate(zip(legacy, matched)):
    assert left["kl_current_forward"] == "training"
    assert left["fixed_step_rng_base_seed"] == 314159
    assert left["fixed_step_rng_step_seed"] == (314159 + 1000003 * index) % 2147483647
    assert left["trust_region"] is None
    assert left["trust_region_retries"] == 0
    assert pathlib.Path(left["source_frozen_batch_file"]).parent.resolve() == source
    assert math.isclose(left["policy_gradient_l2"], right["policy_gradient_l2"],
                        rel_tol=1e-6, abs_tol=1e-8), (
        f"task-policy gradient mismatch at optimizer step {index}")
    assert disagreement_bits(left) == disagreement_bits(right), (
        f"task-policy argmax pattern mismatch at optimizer step {index}")
print("[TPP-PHASE-D] verified 60 legacy-KL steps with matched task-policy gradient norms and argmax patterns")
PY
